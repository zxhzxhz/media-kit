//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Low-latency 2x (or n x) super resolution via VideoToolbox's
//  VTFrameProcessor (iOS 26+): one source frame in, one upscaled frame out,
//  written into a caller supplied CVPixelBuffer.
//
//  Only spatial upscaling: no frame interpolation, no HDR, no Metal. The
//  processor forbids concurrent submissions per session, so calls are
//  serialised on a private queue; the caller owns all buffers.

import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

// VideoToolbox marks the whole VTFrameProcessor family as
// `#if !TARGET_OS_SIMULATOR`, so this file is device only.
#if !targetEnvironment(simulator)

  /// Failures surfaced through `VTFrameScaler.scale(...)`.
  enum VTFrameScalerError: Error {
    /// No session is running: never started, or `invalidate()` was called.
    case notReady
    /// A buffer does not match the configured frame size.
    case dimensionMismatch(
      expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int)
    /// `VTFrameProcessorFrame` rejected a buffer.
    case frameCreationFailed
    /// The scale factor is not in `supportedScaleFactors`.
    case unsupportedScaleFactor(Float)
    case scalingFailed(Error)
    /// `process` neither completed nor reported an error in time.
    case timedOut
    /// BGRA <-> the VideoToolbox format conversion failed.
    case formatConversionFailed
  }

  /// Availability probe callable from iOS 13 deployment targets.
  enum VTFrameScalingSupport {
    static func isSupported(
      width: Int,
      height: Int,
      scaleFactor: Float
    ) -> Bool {
      guard width > 0, height > 0, scaleFactor > 1.0 else {
        return false
      }
      if #available(iOS 26.0, *) {
        return VTFrameScalingSupport.probe(
          width: width,
          height: height,
          scaleFactor: scaleFactor
        )
      }
      return false
    }

    /// Human readable probe result, including the advertised scale factors.
    static func describe(width: Int, height: Int, scaleFactor: Float) -> String {
      guard width > 0, height > 0 else { return "invalid size" }
      if #available(iOS 26.0, *) {
        return VTFrameScalingSupport.describe26(
          width: width,
          height: height,
          scaleFactor: scaleFactor
        )
      }
      return "needs iOS 26"
    }

    @available(iOS 26.0, *)
    private static func probe(
      width: Int,
      height: Int,
      scaleFactor: Float
    ) -> Bool {
      // `isSupported` is the documented gate; a rejected size would otherwise
      // only surface as a `startSession` error.
      guard VTLowLatencySuperResolutionScalerConfiguration.isSupported else {
        return false
      }
      let configuration = VTLowLatencySuperResolutionScalerConfiguration(
        frameWidth: width,
        frameHeight: height,
        scaleFactor: scaleFactor
      )
      let processor = VTFrameProcessor()
      do {
        try processor.startSession(configuration: configuration)
      } catch {
        VTFrameInterpolationSupport.recordStartError(error)
        NSLog("VTFrameScaler: probe startSession failed: \(error)")
        return false
      }
      processor.endSession()
      return true
    }

    @available(iOS 26.0, *)
    private static func describe26(
      width: Int,
      height: Int,
      scaleFactor: Float
    ) -> String {
      var parts: [String] = []
      parts.append("scaleFactor \(scaleFactor)")
      parts.append(
        "processorSupported=\(VTLowLatencySuperResolutionScalerConfiguration.isSupported)"
      )

      let configuration = VTLowLatencySuperResolutionScalerConfiguration(
        frameWidth: width,
        frameHeight: height,
        scaleFactor: scaleFactor
      )
      // `frameSupportedPixelFormats` is NS_REFINED_FOR_SWIFT, so the ObjC name
      // is only reachable with the double underscore prefix in Swift.
      parts.append(
        "sourceFormats=\(String(describing: configuration.__frameSupportedPixelFormats))"
      )

      let error = VTFrameInterpolationSupport.lastStartError()
      if !error.isEmpty {
        parts.append(error)
      }
      return parts.joined(separator: "; ")
    }
  }

  /// Wraps a single `VTFrameProcessor` low-latency super resolution session.
  @available(iOS 26.0, *)
  final class VTFrameScaler {
    /// `process` must complete before the next submission: the processor
    /// forbids concurrent work, and a hung session would silently stop
    /// producing frames (which shows up as a black picture).
    private static let processTimeout = DispatchTimeInterval.milliseconds(250)

    private let frameWidth: Int
    private let frameHeight: Int
    private let outputWidth: Int
    private let outputHeight: Int
    private let scaleFactor: Float
    /// Serialises submissions: the processor forbids concurrent work.
    private let queue: DispatchQueue
    /// Guards `processor` for readers outside `queue`.
    private let lock = NSLock()
    private var processor: VTFrameProcessor?

    /// Pixel formats VideoToolbox advertises for this configuration.
    private(set) var supportedFormats: [OSType] = []
    /// `false` when the advertised formats are known and do not include BGRA.
    private(set) var acceptsBGRASource = true
    /// `true` when BGRA <-> VT format conversion is set up.
    private(set) var canConvertPixelFormat = false

    /// BGRA (source size) -> VT format.
    private var inputConverter: PixelBufferConverter?
    /// VT format (output size) -> BGRA, writing into the caller's buffer.
    private var outputConverter: PixelBufferConverter?
    /// Pool of VT-format destination buffers at the *output* size.
    private var vtDestinationConverter: PixelBufferConverter?

    private var submitted: Int = 0
    private var succeeded: Int = 0
    private var failed: Int = 0
    private var timedOut: Int = 0

    /// Compact counters, surfaced through the enhancer status string.
    var statsDescription: String {
      return "scale submit=\(submitted) ok=\(succeeded) fail=\(failed) to=\(timedOut)"
    }

    /// Output size for a given input, exposed so the caller can size its pool.
    static func outputSize(
      width: Int,
      height: Int,
      scaleFactor: Float
    ) -> CGSize {
      // Swift does not implicitly convert Float to CGFloat.
      return CGSize(
        width: CGFloat((Float(width) * scaleFactor).rounded()),
        height: CGFloat((Float(height) * scaleFactor).rounded())
      )
    }

    init?(width: Int, height: Int, scaleFactor: Float) {
      guard width > 0, height > 0, scaleFactor > 1.0 else {
        return nil
      }
      self.frameWidth = width
      self.frameHeight = height
      self.scaleFactor = scaleFactor

      let output = VTFrameScaler.outputSize(
        width: width,
        height: height,
        scaleFactor: scaleFactor
      )
      self.outputWidth = Int(output.width)
      self.outputHeight = Int(output.height)

      self.queue = DispatchQueue(
        label: "media_kit_video.vt.scaler",
        qos: .userInteractive
      )

      let configuration = VTLowLatencySuperResolutionScalerConfiguration(
        frameWidth: width,
        frameHeight: height,
        scaleFactor: scaleFactor
      )

      let formats = configuration.__frameSupportedPixelFormats.map {
        $0.uint32Value
      }
      self.supportedFormats = formats
      if !formats.isEmpty {
        self.acceptsBGRASource = formats.contains(kCVPixelFormatType_32BGRA)
      }

      if !self.acceptsBGRASource {
        // VT wants a format the renderer does not produce; convert in and out.
        guard let target = formats.first else {
          NSLog("VTFrameScaler: VT advertises no usable pixel format")
          return nil
        }
        guard
          let input = PixelBufferConverter(
            width: width, height: height,
            pixelFormat: target, providesPool: true
          ),
          let destination = PixelBufferConverter(
            width: self.outputWidth, height: self.outputHeight,
            pixelFormat: target, providesPool: true
          ),
          let output = PixelBufferConverter(
            width: self.outputWidth, height: self.outputHeight,
            pixelFormat: kCVPixelFormatType_32BGRA, providesPool: false
          )
        else {
          NSLog("VTFrameScaler: cannot set up BGRA <-> \(target) conversion")
          return nil
        }
        self.inputConverter = input
        self.vtDestinationConverter = destination
        self.outputConverter = output
        self.canConvertPixelFormat = true
        NSLog("VTFrameScaler: converting BGRA <-> \(target) per frame")
      }

      let processor = VTFrameProcessor()
      do {
        try processor.startSession(configuration: configuration)
      } catch {
        VTFrameInterpolationSupport.recordStartError(error)
        NSLog("VTFrameScaler: startSession failed: \(error)")
        return nil
      }
      self.processor = processor
    }

    deinit {
      processor?.endSession()
    }

    var isReady: Bool {
      lock.lock()
      defer {
        lock.unlock()
      }
      return processor != nil
    }

    /// Upscales `source` into `destination`.
    ///
    /// `destination` must be `outputWidth` x `outputHeight` and come from a
    /// pool: VideoToolbox forbids touching either buffer until `completion`
    /// fires. Runs asynchronously on the internal queue.
    func scale(
      source: CVPixelBuffer,
      sourceTime: CMTime,
      destination: CVPixelBuffer,
      completion: @escaping (Result<Void, Error>) -> Void
    ) {
      queue.async {
        self.lock.lock()
        let processor = self.processor
        self.lock.unlock()

        guard let processor = processor else {
          self.finish(completion, .failure(VTFrameScalerError.notReady))
          return
        }

        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        if sourceWidth != self.frameWidth || sourceHeight != self.frameHeight {
          self.finish(
            completion,
            .failure(
              VTFrameScalerError.dimensionMismatch(
                expectedWidth: self.frameWidth,
                expectedHeight: self.frameHeight,
                actualWidth: sourceWidth,
                actualHeight: sourceHeight
              )
            )
          )
          return
        }

        let destWidth = CVPixelBufferGetWidth(destination)
        let destHeight = CVPixelBufferGetHeight(destination)
        if destWidth != self.outputWidth || destHeight != self.outputHeight {
          self.finish(
            completion,
            .failure(
              VTFrameScalerError.dimensionMismatch(
                expectedWidth: self.outputWidth,
                expectedHeight: self.outputHeight,
                actualWidth: destWidth,
                actualHeight: destHeight
              )
            )
          )
          return
        }

        // VT refuses BGRA: convert the source in, run the scaler in its own
        // format, then copy the result back into the caller's BGRA buffer.
        var sourceBuffer = source
        var vtDestination = destination
        var needsBackConversion = false

        if let inputConverter = self.inputConverter,
          let vtDestinationConverter = self.vtDestinationConverter
        {
          guard let converted = inputConverter.convert(source) else {
            self.failed += 1
            self.finish(
              completion,
              .failure(VTFrameScalerError.formatConversionFailed)
            )
            return
          }
          sourceBuffer = converted

          guard let vtBuffer = vtDestinationConverter.makeBuffer() else {
            self.failed += 1
            self.finish(
              completion,
              .failure(VTFrameScalerError.formatConversionFailed)
            )
            return
          }
          vtDestination = vtBuffer
          needsBackConversion = true
        }

        guard
          let sourceFrame = VTFrameProcessorFrame(
            buffer: sourceBuffer,
            presentationTimeStamp: sourceTime.isValid ? sourceTime : .zero
          )
        else {
          self.finish(
            completion,
            .failure(VTFrameScalerError.frameCreationFailed)
          )
          return
        }
        guard
          let destinationFrame = VTFrameProcessorFrame(
            buffer: vtDestination,
            presentationTimeStamp: sourceTime.isValid ? sourceTime : .zero
          )
        else {
          self.finish(
            completion,
            .failure(VTFrameScalerError.frameCreationFailed)
          )
          return
        }

        // Both initialisers are documented as non-failable, unlike the frame
        // and frame-rate-conversion siblings.
        let parameters = VTLowLatencySuperResolutionScalerParameters(
          sourceFrame: sourceFrame,
          destinationFrame: destinationFrame
        )

        // Same reasoning as VTFrameInterpolator: waiting here is what keeps
        // submissions from overlapping, and a timeout is reported instead of
        // blocking the render worker forever.
        self.submitted += 1
        let semaphore = DispatchSemaphore(value: 0)
        let errorLock = NSLock()
        var processError: Error?
        processor.process(parameters: parameters) { _, error in
          errorLock.lock()
          processError = error
          errorLock.unlock()
          semaphore.signal()
        }

        if semaphore.wait(timeout: .now() + VTFrameScaler.processTimeout)
          == .timedOut
        {
          self.timedOut += 1
          self.finish(completion, .failure(VTFrameScalerError.timedOut))
          return
        }

        errorLock.lock()
        let error = processError
        errorLock.unlock()

        if let error = error {
          self.failed += 1
          self.finish(
            completion,
            .failure(VTFrameScalerError.scalingFailed(error))
          )
          return
        }

        if needsBackConversion {
          guard let outputConverter = self.outputConverter,
            outputConverter.convert(vtDestination, to: destination)
          else {
            self.failed += 1
            self.finish(
              completion,
              .failure(VTFrameScalerError.formatConversionFailed)
            )
            return
          }
        }

        self.succeeded += 1
        self.finish(completion, .success(()))
      }
    }

    /// Tears the session down. Safe to call more than once.
    func invalidate() {
      lock.lock()
      let processor = self.processor
      self.processor = nil
      lock.unlock()

      guard let processor = processor else {
        return
      }
      queue.async {
        processor.endSession()
      }
    }

    private func finish(
      _ completion: @escaping (Result<Void, Error>) -> Void,
      _ result: Result<Void, Error>
    ) {
      queue.async { completion(result) }
    }
  }

#endif  // !targetEnvironment(simulator)
