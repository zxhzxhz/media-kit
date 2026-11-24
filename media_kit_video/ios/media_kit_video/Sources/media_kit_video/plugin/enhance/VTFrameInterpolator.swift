//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Low-latency 2x frame interpolation via VideoToolbox's VTFrameProcessor (iOS
//  26+): one frame at phase 0.5 between two consecutive frames, written into a
//  caller supplied CVPixelBuffer. Interpolation only: no super resolution, no
//  HDR, no Metal.
//
//  The processor forbids concurrent submissions per session, so calls are
//  serialised on a private queue; the caller owns all buffers and the
//  destination must be pooled and match `init(width:height:)`.

import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

// VideoToolbox marks the whole VTFrameProcessor family as
// `#if !TARGET_OS_SIMULATOR`, so none of these symbols exist in the simulator
// SDK and the file must be excluded there. The enhancer factory falls back to
// the Metal-only path on the simulator.
#if !targetEnvironment(simulator)

/// Failures surfaced through `VTFrameInterpolator.interpolate(...)`.
enum VTFrameInterpolatorError: Error {
  /// No session is running: never started, or `invalidate()` was called.
  case notReady
  /// A buffer does not match the configured frame size.
  case dimensionMismatch(
    expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int)
  /// `VTFrameProcessorFrame` rejected a buffer.
  case frameCreationFailed
  case interpolationFailed(Error)
  /// `process` neither completed nor reported an error in time.
  case timedOut
  /// BGRA <-> the VideoToolbox format conversion failed.
  case formatConversionFailed
}

/// Availability probe that is callable from iOS 13 deployment targets.
///
/// `VTFrameInterpolator` is `@available(iOS 26.0, *)`, and Swift gates the whole
/// type with it, so iOS 13 code can not name the class at all - not even its
/// static members. This shim mirrors `VTFrameInterpolator.isSupported`.
enum VTFrameInterpolationSupport {
  private static let errorLock = NSLock()
  private static var storedStartError: String = ""

  static func isSupported(width: Int, height: Int) -> Bool {
    guard width > 0, height > 0 else { return false }
    if #available(iOS 26.0, *) {
      return VTFrameInterpolationSupport.probe(width: width, height: height)
    }
    return false
  }

  /// Human readable probe result: whether the configuration can be built for
  /// this size, which pixel formats VideoToolbox advertises, and the error of
  /// the last `startSession` call (if any).
  ///
  /// This is what turns "enhancement is enabled but nothing happens" into an
  /// actionable message, since the failure modes of VTFrameProcessor (size
  /// limits, unsupported formats, session errors) are otherwise invisible.
  static func describe(width: Int, height: Int) -> String {
    guard width > 0, height > 0 else { return "invalid size" }
    if #available(iOS 26.0, *) {
      return VTFrameInterpolationSupport.describe26(width: width, height: height)
    }
    return "needs iOS 26"
  }

  static func recordStartError(_ error: Error) {
    errorLock.lock()
    storedStartError = "startSession failed: \(error)"
    errorLock.unlock()
  }

  static func lastStartError() -> String {
    errorLock.lock()
    defer {
      errorLock.unlock()
    }
    return storedStartError
  }

  @available(iOS 26.0, *)
  private static func probe(width: Int, height: Int) -> Bool {
    // A nil configuration means this size is unsupported on this hardware.
    return VTLowLatencyFrameInterpolationConfiguration(
      frameWidth: width, frameHeight: height, numberOfInterpolatedFrames: 1
    ) != nil
  }

  @available(iOS 26.0, *)
  private static func describe26(width: Int, height: Int) -> String {
    var parts: [String] = []

    guard
      let configuration = VTLowLatencyFrameInterpolationConfiguration(
        frameWidth: width, frameHeight: height, numberOfInterpolatedFrames: 1
      )
    else {
      parts.append("configuration rejected for \(width)x\(height)")
      let error = lastStartError()
      if !error.isEmpty {
        parts.append(error)
      }
      return parts.joined(separator: "; ")
    }
    parts.append("configuration ok")

    // `frameSupportedPixelFormats` is NS_REFINED_FOR_SWIFT, so the ObjC name is
    // only reachable with the double underscore prefix in Swift.
    parts.append(
      "pixel formats: \(String(describing: configuration.__frameSupportedPixelFormats))"
    )

    let error = lastStartError()
    if !error.isEmpty {
      parts.append(error)
    }
    return parts.joined(separator: "; ")
  }

  private static func fourCC(_ code: OSType) -> String {
    let bytes: [UInt8] = [
      UInt8((code >> 24) & 0xFF),
      UInt8((code >> 16) & 0xFF),
      UInt8((code >> 8) & 0xFF),
      UInt8(code & 0xFF),
    ]
    if bytes.allSatisfy({ $0 >= 32 && $0 < 127 }),
      let text = String(bytes: bytes, encoding: .ascii)
    {
      return text
    }
    return "0x\(String(code, radix: 16))"
  }
}

/// Wraps a single `VTFrameProcessor` low-latency interpolation session.
@available(iOS 26.0, *)
final class VTFrameInterpolator {
  /// `process` must complete before the next submission: the processor forbids
  /// concurrent work. 1080p interpolation takes a few ms, so 250 ms means
  /// "hung", not "slow".
  private static let processTimeout = DispatchTimeInterval.milliseconds(250)

  private let frameWidth: Int
  private let frameHeight: Int
  /// Serialises submissions: the processor forbids concurrent work.
  private let queue: DispatchQueue
  /// Guards `processor` for readers outside `queue` (namely `isReady`).
  private let lock = NSLock()
  private var processor: VTFrameProcessor?

  /// Pixel formats VideoToolbox advertises for this configuration. Empty when
  /// the property is unavailable.
  private(set) var supportedFormats: [OSType] = []
  /// `false` when the advertised formats are known and do not include the BGRA
  /// buffers TextureHW produces.
  private(set) var acceptsBGRASource = true
  /// `true` when BGRA <-> VT format conversion is set up, i.e. interpolation
  /// works even though VTFrameProcessor refuses BGRA.
  private(set) var canConvertPixelFormat = false

  /// BGRA -> VT format, with a pool for the converted input frames.
  private var inputConverter: PixelBufferConverter?
  /// VT format -> BGRA, writing straight into the caller's buffer.
  private var outputConverter: PixelBufferConverter?
  /// The previous frame in VT format, reused as `previousFrame` so each frame
  /// is converted once instead of twice.
  private var previousVTFrame: CVPixelBuffer?
  private var previousVTFrameTime: CMTime = .invalid

  private var submitted: Int = 0
  private var succeeded: Int = 0
  private var failed: Int = 0
  private var timedOut: Int = 0
  /// Most recent failure, so the status string can say *why* nothing is being
  /// produced (a dimension mismatch or a rejected parameter object otherwise
  /// looks identical to "the stage never ran").
  private var lastError: String = ""

  /// Compact counters, surfaced through the enhancer status string.
  var statsDescription: String {
    var text =
      "interp submit=\(submitted) ok=\(succeeded) fail=\(failed) to=\(timedOut)"
    if !lastError.isEmpty {
      text += " last=\(lastError)"
    }
    return text
  }

  /// Starts the session; nil when the size is rejected or `startSession` throws.
  init?(width: Int, height: Int) {
    self.frameWidth = width
    self.frameHeight = height
    self.queue = DispatchQueue(label: "media_kit_video.vt.interpolation",
                               qos: .userInteractive)
    guard width > 0, height > 0,
      let configuration = VTLowLatencyFrameInterpolationConfiguration(
        frameWidth: width, frameHeight: height, numberOfInterpolatedFrames: 1
      )
    else { return nil }

    let formats = configuration.__frameSupportedPixelFormats.map {
      $0.uint32Value
    }
    self.supportedFormats = formats
    if !formats.isEmpty {
      self.acceptsBGRASource = formats.contains(kCVPixelFormatType_32BGRA)
    }

    if !self.acceptsBGRASource {
      // VideoToolbox advertises a format the renderer does not produce (on
      // iPad Pro M4 it only accepts 420v). Convert instead of giving up.
      guard let target = formats.first else {
        NSLog("VTFrameInterpolator: VT advertises no usable pixel format")
        return nil
      }
      guard
        let input = PixelBufferConverter(
          width: width, height: height,
          pixelFormat: target, providesPool: true
        ),
        let output = PixelBufferConverter(
          width: width, height: height,
          pixelFormat: kCVPixelFormatType_32BGRA, providesPool: false
        )
      else {
        NSLog(
          "VTFrameInterpolator: cannot set up BGRA <-> \(target) conversion"
        )
        return nil
      }
      self.inputConverter = input
      self.outputConverter = output
      self.canConvertPixelFormat = true
      NSLog("VTFrameInterpolator: converting BGRA <-> \(target) per frame")
    }

    // UNCERTAIN: assumes the no-argument `VTFrameProcessor()` initialiser.
    let processor = VTFrameProcessor()
    do {
      try processor.startSession(configuration: configuration)
    } catch {
      // Remember why: the caller only sees `nil` and needs the reason.
      VTFrameInterpolationSupport.recordStartError(error)
      NSLog("VTFrameInterpolator: startSession failed: \(error)")
      return nil
    }
    self.processor = processor
  }

  deinit {
    processor?.endSession()  // best effort; `invalidate()` is the real path
  }

  var isReady: Bool {
    lock.lock()
    defer { lock.unlock() }
    return processor != nil
  }

  /// Generates one frame at phase 0.5 between `previous` and `source`, written
  /// into `destination`. All three buffers must be `frameWidth` x `frameHeight`;
  /// `destination` is normally drained from the caller's `CVPixelBufferPool`.
  /// VideoToolbox forbids touching the buffers (attachments included) until
  /// `completion` fires; it always runs asynchronously on the internal queue.
  func interpolate(
    previous: CVPixelBuffer,
    previousTime: CMTime,
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
        self.finish(completion, .failure(VTFrameInterpolatorError.notReady))
        return
      }

      for buffer in [previous, source, destination] {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        if width != self.frameWidth || height != self.frameHeight {
          let error = VTFrameInterpolatorError.dimensionMismatch(
            expectedWidth: self.frameWidth, expectedHeight: self.frameHeight,
            actualWidth: width, actualHeight: height
          )
          self.finish(completion, .failure(error))
          return
        }
      }

      // UNCERTAIN: the docs do not state whether the processor derives output
      // timing from the destination timestamp or ignores it; for phase 0.5 the
      // midpoint of the two source frames is the natural value.
      let delta = CMTimeSubtract(sourceTime, previousTime)
      let destinationTime = CMTimeAdd(previousTime,
                                      CMTimeMultiplyByFloat64(delta, multiplier: 0.5))

      // When VT refuses BGRA, convert into its format first and back into the
      // caller's buffer afterwards. The previous frame's converted copy is
      // cached, so each frame is converted once instead of twice.
      var previousBuffer = previous
      var sourceBuffer = source
      var vtDestination = destination
      var needsBackConversion = false

      if let inputConverter = self.inputConverter {
        let convertedPrevious: CVPixelBuffer
        if let cached = self.previousVTFrame,
          CMTimeCompare(self.previousVTFrameTime, previousTime) == 0
        {
          convertedPrevious = cached
        } else {
          guard let converted = inputConverter.convert(previous) else {
            self.finish(
              completion,
              .failure(VTFrameInterpolatorError.formatConversionFailed)
            )
            return
          }
          convertedPrevious = converted
        }
        previousBuffer = convertedPrevious

        guard let convertedSource = inputConverter.convert(source) else {
          self.finish(
            completion,
            .failure(VTFrameInterpolatorError.formatConversionFailed)
          )
          return
        }
        sourceBuffer = convertedSource
        self.previousVTFrame = convertedSource
        self.previousVTFrameTime = sourceTime

        guard let vtDestinationBuffer = inputConverter.makeBuffer() else {
          self.finish(
            completion,
            .failure(VTFrameInterpolatorError.formatConversionFailed)
          )
          return
        }
        vtDestination = vtDestinationBuffer
        needsBackConversion = true
      }

      guard
        let previousFrame = VTFrameProcessorFrame(
          buffer: previousBuffer, presentationTimeStamp: previousTime
        ),
        let sourceFrame = VTFrameProcessorFrame(
          buffer: sourceBuffer, presentationTimeStamp: sourceTime
        ),
        let destinationFrame = VTFrameProcessorFrame(
          buffer: vtDestination, presentationTimeStamp: destinationTime
        )
      else {
        self.finish(completion, .failure(VTFrameInterpolatorError.frameCreationFailed))
        return
      }

      // Four arguments, verified against
      // VTFrameProcessor_LowLatencyFrameInterpolation.h (iOS 26.5 SDK).
      // The initialiser is nullable, hence the guard.
      guard
        let parameters = VTLowLatencyFrameInterpolationParameters(
          sourceFrame: sourceFrame,
          previousFrame: previousFrame,
          interpolationPhase: [0.5],
          destinationFrames: [destinationFrame]
        )
      else {
        self.finish(completion, .failure(VTFrameInterpolatorError.frameCreationFailed))
        return
      }

      // The processor forbids concurrent submissions, and `queue.async` only
      // serialises the *calls*, not the completions. Wait here so the next
      // submission cannot overlap this one; a timeout means the processor is
      // stuck, in which case the frame is dropped instead of blocking forever.
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

      if semaphore.wait(timeout: .now() + VTFrameInterpolator.processTimeout)
        == .timedOut
      {
        self.timedOut += 1
        self.finish(completion, .failure(VTFrameInterpolatorError.timedOut))
        return
      }

      errorLock.lock()
      let error = processError
      errorLock.unlock()

      if let error = error {
        self.finish(
          completion,
          .failure(VTFrameInterpolatorError.interpolationFailed(error))
        )
        return
      }

      // Copy the interpolated frame back into the caller's BGRA buffer.
      if needsBackConversion {
        guard let outputConverter = self.outputConverter,
          outputConverter.convert(vtDestination, to: destination)
        else {
          self.finish(
            completion,
            .failure(VTFrameInterpolatorError.formatConversionFailed)
          )
          return
        }
      }

      self.succeeded += 1
      self.finish(completion, .success(()))
    }
  }

  /// Ends the session and drops the processor; pending submissions fail with
  /// `notReady`. Call it only after the last completion fired.
  func invalidate() {
    queue.async {
      self.lock.lock()
      let processor = self.processor
      self.processor = nil
      self.lock.unlock()

      processor?.endSession()
    }
  }

  /// Mirrors `VTFrameInterpolationSupport.isSupported(width:height:)`, but only
  /// reachable behind `#available(iOS 26.0, *)` since the type itself is gated.
  static func isSupported(width: Int, height: Int) -> Bool {
    return VTFrameInterpolationSupport.isSupported(width: width, height: height)
  }

  /// Every failure funnels through here, so the counters and the last error
  /// stay accurate even for branches that fail before submission (dimension
  /// checks, frame or parameter construction). Those used to be invisible:
  /// `submit` and `fail` both stayed at zero.
  private func finish(_ completion: @escaping (Result<Void, Error>) -> Void,
                      _ result: Result<Void, Error>) {
    if case .failure(let error) = result {
      failed += 1
      lastError = "\(error)"
    }
    queue.async { completion(result) }
  }
}

#endif  // !targetEnvironment(simulator)
