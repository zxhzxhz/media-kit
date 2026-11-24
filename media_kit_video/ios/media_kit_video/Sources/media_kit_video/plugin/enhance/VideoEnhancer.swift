//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Concrete enhancement pipelines.
//
//  * MetalOnlyEnhancer  - Anime4K on Metal. Works on every Metal capable
//                         device, no VideoToolbox requirement.
//  * IOS26VideoEnhancer - VideoToolbox frame interpolation and/or super
//                         resolution (with optional Anime4K). iOS 26.0+ only.
//
//  Both are deliberately conservative: any failure disables the stage and
//  leaves the plain TextureHW path untouched, so enhancement can never break
//  playback. Every failure is recorded in `statusDescription` so that the app
//  can tell the user *why* nothing happened.

import CoreMedia
import CoreVideo
import Foundation

// MARK: - Anime4K only

/// Runs Anime4K on Metal for every decoded frame.
final class MetalOnlyEnhancer: FrameEnhancerProtocol {
  private var processor: MetalFrameProcessor?
  private let frames = EnhanceFrameQueue(capacity: 8)
  private var status: String = "Metal: idle"

  var onFrameAvailable: (() -> Void)?
  private(set) var isActive: Bool = false

  var statusDescription: String {
    return status
  }

  init() {
    processor = MetalFrameProcessor()
    if processor == nil {
      status = "Metal: device unavailable"
    }
  }

  func configure(
    configuration: VideoEnhanceConfiguration,
    inputSize: CGSize,
    outputSize: CGSize
  ) -> Bool {
    // The VideoToolbox stages belong to IOS26VideoEnhancer.
    guard !configuration.interpolation, configuration.vtScaleFactor <= 0 else {
      status = "Metal: not applicable (VideoToolbox requested)"
      return false
    }
    guard !configuration.anime4kShaders.isEmpty else {
      status = "Metal: no Anime4K shader selected"
      return false
    }
    let width = Int(inputSize.width)
    let height = Int(inputSize.height)
    guard width > 0, height > 0 else {
      return false
    }
    guard let processor = processor else {
      status = "Metal: device unavailable"
      return false
    }

    let configured = processor.configure(
      profile: configuration.anime4kProfile,
      glslSources: configuration.anime4kShaders,
      width: width,
      height: height
    )
    guard configured, processor.hasAnime4K else {
      status = "Metal: Anime4K unavailable (\(width)x\(height), "
        + "\(configuration.anime4kProfile)); \(processor.statusDescription)"
      return false
    }

    frames.clear()
    isActive = true
    status =
      "active: Anime4K \(configuration.anime4kProfile) \(width)x\(height), "
      + "\(processor.enabledPassCount) passes"
    NSLog("MetalOnlyEnhancer: \(status)")
    return true
  }

  func submit(
    _ source: CVPixelBuffer,
    presentationTime: CMTime,
    frameIndex: Int64
  ) {
    guard isActive, let processor = processor else {
      return
    }
    guard let output = processor.process(source) else {
      return
    }
    frames.push(
      EnhanceFrame(pixelBuffer: output, presentationTime: presentationTime)
    )
    onFrameAvailable?()
  }

  func latestFrame() -> CVPixelBuffer? {
    return frames.latest()?.pixelBuffer
  }

  func frame(nearestTo time: CMTime) -> CVPixelBuffer? {
    return frames.nearest(to: time)?.pixelBuffer
  }

  func invalidate() {
    isActive = false
    frames.clear()
    processor?.invalidate()
  }
}

// MARK: - VideoToolbox (iOS 26+)

// The VideoToolbox frame processor family is unavailable in the simulator SDK
// (`#if !TARGET_OS_SIMULATOR` in the headers), so this class is device only.
#if !targetEnvironment(simulator)

  /// Frame interpolation and/or super resolution through VideoToolbox.
  ///
  /// Frame pacing is owned by `VideoOutput`'s display link: this class only
  /// produces frames with their presentation timestamps and the renderer decides
  /// which one belongs on screen for the current clock.
  ///
  /// Super resolution writes into a second pool at the upscaled size, so the
  /// Anime4K stage is disabled whenever VT scaling is requested (chaining both
  /// would upscale twice and blow the memory budget).
  @available(iOS 26.0, *)
  final class IOS26VideoEnhancer: FrameEnhancerProtocol {
    private var processor: MetalFrameProcessor?
    private var interpolator: VTFrameInterpolator?
    private var scaler: VTFrameScaler?
    /// Output size pool, only allocated when VT scaling is requested.
    private var outputPool: CVPixelBufferPool?
    private var outputWidth: Int = 0
    private var outputHeight: Int = 0
    private var scaleFactor: Float = 1.0
    /// Holds [previous, current, interpolated] at most, so 4 entries are plenty.
    private let frames = EnhanceFrameQueue(capacity: 8)

    var onFrameAvailable: (() -> Void)?
    private(set) var isActive: Bool = false
    private var baseStatus: String = "VideoToolbox: idle"

    /// Includes the live submission counters, so the FPS overlay can show
    /// whether VT is actually producing frames (`ok` growing) or stalling
    /// (`to` / `fail` growing) without a Console connection.
    var statusDescription: String {
      guard isActive else {
        return baseStatus
      }
      var parts: [String] = [baseStatus]
      parts.append(
        "frames=\(submittedFrames)/\(interpolateCalls) emit=\(emitCount) push=\(pushCount)"
      )
      if let interpolator = interpolator {
        parts.append(interpolator.statsDescription)
      }
      if let scaler = scaler {
        parts.append(scaler.statsDescription)
      }
      if failureCount > 0 {
        parts.append("emitFail=\(failureCount)")
      }
      return parts.joined(separator: " | ")
    }

    private var width: Int = 0
    private var height: Int = 0
    private var previousBuffer: CVPixelBuffer?
    private var previousTime: CMTime = .invalid
    private var submittedCount: Int64 = 0
    private var failureCount: Int = 0
    private var lastFailure: String = ""
    /// Frames handed to `submit`, and how many of those reached the
    /// interpolator. Together they separate "submit is never called" from
    /// "submit runs but interpolation is skipped".
    private var submittedFrames: Int = 0
    private var interpolateCalls: Int = 0
    /// BGRA -> 420v for the frames VT did not produce (the mpv frame when only
    /// scaling is on, or when nothing is being scaled).
    private var toVTConverter: PixelBufferConverter?
    /// Published frame counters, so the status can tell "emit never ran" from
    /// "emit ran but nothing was pushed".
    private var emitCount: Int = 0
    private var pushCount: Int = 0

    init?() {
      guard let processor = MetalFrameProcessor() else {
        return nil
      }
      self.processor = processor
    }

    func configure(
      configuration: VideoEnhanceConfiguration,
      inputSize: CGSize,
      outputSize: CGSize
    ) -> Bool {
      let width = Int(inputSize.width)
      let height = Int(inputSize.height)
      guard width > 0, height > 0 else {
        return false
      }

      let wantsInterpolation = configuration.interpolation
      let requestedScale = Float(configuration.vtScaleFactor)
      let wantsScaling = requestedScale > 1.0
      guard wantsInterpolation || wantsScaling else {
        baseStatus = "nothing requested"
        return false
      }

      // Probe both stages before touching any state: these are the most common
      // reasons for "enabled but no visible difference" on large frames, and
      // the reason must reach the app.
      var diagnostics: [String] = []
      if wantsInterpolation {
        let probe = VTFrameInterpolationSupport.describe(
          width: width,
          height: height
        )
        diagnostics.append("interp[\(probe)]")
        guard VTFrameInterpolator.isSupported(width: width, height: height)
        else {
          baseStatus = "unsupported interpolation \(width)x\(height) — \(probe)"
          NSLog("IOS26VideoEnhancer: \(baseStatus)")
          return false
        }
      }
      if wantsScaling {
        let probe = VTFrameScalingSupport.describe(
          width: width,
          height: height,
          scaleFactor: requestedScale
        )
        diagnostics.append("scale[\(probe)]")
        guard
          VTFrameScalingSupport.isSupported(
            width: width,
            height: height,
            scaleFactor: requestedScale
          )
        else {
          baseStatus = "unsupported scaling \(width)x\(height) — \(probe)"
          NSLog("IOS26VideoEnhancer: \(baseStatus)")
          return false
        }
      }

      self.width = width
      self.height = height
      self.scaleFactor = wantsScaling ? requestedScale : 1.0

      // The Metal processor always runs: it copies the mpv frame into a buffer
      // that survives across frames. Anime4K only when VT is not scaling.
      guard let processor = processor else {
        baseStatus = "Metal unavailable"
        return false
      }
      let metalReady = processor.configure(
        profile: configuration.anime4kProfile,
        glslSources: wantsScaling ? [] : configuration.anime4kShaders,
        width: width,
        height: height
      )
      guard metalReady else {
        baseStatus = "Metal rejected \(width)x\(height): \(processor.statusDescription)"
        NSLog("IOS26VideoEnhancer: \(baseStatus)")
        return false
      }

      // Converts the mpv BGRA frame into the format Flutter consumes directly.
      if toVTConverter == nil {
        toVTConverter = PixelBufferConverter(
          width: width,
          height: height,
          pixelFormat: kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
          providesPool: true
        )
      }

      if wantsScaling {
        let outputSize = VTFrameScaler.outputSize(
          width: width,
          height: height,
          scaleFactor: requestedScale
        )
        outputWidth = Int(outputSize.width)
        outputHeight = Int(outputSize.height)

        guard
          let pool = IOS26VideoEnhancer.makePool(
            width: outputWidth,
            height: outputHeight
          )
        else {
          baseStatus = "output pool allocation failed \(outputWidth)x\(outputHeight)"
          NSLog("IOS26VideoEnhancer: \(baseStatus)")
          return false
        }
        outputPool = pool

        guard
          let scaler = VTFrameScaler(
            width: width,
            height: height,
            scaleFactor: requestedScale
          )
        else {
          baseStatus = "scaler session failed \(width)x\(height) — \(diagnostics.joined(separator: " "))"
          NSLog("IOS26VideoEnhancer: \(baseStatus)")
          return false
        }
        self.scaler = scaler
        if !scaler.acceptsBGRASource && !scaler.canConvertPixelFormat {
          baseStatus =
            "scaler needs BGRA conversion but VTPixelTransferSession is "
            + "unavailable \(width)x\(height)"
          NSLog("IOS26VideoEnhancer: \(baseStatus)")
          return false
        }
      } else {
        outputPool = nil
        outputWidth = width
        outputHeight = height
        scaler?.invalidate()
        scaler = nil
      }

      if wantsInterpolation {
        guard
          let interpolator = VTFrameInterpolator(width: width, height: height)
        else {
          baseStatus = "interpolation session failed \(width)x\(height) — \(diagnostics.joined(separator: " "))"
          NSLog("IOS26VideoEnhancer: \(baseStatus)")
          return false
        }
        self.interpolator = interpolator
        if !interpolator.acceptsBGRASource && !interpolator.canConvertPixelFormat
        {
          baseStatus =
            "interpolation needs BGRA conversion but VTPixelTransferSession is "
            + "unavailable \(width)x\(height)"
          NSLog("IOS26VideoEnhancer: \(baseStatus)")
          return false
        }
      } else {
        interpolator?.invalidate()
        interpolator = nil
      }

      frames.clear()
      previousBuffer = nil
      previousTime = .invalid
      submittedCount = 0
      failureCount = 0
      lastFailure = ""
      submittedFrames = 0
      interpolateCalls = 0
      isActive = true

      let stages = [
        wantsInterpolation ? "interp 2x" : nil,
        wantsScaling ? "scale \(requestedScale)x" : nil,
      ].compactMap { $0 }.joined(separator: " + ")
      // Call out the conversion: VT only accepts 420v on this hardware, so the
      // frames go through VTPixelTransferSession in both directions.
      let conversion =
        (interpolator?.canConvertPixelFormat == true
          || scaler?.canConvertPixelFormat == true)
        ? " [420v conv]" : ""
      baseStatus =
        "active: \(stages)\(conversion) \(width)x\(height) → \(outputWidth)x\(outputHeight), "
        + "anime4k \(wantsScaling ? "off" : configuration.anime4kProfile)"
      NSLog("IOS26VideoEnhancer: \(baseStatus)")
      return true
    }

    func submit(
      _ source: CVPixelBuffer,
      presentationTime: CMTime,
      frameIndex: Int64
    ) {
      guard isActive, let processor = processor else {
        return
      }
      submittedFrames += 1

      // 1. Take ownership of the decoded frame. The GL texture buffers are
      //    recycled by TextureHW, so anything kept across frames must be a copy.
      guard let owned = processor.process(source) else {
        recordFailure("frame copy failed")
        return
      }

      // 2. Emit the original frame.
      emit(owned, at: presentationTime)

      let previous = previousBuffer
      let previousPTS = previousTime
      previousBuffer = owned
      previousTime = presentationTime
      submittedCount += 1

      // 3. Emit the interpolated frame between the previous and the current one.
      guard let interpolator = interpolator, let previous = previous,
        previousPTS.isValid, presentationTime.isValid
      else {
        return
      }
      guard let destination = interpolator.makeDestinationBuffer() else {
        recordFailure("interpolation destination unavailable")
        return
      }

      let middleTime = CMTimeAdd(
        previousPTS,
        CMTimeMultiplyByFloat64(
          CMTimeSubtract(presentationTime, previousPTS),
          multiplier: 0.5
        )
      )

      interpolateCalls += 1
      interpolator.interpolate(
        previous: previous,
        previousTime: previousPTS,
        source: owned,
        sourceTime: presentationTime,
        destination: destination
      ) { [weak self] result in
        guard let self = self else {
          return
        }
        switch result {
        case .failure(let error):
          self.recordFailure("\(error)")
        case .success:
          if let processor = self.processor, processor.hasAnime4K {
            let output = processor.process(destination) ?? destination
            self.emit(output, at: middleTime)
          } else {
            self.emit(destination, at: middleTime)
          }
        }
      }
    }

    func latestFrame() -> CVPixelBuffer? {
      return frames.latest()?.pixelBuffer
    }

    func frame(nearestTo time: CMTime) -> CVPixelBuffer? {
      return frames.nearest(to: time)?.pixelBuffer
    }

    func invalidate() {
      isActive = false
      frames.clear()
      previousBuffer = nil
      previousTime = .invalid
      interpolator?.invalidate()
      interpolator = nil
      scaler?.invalidate()
      scaler = nil
      outputPool = nil
      processor?.invalidate()
    }

    // MARK: - Internals

    /// Publishes one frame, upscaling it first when VT scaling is active.
    ///
    /// The scaler is asynchronous, but submissions are already serialised on
    /// the render worker and the wait is bounded; doing it synchronously keeps
    /// frame ordering deterministic and stops the output pool from starving.
    /// Publishes one frame. Scaling runs first when enabled; either way the
    /// published buffer is 420v, which Flutter's iOS engine consumes directly.
    ///
    /// This removes the 420v -> BGRA pass entirely. That pass was where scaling
    /// went black: CoreImage would report success while writing nothing into a
    /// 4K BGRA target, and a BGRA-side blank check cannot even tell an unwritten
    /// buffer apart from content.
    private func emit(_ buffer: CVPixelBuffer, at time: CMTime) {
      emitCount += 1

      guard let scaler = scaler else {
        publish(buffer, at: time)
        return
      }

      guard let destination = scaler.makeDestinationBuffer() else {
        recordFailure("scale destination unavailable")
        publish(buffer, at: time)
        return
      }

      let semaphore = DispatchSemaphore(value: 0)
      let lock = NSLock()
      var failure: String?
      scaler.scale(
        source: buffer,
        sourceTime: time,
        destination: destination
      ) { result in
        if case .failure(let error) = result {
          lock.lock()
          failure = "\(error)"
          lock.unlock()
        }
        semaphore.signal()
      }

      if semaphore.wait(timeout: .now() + 0.3) == .timedOut {
        recordFailure("scaler timed out")
        publish(buffer, at: time)
        return
      }

      lock.lock()
      let recorded = failure
      lock.unlock()
      if let recorded = recorded {
        recordFailure("scale: \(recorded)")
        publish(buffer, at: time)
        return
      }

      publish(destination, at: time)
    }

    /// Publishes a 420v frame. BGRA sources (the mpv frame when scaling and
    /// interpolation are both off) are converted once; VT output is passed
    /// through untouched.
    private func publish(_ buffer: CVPixelBuffer, at time: CMTime) {
      var output: CVPixelBuffer? = buffer
      if CVPixelBufferGetPixelFormatType(buffer)
        != kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
      {
        output = toVTConverter?.convert(buffer)
      }
      guard let published = output else {
        recordFailure("output conversion failed")
        return
      }
      if PixelBufferConverter.isBlankYUV(published) {
        recordFailure("published frame is blank")
        return
      }
      frames.push(EnhanceFrame(pixelBuffer: published, presentationTime: time))
      pushCount += 1
      onFrameAvailable?()
    }

    /// Keeps the most recent failure in `statusDescription` so the app can show
    /// it instead of "enhancement is on but nothing changed".
    private func recordFailure(_ reason: String) {
      failureCount += 1
      guard reason != lastFailure else {
        return
      }
      lastFailure = reason
      baseStatus =
        "active but failing: \(width)x\(height) — \(reason) "
        + "(\(failureCount) failures)"
      NSLog("IOS26VideoEnhancer: \(baseStatus)")
    }

    /// `true` when the BGRA frame carries no visible content.
    ///
    /// Nine samples across the frame: a single non-zero channel means there is
    /// real picture data. Used to reject frames the conversion pipeline left
    /// empty even though VideoToolbox reported success.
    private static func isBlank(_ buffer: CVPixelBuffer) -> Bool {
      guard
        CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA,
        CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess
      else {
        return false
      }
      defer {
        CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
      }
      guard let base = CVPixelBufferGetBaseAddress(buffer) else {
        return false
      }
      let width = CVPixelBufferGetWidth(buffer)
      let height = CVPixelBufferGetHeight(buffer)
      let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
      guard width > 0, height > 0 else {
        return false
      }
      let bytes = base.assumingMemoryBound(to: UInt8.self)
      for row in 0..<3 {
        for col in 0..<3 {
          let x = width * (col * 2 + 1) / 6
          let y = height * (row * 2 + 1) / 6
          let offset = y * bytesPerRow + x * 4
          if bytes[offset] != 0 || bytes[offset + 1] != 0
            || bytes[offset + 2] != 0
          {
            return false
          }
        }
      }
      return true
    }

    private static func makePool(
      width: Int,
      height: Int
    ) -> CVPixelBufferPool? {
      // Route through the shared helper: it puts the minimum buffer count in
      // the pool attributes (where it belongs) and creates the buffers with
      // the IOSurface properties CoreImage needs for a 4K render target.
      return PixelBufferConverter.makeResolvedPool(
        vtAttributes: nil,
        width: width,
        height: height,
        pixelFormat: kCVPixelFormatType_32BGRA
      )
    }
  }

#endif  // !targetEnvironment(simulator)

// MARK: - Factory

enum VideoEnhancerFactory {
  /// Builds the best pipeline the current device can run for `configuration`.
  /// Returns nil when none of the requested stages can be honoured; callers
  /// must then keep the plain rendering path.
  ///
  /// The `reason` out parameter always receives a human readable explanation.
  static func make(
    configuration: VideoEnhanceConfiguration,
    reason: inout String
  ) -> FrameEnhancerProtocol? {
    let wantsVideoToolbox =
      configuration.interpolation || configuration.vtScaleFactor > 0

    if wantsVideoToolbox {
      #if !targetEnvironment(simulator)
        if #available(iOS 26.0, *) {
          if let enhancer = IOS26VideoEnhancer() {
            reason = enhancer.statusDescription
            return enhancer
          }
          reason = "VideoToolbox: Metal unavailable"
        } else {
          reason = "VideoToolbox: needs iOS 26"
        }
      #else
        reason = "VideoToolbox: unavailable in the simulator"
      #endif
      NSLog(
        "VideoEnhancerFactory: VideoToolbox unavailable (\(reason)), "
          + "falling back to Anime4K only"
      )
    }

    if !configuration.anime4kShaders.isEmpty {
      let enhancer = MetalOnlyEnhancer()
      reason = enhancer.statusDescription
      return enhancer
    }

    if reason.isEmpty {
      reason = wantsVideoToolbox
        ? "VideoToolbox unavailable and no Anime4K shader selected"
        : "no stage enabled"
    }
    return nil
  }
}
