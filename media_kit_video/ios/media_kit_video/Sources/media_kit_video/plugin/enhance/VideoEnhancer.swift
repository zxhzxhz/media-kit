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
  private let frames = EnhanceFrameQueue(capacity: 4)
  private var status: String = "Metal: idle"

  private(set) var isActive: Bool = false

  var producesInterpolatedFrames: Bool {
    return false
  }

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
    // No notification: `VideoOutput` drives presentation from the mpv frame
    // callback, so the enqueue is enough.
  }

  func latestFrame() -> CVPixelBuffer? {
    return frames.latest()?.pixelBuffer
  }

  func newestFrame() -> CVPixelBuffer? {
    return frames.newest()?.pixelBuffer
  }

  func newestEnhanceFrame() -> EnhanceFrame? {
    return frames.newest()
  }

  func frame(nearestTo time: CMTime) -> CVPixelBuffer? {
    return frames.nearest(to: time)?.pixelBuffer
  }

  func enhanceFrame(nearestTo time: CMTime) -> EnhanceFrame? {
    return frames.nearest(to: time)
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
    private var outputWidth: Int = 0
    private var outputHeight: Int = 0
    private var scaleFactor: Float = 1.0
    /// Holds [F(n-2), F(n-1.5), F(n-1), F(n-0.5), F(n)] plus slack for the
    /// two frame presentation lag: 4 was too tight once extraction and
    /// interpolation both publish into the same queue.
    private let frames = EnhanceFrameQueue(capacity: 8)

    private var _isActive: Bool = false
    var isActive: Bool {
      stateLock.lock()
      defer { stateLock.unlock() }
      return _isActive
    }

    var producesInterpolatedFrames: Bool {
      stateLock.lock()
      defer { stateLock.unlock() }
      return interpolator != nil
    }

    private var baseStatus: String = "VideoToolbox: idle"

    /// Includes the live submission counters, so the FPS overlay can show
    /// whether VT is actually producing frames (`ok` growing) or stalling
    /// (`to` / `fail` growing) without a Console connection.
    var statusDescription: String {
      stateLock.lock()
      let active = _isActive
      let base = baseStatus
      let sub = submittedFrames
      let icall = interpolateCalls
      let em = emitCount
      let pu = pushCount
      let pSrc = pushSrc
      let eSrc = emitSrc
      let pInt = pushInterp
      let eInt = emitInterp
      let fail = failureCount
      let interp = interpolator
      let scl = scaler
      stateLock.unlock()

      guard active else {
        return base
      }
      var parts: [String] = [base]
      parts.append(
        "frames=\(sub)/\(icall) emit=\(em) push=\(pu)"
      )
      parts.append(
        "src=\(pSrc)/\(eSrc) interp=\(pInt)/\(eInt)"
      )
      if let interp = interp {
        parts.append(interp.statsDescription)
      }
      if let scl = scl {
        parts.append(scl.statsDescription)
      }
      if fail > 0 {
        parts.append("emitFail=\(fail)")
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
    /// 420v -> 32BGRA for interpolated frames when VT scaling is disabled, so
    /// every published frame reaches Flutter as untagged 32BGRA identical to plain TextureHW.
    private var toBGRAConverter: PixelBufferConverter?
    /// Published frame counters, so the status can tell "emit never ran" from
    /// "emit ran but nothing was pushed".
    private var emitCount: Int = 0
    private var pushCount: Int = 0
    /// Split by origin. A bare `emit`/`push` pair cannot distinguish "the
    /// interpolated frame is produced but rejected" from "the decoded frame is
    /// rejected twice": both look like a 2:1 ratio.
    private enum FrameOrigin {
      case decoded
      case interpolated
    }
    private var emitSrc: Int = 0
    private var emitInterp: Int = 0
    private var pushSrc: Int = 0
    private var pushInterp: Int = 0
    /// Sampling budget for the blankness probe (see `BlankCheckBudget`).
    ///
    /// Guarded by `stateLock` alongside the counters and status fields:
    /// `publish` runs both on the render worker (decoded frame) and on the
    /// VideoToolbox completion queue (interpolated frame), while `GetStats`
    /// reads from the main thread.
    private var blankBudget = BlankCheckBudget()
    private let stateLock = NSLock()

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
        stateLock.lock()
        baseStatus = "nothing requested"
        stateLock.unlock()
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
          let message = "unsupported interpolation \(width)x\(height) — \(probe)"
          stateLock.lock()
          baseStatus = message
          stateLock.unlock()
          NSLog("IOS26VideoEnhancer: \(message)")
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
          let message = "unsupported scaling \(width)x\(height) — \(probe)"
          stateLock.lock()
          baseStatus = message
          stateLock.unlock()
          NSLog("IOS26VideoEnhancer: \(message)")
          return false
        }
      }
      NSLog("IOS26VideoEnhancer: probes \(diagnostics.joined(separator: " "))")

      let sizeChanged = self.width != width || self.height != height

      // The Metal processor always runs: it copies the mpv frame into a buffer
      // that survives across frames. Anime4K only when VT is not scaling.
      guard let processor = processor else {
        stateLock.lock()
        baseStatus = "Metal unavailable"
        stateLock.unlock()
        return false
      }
      let metalReady = processor.configure(
        profile: configuration.anime4kProfile,
        glslSources: wantsScaling ? [] : configuration.anime4kShaders,
        width: width,
        height: height
      )
      guard metalReady else {
        let message = "Metal rejected \(width)x\(height): \(processor.statusDescription)"
        stateLock.lock()
        baseStatus = message
        stateLock.unlock()
        NSLog("IOS26VideoEnhancer: \(message)")
        return false
      }

      let nextScaler: VTFrameScaler?
      let nextOutputWidth: Int
      let nextOutputHeight: Int
      if wantsScaling {
        let scaledSize = VTFrameScaler.outputSize(
          width: width,
          height: height,
          scaleFactor: requestedScale
        )
        nextOutputWidth = Int(scaledSize.width)
        nextOutputHeight = Int(scaledSize.height)

        guard
          let scaler = VTFrameScaler(
            width: width,
            height: height,
            scaleFactor: requestedScale
          )
        else {
          let message =
            "scaler session failed \(width)x\(height) — \(diagnostics.joined(separator: " "))"
          stateLock.lock()
          baseStatus = message
          stateLock.unlock()
          NSLog("IOS26VideoEnhancer: \(message)")
          return false
        }
        if !scaler.acceptsBGRASource && !scaler.canConvertPixelFormat {
          let message =
            "scaler needs BGRA conversion but VTPixelTransferSession is "
            + "unavailable \(width)x\(height)"
          stateLock.lock()
          baseStatus = message
          stateLock.unlock()
          NSLog("IOS26VideoEnhancer: \(message)")
          return false
        }
        nextScaler = scaler
      } else {
        nextOutputWidth = width
        nextOutputHeight = height
        nextScaler = nil
      }

      let nextInterpolator: VTFrameInterpolator?
      if wantsInterpolation {
        guard
          let interpolator = VTFrameInterpolator(width: width, height: height)
        else {
          nextScaler?.invalidate()
          let message =
            "interpolation session failed \(width)x\(height) — \(diagnostics.joined(separator: " "))"
          stateLock.lock()
          baseStatus = message
          stateLock.unlock()
          NSLog("IOS26VideoEnhancer: \(message)")
          return false
        }
        if !interpolator.acceptsBGRASource && !interpolator.canConvertPixelFormat
        {
          nextScaler?.invalidate()
          interpolator.invalidate()
          let message =
            "interpolation needs BGRA conversion but VTPixelTransferSession is "
            + "unavailable \(width)x\(height)"
          stateLock.lock()
          baseStatus = message
          stateLock.unlock()
          NSLog("IOS26VideoEnhancer: \(message)")
          return false
        }
        nextInterpolator = interpolator
      } else {
        nextInterpolator = nil
      }

      // Convert interpolated 420v outputs back to untagged 32BGRA via exact
      // BT.709 Metal math when VT scaling is disabled, so Flutter receives the
      // same 32BGRA format as plain playback and bypasses Impeller's BT.601 YUV pass.
      let nextConverter: PixelBufferConverter?
      if !wantsInterpolation || wantsScaling {
        nextConverter = nil
      } else if toBGRAConverter == nil || sizeChanged {
        guard
          let created = PixelBufferConverter(
            width: width,
            height: height,
            pixelFormat: kCVPixelFormatType_32BGRA,
            providesPool: true
          )
        else {
          nextScaler?.invalidate()
          nextInterpolator?.invalidate()
          let message = "interpolation BGRA converter unavailable \(width)x\(height)"
          stateLock.lock()
          baseStatus = message
          stateLock.unlock()
          NSLog("IOS26VideoEnhancer: \(message)")
          return false
        }
        nextConverter = created
      } else {
        nextConverter = toBGRAConverter
      }

      let stages = [
        wantsInterpolation ? "interp 2x" : nil,
        wantsScaling ? "scale \(requestedScale)x" : nil,
      ].compactMap { $0 }.joined(separator: " + ")
      var convTags: [String] = []
      if nextInterpolator?.canConvertPixelFormat == true {
        convTags.append(wantsScaling ? "420v conv" : "BGRA out")
      }
      if let nextScaler = nextScaler, nextScaler.canConvertPixelFormat {
        convTags.append(
          "scale:\(nextScaler.backendLabel):\(PixelBufferConverter.fourCC(nextScaler.targetPixelFormat))->BGRA"
        )
      }
      let conversion =
        convTags.isEmpty ? "" : " [\(convTags.joined(separator: ", "))]"
      let readyStatus =
        "active: \(stages)\(conversion) \(width)x\(height) → \(nextOutputWidth)x\(nextOutputHeight), "
        + "anime4k \(wantsScaling ? "off" : configuration.anime4kProfile)"

      stateLock.lock()
      let oldScaler = self.scaler
      let oldInterpolator = self.interpolator
      self.scaler = nextScaler
      self.interpolator = nextInterpolator
      self.toBGRAConverter = nextConverter
      self.width = width
      self.height = height
      self.outputWidth = nextOutputWidth
      self.outputHeight = nextOutputHeight
      self.scaleFactor = wantsScaling ? requestedScale : 1.0
      frames.clear()
      previousBuffer = nil
      previousTime = .invalid
      submittedCount = 0
      failureCount = 0
      lastFailure = ""
      submittedFrames = 0
      interpolateCalls = 0
      emitCount = 0
      pushCount = 0
      emitSrc = 0
      emitInterp = 0
      pushSrc = 0
      pushInterp = 0
      blankBudget.reset()
      _isActive = true
      baseStatus = readyStatus
      stateLock.unlock()

      oldScaler?.invalidate()
      oldInterpolator?.invalidate()

      NSLog("IOS26VideoEnhancer: \(readyStatus)")
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
      stateLock.lock()
      submittedFrames += 1
      let activeInterpolator = interpolator
      let wantsScaling = scaler != nil
      let bgraConverter = toBGRAConverter
      stateLock.unlock()

      // 1. Take ownership of the decoded frame. The GL texture buffers are
      //    recycled by TextureHW, so anything kept across frames must be a copy.
      guard let owned = processor.process(source) else {
        recordFailure("frame copy failed")
        return
      }

      // 2. Emit the decoded frame as untagged 32BGRA. When VT scaling runs,
      //    VTFrameScaler converts tiles to BT.709 420v via Metal and composites
      //    back to 32BGRA with the exact inverse matrix.
      emit(owned, at: presentationTime, origin: .decoded)

      let previous = previousBuffer
      let previousPTS = previousTime
      previousBuffer = owned
      previousTime = presentationTime
      submittedCount += 1

      // 3. Emit the interpolated frame between the previous and the current one.
      guard let interpolator = activeInterpolator, let previous = previous,
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

      stateLock.lock()
      interpolateCalls += 1
      stateLock.unlock()

      interpolator.interpolate(
        previous: previous,
        previousTime: previousPTS,
        source: owned,
        sourceTime: presentationTime,
        destination: destination
      ) { [weak self] result in
        guard let self = self, self.isActive else {
          return
        }
        switch result {
        case .failure(let error):
          self.recordFailure("\(error)")
        case .success:
          if wantsScaling {
            self.emit(destination, at: middleTime, origin: .interpolated)
          } else {
            guard let bgra = bgraConverter?.convert(destination) else {
              self.recordFailure("interpolation BGRA conversion failed")
              return
            }
            if let processor = self.processor, processor.hasAnime4K {
              let output = processor.process(bgra) ?? bgra
              self.emit(output, at: middleTime, origin: .interpolated)
            } else {
              self.emit(bgra, at: middleTime, origin: .interpolated)
            }
          }
        }
      }
    }

    func latestFrame() -> CVPixelBuffer? {
      return frames.latest()?.pixelBuffer
    }

    func newestFrame() -> CVPixelBuffer? {
      return frames.newest()?.pixelBuffer
    }

    func newestEnhanceFrame() -> EnhanceFrame? {
      return frames.newest()
    }

    func frame(nearestTo time: CMTime) -> CVPixelBuffer? {
      return frames.nearest(to: time)?.pixelBuffer
    }

    func enhanceFrame(nearestTo time: CMTime) -> EnhanceFrame? {
      return frames.nearest(to: time)
    }

    func invalidate() {
      stateLock.lock()
      _isActive = false
      let oldInterpolator = interpolator
      let oldScaler = scaler
      interpolator = nil
      scaler = nil
      toBGRAConverter = nil
      stateLock.unlock()

      frames.clear()
      previousBuffer = nil
      previousTime = .invalid
      oldInterpolator?.invalidate()
      oldScaler?.invalidate()
      processor?.invalidate()
    }

    // MARK: - Internals

    /// Publishes one frame, upscaling it first when VT scaling is active.
    ///
    /// The scaler is asynchronous, but submissions are already serialised on
    /// the render worker and the wait is bounded; doing it synchronously keeps
    /// frame ordering deterministic and stops the destination pool from
    /// starving. Either way the published buffer is consumed directly by
    /// Flutter's iOS engine — there is no 420v -> BGRA pass.
    private func emit(
      _ buffer: CVPixelBuffer,
      at time: CMTime,
      origin: FrameOrigin
    ) {
      stateLock.lock()
      emitCount += 1
      switch origin {
      case .decoded: emitSrc += 1
      case .interpolated: emitInterp += 1
      }
      let activeScaler = scaler
      stateLock.unlock()

      guard let scaler = activeScaler else {
        publish(buffer, at: time, origin: origin)
        return
      }

      guard let destination = scaler.makeDestinationBuffer() else {
        recordFailure("scale destination unavailable")
        publish(buffer, at: time, origin: origin)
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

      if semaphore.wait(timeout: .now() + 0.5) == .timedOut {
        recordFailure("scaler timed out")
        publish(buffer, at: time, origin: origin)
        return
      }

      lock.lock()
      let recorded = failure
      lock.unlock()
      if let recorded = recorded {
        recordFailure("scale: \(recorded)")
        publish(buffer, at: time, origin: origin)
        return
      }

      publish(destination, at: time, origin: origin)
    }

    /// Publishes a frame for presentation.
    ///
    /// Every enhanced path publishes untagged `32BGRA` matching plain
    /// `TextureHW`, so Flutter displays consistent colors without BT.601 shifts.
    private func publish(
      _ buffer: CVPixelBuffer,
      at time: CMTime,
      origin: FrameOrigin
    ) {
      stateLock.lock()
      let active = _isActive
      stateLock.unlock()

      guard active else {
        return
      }

      // Sampled, not per frame: locking the base address of a GPU written
      // IOSurface is a synchronisation point, and the VT stages already probe
      // their own 420v output (see `BlankCheckBudget`).
      if checkBlankIfNeeded(buffer) {
        recordFailure(
          "published frame is blank (\(PixelBufferConverter.sampleCenter(buffer)))"
        )
        return
      }

      stateLock.lock()
      guard _isActive else {
        stateLock.unlock()
        return
      }
      frames.push(EnhanceFrame(pixelBuffer: buffer, presentationTime: time))
      pushCount += 1
      switch origin {
      case .decoded: pushSrc += 1
      case .interpolated: pushInterp += 1
      }
      stateLock.unlock()
    }

    /// Thread safe wrapper around `BlankCheckBudget` that probes without
    /// holding `stateLock` across CPU pixel-buffer lock/sampling.
    private func checkBlankIfNeeded(_ buffer: CVPixelBuffer) -> Bool {
      stateLock.lock()
      let needs = blankBudget.needsCheck()
      if !needs {
        blankBudget.recordSkipped()
      }
      stateLock.unlock()
      guard needs else {
        return false
      }
      let isBlank = PixelBufferConverter.isBlank(buffer)
      stateLock.lock()
      blankBudget.recordCheckResult(isBlank: isBlank)
      stateLock.unlock()
      return isBlank
    }

    /// Keeps the most recent failure in `statusDescription` so the app can show
    /// it instead of "enhancement is on but nothing changed".
    private func recordFailure(_ reason: String) {
      stateLock.lock()
      failureCount += 1
      let count = failureCount
      let w = width
      let h = height
      let changed = reason != lastFailure
      if changed {
        lastFailure = reason
        baseStatus =
          "active but failing: \(w)x\(h) — \(reason) "
          + "(\(count) failures)"
      }
      let message = baseStatus
      stateLock.unlock()

      if changed {
        NSLog("IOS26VideoEnhancer: \(message)")
      }
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
