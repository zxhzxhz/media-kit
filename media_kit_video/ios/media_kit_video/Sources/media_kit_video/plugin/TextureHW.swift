import CoreMedia
import Flutter
import OpenGLES
import QuartzCore

#if SWIFT_PACKAGE
  import Mpv
#endif

public class TextureHW: NSObject, FlutterTexture, ResizableTextureProtocol {
  public typealias UpdateCallback = () -> Void

  private let handle: OpaquePointer
  private let updateCallback: UpdateCallback
  private let notifyCallback: UpdateCallback
  private let context: EAGLContext
  private let textureCache: CVOpenGLESTextureCache
  private var renderContext: OpaquePointer?
  private var textureContexts = SwappableObjectManager<TextureGLESContext>(
    objects: [],
    skipCheckArgs: true
  )

  // MARK: Enhancement pipeline

  /// Requested enhancement configuration (nil = plain rendering).
  private var enhanceConfiguration: VideoEnhanceConfiguration?
  /// Active pipeline. Only created when the device can run what was requested.
  private var enhancer: FrameEnhancerProtocol?
  /// Size of the buffers mpv renders into; also the size handed to Flutter.
  private var currentSize: CGSize = .zero
  /// Monotonic frame counter, exposed to the enhancer for logging/ordering.
  private var frameIndex: Int64 = 0
  /// How many times `render` ran, so the status can tell "mpv never delivered a
  /// frame" from "frames arrive but the enhancer skips them".
  private var renderCount: Int64 = 0
  /// Frame the display link picked for the current clock. Guarded by
  /// \`presentationLock\` because the render worker writes it while the Flutter
  /// raster thread reads it from \`copyPixelBuffer\`.
  private var selectedBuffer: CVPixelBuffer?
  private let presentationLock = NSLock()
  /// Last buffer handed to Flutter, used to count *distinct* presented frames.
  private var lastPresentedBuffer: CVPixelBuffer?
  private var presentedFrameCount: Int64 = 0
  /// Anchor for the presentation clock: the PTS of the most recent frame handed
  /// to the enhancer, plus the host time at which that happened.
  ///
  /// Sampling `time-pos` at display-link time and comparing it against frame
  /// timestamps taken at render time mixes two different instants: the clock
  /// always reads up to one frame interval *ahead* of the frame it belongs to,
  /// so `nearest` kept picking the newest queued frame and the interpolated one
  /// was never shown (interpolation "active" at a flat 24 fps). Anchoring the
  /// clock to the last submitted frame removes that offset entirely.
  private var clockAnchorPTS: CMTime = .invalid
  private var clockAnchorHostTime: CFTimeInterval = 0
  /// PTS of the previously submitted frame. Used as the clock anchor so
  /// presentation runs behind the decoder — see `playbackClock()`.
  private var lastSubmittedPTS: CMTime = .invalid
  /// PTS of the frame before that: presentation lags *two* frames.
  private var previousSubmittedPTS: CMTime = .invalid
  /// Smoothed interval between submitted frames, used to schedule the
  /// half-frame presentation tick.
  private var frameIntervalSeconds: Double = 1.0 / 24.0
  /// Number of times Flutter pulled a frame, for diagnostics.
  private var copyCount: Int64 = 0
  /// Monotonic timestamp handed to the enhancer with each frame.
  private var syntheticPTS: Double = 0
  private var lastRenderHostTime: CFTimeInterval = 0
  /// Human readable state of the enhancement pipeline.
  private var enhanceStatus: String = "disabled"

  init(
    handle: OpaquePointer,
    configuration: VideoEnhanceConfiguration?,
    updateCallback: @escaping UpdateCallback,
    notifyCallback: @escaping UpdateCallback
  ) {
    self.handle = handle
    self.enhanceConfiguration = configuration
    self.updateCallback = updateCallback
    self.notifyCallback = notifyCallback
    self.context = OpenGLESHelpers.createContext()
    self.textureCache = OpenGLESHelpers.createTextureCache(context)

    super.init()

    self.initMPV()
  }

  deinit {
    enhancer?.invalidate()
    enhancer = nil
    disposePixelBuffer()
    disposeMPV()
    OpenGLESHelpers.deleteTextureCache(textureCache)

    // Deleting the context may cause potential RAM or VRAM memory leaks, as it
    // is used in the `deinit` method of the `TextureGLESContext`.
    // Potential fix: use a counter, and delete it only when the counter reaches
    // zero
    OpenGLESHelpers.deleteContext(context)
  }

  public func copyPixelBuffer() -> Unmanaged<CVPixelBuffer>? {
    // The display link picks the frame that belongs to the current clock. Until
    // it has picked one the freshly rendered GL buffer is used, which keeps the
    // first frames and the non-enhanced path working unchanged.
    presentationLock.lock()
    let selected = selectedBuffer
    presentationLock.unlock()

    let buffer = selected ?? textureContexts.current?.pixelBuffer
    guard let buffer = buffer else {
      return nil
    }

    // Count distinct frames so Dart can show the real presentation rate.
    presentationLock.lock()
    if lastPresentedBuffer !== buffer {
      lastPresentedBuffer = buffer
      presentedFrameCount += 1
    }
    copyCount += 1
    presentationLock.unlock()

    return Unmanaged.passRetained(buffer)
  }

  /// Half the current frame interval, used to schedule the mid-frame
  /// presentation tick.
  public func halfFrameInterval() -> Double {
    presentationLock.lock()
    defer {
      presentationLock.unlock()
    }
    return max(frameIntervalSeconds * 0.5, 0.004)
  }

  /// Number of `copyPixelBuffer` calls, for diagnostics.
  public func copyCounter() -> Int64 {
    presentationLock.lock()
    defer {
      presentationLock.unlock()
    }
    return copyCount
  }

  /// \`true\` once an enhancement pipeline is configured and running.
  public var hasActiveEnhancer: Bool {
    return enhancer?.isActive ?? false
  }

  /// \`true\` once the display link has taken over presentation. From that moment
  /// \`VideoOutput\` must not notify Flutter on every mpv frame, otherwise the
  /// pacing decided here is overwritten.
  public var isDrivingPresentation: Bool {
    presentationLock.lock()
    defer {
      presentationLock.unlock()
    }
    return selectedBuffer != nil
  }

  /// Picks the enhanced frame that belongs to \`clock\`.
  ///
  /// Returns \`true\` when the frame to present changed, so the caller knows it
  /// has to wake Flutter. Runs on the render worker.
  public func selectFrame(at clock: CMTime) -> Bool {
    guard let enhancer = enhancer, enhancer.isActive else {
      return false
    }
    // Publish the newest frame instead of resolving against the clock.
    //
    // Clock based selection needs the presentation clock to be frame accurate,
    // and it is not: ProMotion throttles the whole app to the video's 24 Hz, so
    // the half-frame tick fires before VT has produced the interpolated frame
    // and the clock has already moved past it by the time it arrives. Taking
    // whatever is newest means an interpolated frame is shown as soon as it
    // exists, which is the only way it can ever be visible under a coarse
    // refresh clock.
    guard let next = enhancer.latestFrame() else {
      return false
    }

    presentationLock.lock()
    let changed = selectedBuffer !== next
    selectedBuffer = next
    presentationLock.unlock()

    return changed
  }

  /// Presentation clock anchored to the last submitted frame.
  ///
  /// This is the clock the display link should use: it advances in the same
  /// time base as the frame timestamps, so the frame picked for a given instant
  /// is the one that actually belongs there.
  public func playbackClock() -> CMTime {
    presentationLock.lock()
    let anchor = clockAnchorPTS
    let hostTime = clockAnchorHostTime
    presentationLock.unlock()

    guard anchor.isValid, hostTime > 0 else {
      return .invalid
    }
    let elapsed = min(max(CACurrentMediaTime() - hostTime, 0), 0.12)
    return CMTimeAdd(
      anchor,
      CMTime(seconds: elapsed, preferredTimescale: 1000)
    )
  }

  /// Human readable enhancement state (see \`FrameEnhancerProtocol.statusDescription\`).
  public func enhanceStatusDescription() -> String {
    // Read live instead of returning the string captured in
    // `configureEnhancer`: a cached snapshot made every counter look frozen at
    // its configure-time value, hiding the difference between "never
    // submitted" and "submitted but skipped".
    presentationLock.lock()
    let renders = renderCount
    let submits = frameIndex
    presentationLock.unlock()

    var text = enhanceStatus
    if let enhancer = enhancer, enhancer.isActive {
      text = enhancer.statusDescription
    }
    return "\(text) | renders=\(renders) submits=\(submits)"
  }

  /// Number of distinct frames handed to Flutter since playback started.
  public func presentedFrames() -> Int64 {
    presentationLock.lock()
    defer {
      presentationLock.unlock()
    }
    return presentedFrameCount
  }

  /// Forgets the presentation state, e.g. after a seek or a size change.
  private func resetPresentation() {
    presentationLock.lock()
    selectedBuffer = nil
    lastPresentedBuffer = nil
    clockAnchorPTS = .invalid
    clockAnchorHostTime = 0
    lastSubmittedPTS = .invalid
    previousSubmittedPTS = .invalid
    frameIntervalSeconds = 1.0 / 24.0
    copyCount = 0
    syntheticPTS = 0
    lastRenderHostTime = 0
    presentationLock.unlock()
  }

  public func dispose() {
    enhancer?.invalidate()
    enhancer = nil
    resetPresentation()
    disposeMPV()
  }

  /// Hot-swaps the enhancement pipeline. Must be called on the render worker.
  public func setEnhancePipeline(_ configuration: VideoEnhanceConfiguration?) {
    enhanceConfiguration = configuration
    // The frame the display link picked belongs to the old pipeline: drop it so
    // the next tick starts from a clean state.
    resetPresentation()
    configureEnhancer(currentSize)
  }

  private func initMPV() {
    EAGLContext.setCurrent(context)
    defer {
      OpenGLESHelpers.checkError("initMPV")
      EAGLContext.setCurrent(nil)
    }

    let api = UnsafeMutableRawPointer(
      mutating: (MPV_RENDER_API_TYPE_OPENGL as NSString).utf8String
    )
    var procAddress = mpv_opengl_init_params(
      get_proc_address: {
        (ctx, name) in
        return TextureHW.getProcAddress(ctx, name)
      },
      get_proc_address_ctx: nil
    )

    var params: [mpv_render_param] = withUnsafeMutableBytes(of: &procAddress) {
      procAddress in
      return [
        mpv_render_param(type: MPV_RENDER_PARAM_API_TYPE, data: api),
        mpv_render_param(
          type: MPV_RENDER_PARAM_OPENGL_INIT_PARAMS,
          data: procAddress.baseAddress.map {
            UnsafeMutableRawPointer($0)
          }
        ),
        mpv_render_param(),
      ]
    }

    MPVHelpers.checkError(
      media_kit_mpv_render_context_create(&renderContext, handle, &params)
    )

    media_kit_mpv_render_context_set_update_callback(
      renderContext,
      { (ctx) in
        let that = unsafeBitCast(ctx, to: TextureHW.self)
        DispatchQueue.main.async {
          that.updateCallback()
        }
      },
      UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque())
    )
  }

  private func disposeMPV() {
    guard let renderContext = renderContext else { return }
    EAGLContext.setCurrent(context)
    defer {
      OpenGLESHelpers.checkError("disposeMPV")
      EAGLContext.setCurrent(nil)
    }

    media_kit_mpv_render_context_set_update_callback(renderContext, nil, nil)
    media_kit_mpv_render_context_free(renderContext)
    self.renderContext = nil
  }

  public func resize(_ size: CGSize) {
    if size.width == 0 || size.height == 0 {
      return
    }

    NSLog("TextureGL: resize: \(size.width)x\(size.height)")
    createPixelBuffer(size)
    currentSize = size
    resetPresentation()
    configureEnhancer(size)
  }

  private func createPixelBuffer(_ size: CGSize) {
    disposePixelBuffer()

    textureContexts.reinit(
      objects: [
        TextureGLESContext(
          context: context,
          textureCache: textureCache,
          size: size
        ),
        TextureGLESContext(
          context: context,
          textureCache: textureCache,
          size: size
        ),
        TextureGLESContext(
          context: context,
          textureCache: textureCache,
          size: size
        ),
      ],
      skipCheckArgs: true
    )
  }

  private func disposePixelBuffer() {
    textureContexts.reinit(objects: [], skipCheckArgs: true)
  }

  // MARK: - Enhancement

  /// Builds (or rebuilds) the pipeline for the given frame size.
  ///
  /// Compiling Anime4K takes hundreds of milliseconds and runs on the render
  /// worker, so it only happens when the size or the configuration changes.
  private func configureEnhancer(_ size: CGSize) {
    guard size.width > 0, size.height > 0 else {
      return
    }

    guard let configuration = enhanceConfiguration else {
      enhancer?.invalidate()
      enhancer = nil
      enhanceStatus = "disabled"
      return
    }

    if enhancer == nil {
      var reason = ""
      enhancer = VideoEnhancerFactory.make(
        configuration: configuration,
        reason: &reason
      )
      enhancer?.onFrameAvailable = { [weak self] in
        guard let self = self else { return }
        // An enhanced frame (usually the interpolated one) is ready: publish it
        // immediately. Waiting for the next mpv frame lets it be overwritten by
        // the next source frame before it is ever shown.
        _ = self.selectFrame(at: .invalid)
        self.notifyCallback()
      }
      if enhancer == nil {
        enhanceStatus = "unavailable: \(reason)"
        NSLog("TextureHW: no enhancement pipeline available — \(reason)")
        return
      }
    }

    guard let enhancer = enhancer else {
      enhanceStatus = "unavailable"
      return
    }

    // \`outputSize\` feeds the Anime4K \`OUTPUT\` variable used by the
    // AutoDownscalePre passes. The Flutter video surface size is not known
    // here, so the decoded size is used; the CNN is then downsampled back to
    // it by the finalize kernel.
    let ready = enhancer.configure(
      configuration: configuration,
      inputSize: size,
      outputSize: size
    )

    if !ready {
      enhanceStatus = "rejected: \(enhancer.statusDescription)"
      NSLog("TextureHW: \(enhanceStatus); using plain rendering")
      enhancer.invalidate()
      self.enhancer = nil
    } else {
      enhanceStatus = enhancer.statusDescription
      NSLog("TextureHW: enhancement ready — \(enhanceStatus)")
    }
  }

  private func currentPresentationTime() -> CMTime {
    // `time-pos` is quantised by mpv's property update rate and can return the
    // *same* value for two consecutive frames. Identical timestamps make an
    // interpolated frame indistinguishable from its source, so the display
    // link keeps picking the source and the presentation rate stays at the
    // container frame rate (24 fps with interpolation reporting success).
    //
    // Deriving the timestamp from the host clock delta between renders keeps
    // every frame strictly ordered, and it shares the time base of
    // `playbackClock()` (which advances from the same anchor).
    let now = CACurrentMediaTime()
    if lastRenderHostTime > 0 {
      let delta = now - lastRenderHostTime
      // Ignore stalls (seek, backgrounding) instead of jumping the clock.
      if delta > 0 && delta < 0.5 {
        syntheticPTS += delta
      }
    }
    lastRenderHostTime = now
    return CMTime(seconds: syntheticPTS, preferredTimescale: 1000)
  }

  public func render(_ size: CGSize) {
    guard renderContext != nil else { return }
    presentationLock.lock()
    renderCount += 1
    presentationLock.unlock()
    let textureContext = textureContexts.nextAvailable()
    if textureContext == nil {
      return
    }

    EAGLContext.setCurrent(context)
    defer {
      OpenGLESHelpers.checkError("render")
      EAGLContext.setCurrent(nil)
    }

    glBindFramebuffer(GLenum(GL_FRAMEBUFFER), textureContext!.frameBuffer)
    defer {
      glBindFramebuffer(GLenum(GL_FRAMEBUFFER), 0)
    }

    var fbo = mpv_opengl_fbo(
      fbo: Int32(textureContext!.frameBuffer),
      w: Int32(size.width),
      h: Int32(size.height),
      internal_format: 0
    )
    let fboPtr = withUnsafeMutablePointer(to: &fbo) { $0 }

    var params: [mpv_render_param] = [
      mpv_render_param(type: MPV_RENDER_PARAM_OPENGL_FBO, data: fboPtr),
      mpv_render_param(type: MPV_RENDER_PARAM_INVALID, data: nil),
    ]

    media_kit_mpv_render_context_render(renderContext, &params)

    glFlush()

    textureContexts.pushAsReady(textureContext!)

    // Hand the freshly rendered buffer to the enhancement pipeline. \`submit\` is
    // documented to copy whatever it needs, so the GL buffer can be recycled.
    if let enhancer = enhancer, enhancer.isActive {
      frameIndex += 1
      let presentationTime = currentPresentationTime()
      // Present two frames behind the newest decoded frame.
      //
      // The interpolated frame sits between the previous and the current frame.
      // With a one-frame lag the clock reaches the midpoint only 31.25 ms into
      // a 41.67 ms interval, so a VT pipeline that takes 32~40 ms produces the
      // frame *after* the selection window has passed: it arrives already
      // expired, is never selected, and nothing reports a failure. Two frames
      // of lag give the asynchronous pipeline a full interval.
      presentationLock.lock()
      if lastSubmittedPTS.isValid {
        let delta = CMTimeGetSeconds(
          CMTimeSubtract(presentationTime, lastSubmittedPTS)
        )
        if delta > 0.005 && delta < 0.2 {
          frameIntervalSeconds = delta
        }
      }
      if previousSubmittedPTS.isValid {
        clockAnchorPTS = previousSubmittedPTS
      } else if lastSubmittedPTS.isValid {
        clockAnchorPTS = lastSubmittedPTS
      } else {
        clockAnchorPTS = presentationTime
      }
      clockAnchorHostTime = CACurrentMediaTime()
      previousSubmittedPTS = lastSubmittedPTS
      lastSubmittedPTS = presentationTime
      presentationLock.unlock()
      enhancer.submit(
        textureContext!.pixelBuffer,
        presentationTime: presentationTime,
        frameIndex: frameIndex
      )
    }
  }

  static private func getProcAddress(
    _ ctx: UnsafeMutableRawPointer?,
    _ name: UnsafePointer<Int8>?
  ) -> UnsafeMutableRawPointer? {
    let symbol: CFString = CFStringCreateWithCString(
      kCFAllocatorDefault,
      name,
      kCFStringEncodingASCII
    )
    let indentifier = CFBundleGetBundleWithIdentifier(
      "com.apple.opengles" as CFString
    )
    let addr = CFBundleGetFunctionPointerForName(indentifier, symbol)

    if addr == nil {
      NSLog("Cannot get OpenGLES function pointer!")
    }
    return addr
  }
}
