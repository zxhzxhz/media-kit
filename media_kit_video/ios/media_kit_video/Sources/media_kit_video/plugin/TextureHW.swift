import CoreMedia
import Flutter
import OpenGLES

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
    // Enhanced frames take priority. The queue hands out the newest finished
    // frame without consuming it, so repeated pulls within one vsync return the
    // same buffer (Flutter may pull several times per frame).
    if let enhancer = enhancer, let enhanced = enhancer.latestFrame() {
      return Unmanaged.passRetained(enhanced)
    }

    let textureContext = textureContexts.current
    if textureContext == nil {
      return nil
    }

    return Unmanaged.passRetained(textureContext!.pixelBuffer)
  }

  public func dispose() {
    enhancer?.invalidate()
    enhancer = nil
    disposeMPV()
  }

  /// Hot-swaps the enhancement pipeline. Must be called on the render worker.
  public func setEnhancePipeline(_ configuration: VideoEnhanceConfiguration?) {
    enhanceConfiguration = configuration
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
      return
    }

    if enhancer == nil {
      enhancer = VideoEnhancerFactory.make(configuration: configuration)
      enhancer?.onFrameAvailable = { [weak self] in
        // An enhanced frame (e.g. an interpolated one) is ready without mpv
        // having produced anything: notify Flutter without re-rendering.
        self?.notifyCallback()
      }
    }

    guard let enhancer = enhancer else {
      NSLog("TextureHW: no enhancement pipeline available; using plain rendering")
      self.enhancer = nil
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
      NSLog(
        "TextureHW: enhancement pipeline rejected the configuration (\(configuration.anime4kProfile)); using plain rendering"
      )
      enhancer.invalidate()
      self.enhancer = nil
    }
  }

  private func currentPresentationTime() -> CMTime {
    // \`time-pos\` is the playback position mpv is showing. It is a far better
    // anchor than a wall clock because it survives seeks and rate changes.
    if let position = MPVHelpers.getDoubleProperty(handle, "time-pos") {
      return CMTime(seconds: position, preferredTimescale: 1000)
    }
    return CMClockGetTime(CMClockGetHostTimeClock())
  }

  public func render(_ size: CGSize) {
    guard renderContext != nil else { return }
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
      enhancer.submit(
        textureContext!.pixelBuffer,
        presentationTime: currentPresentationTime(),
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
