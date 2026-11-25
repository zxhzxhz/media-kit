import CoreGraphics
import CoreMedia
import Foundation
import QuartzCore

#if canImport(UIKit)
  import UIKit
#endif

#if canImport(Flutter)
  import Flutter
#elseif canImport(FlutterMacOS)
  import FlutterMacOS
#endif

// This class creates and manipulates the different types of FlutterTexture,
// handles resizing, rendering calls, and notify Flutter when a new frame is
// available to render.
//
// To improve the user experience, a worker is used to execute heavy tasks on a
// dedicated thread.
public class VideoOutput: NSObject {
  // Will be called on the main thread
  public typealias TextureUpdateCallback = (Int64, CGSize) -> Void

  private static let isSimulator: Bool = {
    let isSim: Bool
    #if targetEnvironment(simulator)
      isSim = true
    #else
      isSim = false
    #endif
    return isSim
  }()

  private let handle: OpaquePointer
  private let enableHardwareAcceleration: Bool
  private let registry: FlutterTextureRegistry
  private let textureUpdateCallback: TextureUpdateCallback
  private let worker: Worker = .init()
  private var width: Int64?
  private var height: Int64?
  private var texture: ResizableTextureProtocol!
  private var textureId: Int64 = -1
  private var currentSize: CGSize = CGSize.zero
  private var disposed: Bool = false
  // Requested iOS enhancement pipeline (nil on every other platform).
  private var enhanceConfiguration: VideoEnhanceConfiguration?
  // Kept so that the enhancement pipeline can be reconfigured at runtime
  // without recreating the texture.
  private var hardwareTexture: TextureHW?

  #if os(iOS)
    // Drives presentation directly from ProMotion when an interpolated
    // enhancement pipeline is active.
    private var displayLink: CADisplayLink?
  #endif
  // Guards the diagnostic counters below: they are written from the display
  // link callback (main thread) and from the render worker.
  private let tickLock = NSLock()
  // Playback clock sampling (see `currentPlaybackClock`).
  private var clockBaseSeconds: Double = 0
  private var clockBaseHostTime: CFTimeInterval = 0
  private var clockSampleHostTime: CFTimeInterval = 0
  // Per-second rates for diagnostics: presentation changes and how often
  // Flutter actually pulled a texture.
  private var selectCount: Int = 0
  private var selectRate: Int = 0
  private var copyRate: Int64 = 0
  private var copyRateBase: Int64 = 0
  private var lastRateReset: CFTimeInterval = 0
  // Display link callbacks, counted straight in the callback (no worker hop,
  // no coalescing), plus the frame interval the system actually granted.
  // `dl=24@41.67ms` means the system really granted 24 Hz; `dl=120@8.33ms`
  // with a low `copy` means Flutter is the one dropping the wake up.
  private var rawTickCount: Int = 0
  private var displayLinkRate: Int = 0
  private var displayLinkIntervalMs: Double = 0
  // Texture notifications actually delivered to Flutter, per second. Compared
  // against `copyRate` this says whether Flutter dropped the wake up or never
  // scheduled a frame for it.
  private var notifyCount: Int = 0
  private var notifyBase: Int = 0
  private var notifyRate: Int = 0

  init(
    handle: Int64,
    configuration: VideoOutputConfiguration,
    registry: FlutterTextureRegistry,
    textureUpdateCallback: @escaping TextureUpdateCallback
  ) {
    let handle = OpaquePointer(bitPattern: Int(handle))
    assert(handle != nil, "handle casting")

    self.handle = handle!
    width = configuration.width
    height = configuration.height
    enableHardwareAcceleration = configuration.enableHardwareAcceleration
    enhanceConfiguration = configuration.enhance
    self.registry = registry
    self.textureUpdateCallback = textureUpdateCallback

    super.init()

    worker.enqueue {
      self._init()
    }
  }

  deinit {
    worker.cancel()

    disposed = true
    disposeTextureId()
  }

  public func setSize(width: Int64?, height: Int64?) {
    worker.enqueue {
      guard !self.disposed else { return }
      self.width = width
      self.height = height
    }
  }

  /// Hot-swaps the enhancement pipeline. `nil` disables it.
  ///
  /// Runs on the render worker so that it is ordered against `render` calls.
  public func setEnhancePipeline(_ configuration: VideoEnhanceConfiguration?) {
    enhanceConfiguration = configuration
    worker.enqueue {
      guard !self.disposed else { return }
      #if os(iOS)
        self.hardwareTexture?.setEnhancePipeline(configuration)
      #endif
    }
  }

  public func dispose(completion: @escaping () -> Void) {
    stopDisplayLink()
    // Drain rendering on its worker without blocking the platform thread,
    // which pending render jobs may still need for texture notifications.
    worker.enqueue {
      self.disposed = true
      self.texture?.dispose()
      self.disposeTextureId {
        self.worker.cancel()
        completion()
      }
    }
  }

  private func updateRatesIfNeeded() {
    let now = CACurrentMediaTime()
    tickLock.lock()
    if lastRateReset == 0 {
      lastRateReset = now
      tickLock.unlock()
      return
    }
    guard now - lastRateReset >= 1.0 else {
      tickLock.unlock()
      return
    }
    lastRateReset = now
    let currentSelect = selectCount
    selectCount = 0
    let currentTicks = rawTickCount
    rawTickCount = 0
    let currentNotify = notifyCount
    tickLock.unlock()

    #if os(iOS)
      let totalCopies = hardwareTexture?.copyCounter() ?? 0
    #endif

    tickLock.lock()
    selectRate = currentSelect
    displayLinkRate = currentTicks
    notifyRate = max(0, currentNotify - notifyBase)
    notifyBase = currentNotify
    #if os(iOS)
      copyRate = max(0, totalCopies - copyRateBase)
      copyRateBase = totalCopies
    #endif
    tickLock.unlock()
  }

  /// Human readable enhancement state, surfaced to Dart for diagnostics.
  ///
  /// `dl` / `dlMs` come from the display link's own `targetTimestamp`, counted
  /// before any coalescing, so they cannot be polluted by a slow worker:
  /// `dl=24@41.67ms` means the system really granted 24 Hz, while
  /// `dl=120@8.33ms` with a low `copy` means Flutter is dropping the wake up.
  public func enhanceStatus() -> String {
    #if os(iOS)
      var text =
        hardwareTexture?.enhanceStatusDescription()
        ?? "unavailable (software rendering)"
      tickLock.lock()
      let sel = selectRate
      let copy = copyRate
      let notify = notifyRate
      let dl = displayLinkRate
      let interval = String(format: "%.2f", displayLinkIntervalMs)
      tickLock.unlock()
      text +=
        " | hz sel=\(sel) copy=\(copy)"
        + " notify=\(notify) dl=\(dl)@\(interval)ms"
      return text
    #else
      return "unavailable (not iOS)"
    #endif
  }

  /// Number of distinct frames handed to Flutter since playback started.
  public func presentedFrames() -> Int64 {
    #if os(iOS)
      return hardwareTexture?.presentedFrames() ?? 0
    #else
      return 0
    #endif
  }

  private func _init() {
    let enableHardwareAcceleration =
      VideoOutput.isSimulator ? false : enableHardwareAcceleration

    NSLog(
      "VideoOutput: enableHardwareAcceleration: \(enableHardwareAcceleration)"
    )

    if VideoOutput.isSimulator {
      NSLog(
        "VideoOutput: warning: hardware rendering is disabled in the iOS simulator, due to an incompatibility with OpenGL ES"
      )
    }

    let update: () -> Void = { [weak self] in
      self?.updateCallback()
    }
    var hardwareTexture: ResizableTextureProtocol?
    if enableHardwareAcceleration {
      #if os(iOS)
        // Presentation is driven from `_updateCallback` and the half frame
        // timer only; the texture does not wake Flutter on its own.
        let texture = TextureHW(
          handle: handle,
          configuration: enhanceConfiguration,
          updateCallback: update
        )
        self.hardwareTexture = texture
        hardwareTexture = texture
      #else
        hardwareTexture = TextureHW(
          handle: handle,
          updateCallback: update
        )
      #endif
    }
    if let hardwareTexture = hardwareTexture {
      texture = SafeResizableTexture(hardwareTexture)
    } else {
      if enableHardwareAcceleration {
        NSLog("VideoOutput: hardware rendering unavailable; using software rendering")
      }
      texture = SafeResizableTexture(
        TextureSW(
          handle: handle,
          updateCallback: update
        )
      )
    }

    DispatchQueue.main.sync { [weak self]() in
      guard let that = self else {
        return
      }
      that.registerTextureId()
    }

    // The display link runs for the whole lifetime of the output: it is a no-op
    // until an enhancement pipeline takes over presentation.
    startDisplayLink()
  }

  // MARK: - Display link pacing

  private func startDisplayLink() {
    #if os(iOS)
      DispatchQueue.main.async { [weak self] in
        guard let self = self, self.displayLink == nil, !self.disposed else {
          return
        }
        let link = CADisplayLink(
          target: self,
          selector: #selector(self.handleDisplayTick(_:))
        )
        // 48~120 keeps ProMotion responsive; the value is a request, the system
        // decides the real rate, so the pacing logic never assumes 120 Hz.
        // The property itself is iOS 15+, the deployment target is 13.0.
        if #available(iOS 15.0, *) {
          link.preferredFrameRateRange = CAFrameRateRange(
            minimum: 48,
            maximum: 120,
            preferred: 120
          )
        }
        link.add(to: .main, forMode: .common)
        self.displayLink = link
        #if canImport(UIKit)
          // Logged once: if `maximumFramesPerSecond` is 120 while `dl` in the
          // status line stays at 24, the ceiling is the granted range or the
          // system's adaptive sync, not the panel.
          NSLog(
            "VideoOutput: display link started, screen max "
              + "\(UIScreen.main.maximumFramesPerSecond) Hz, requested 48~120 Hz"
          )
        #endif
      }
    #endif
  }

  private func stopDisplayLink() {
    #if os(iOS)
      DispatchQueue.main.async { [weak self] in
        self?.displayLink?.invalidate()
        self?.displayLink = nil
      }
    #endif
  }

  #if os(iOS)
    @objc private func handleDisplayTick(_ link: CADisplayLink) {
      // Count every callback and the granted interval *before* selecting so
      // ProMotion headroom stays visible even when a frame is unchanged.
      tickLock.lock()
      rawTickCount += 1
      let granted = link.targetTimestamp - link.timestamp
      if granted > 0 {
        displayLinkIntervalMs = granted * 1000
      }
      tickLock.unlock()

      guard !disposed else { return }
      updateRatesIfNeeded()

      let tid = textureId
      guard tid >= 0,
        let hardwareTexture = hardwareTexture,
        hardwareTexture.needsHalfFrameTick
      else {
        return
      }

      // Direct presentation from the main-thread display link avoids hopping
      // onto `worker`, which previously dropped half-frame ticks behind
      // synchronous GL + CoreImage + VT work for the next decoded frame.
      if hardwareTexture.selectFrame(at: hardwareTexture.playbackClock()) {
        tickLock.lock()
        selectCount += 1
        notifyCount += 1
        tickLock.unlock()
        registry.textureFrameAvailable(tid)
      }
    }
  #endif

  /// Audio clock of the playing stream, extrapolated between samples.
  ///
  /// \`audio-pts\` is the real audio master clock (and the right anchor for
  /// frame selection); \`time-pos\` is the fallback when there is no audio
  /// track. Sampling at ~15 Hz keeps mpv property locks off the hot path.
  private func currentPlaybackClock() -> CMTime {
    let now = CACurrentMediaTime()
    let needsSample = clockSampleHostTime == 0
      || now - clockSampleHostTime > 0.066
    if needsSample {
      // `time-pos` first: TextureHW stamps every submitted frame with
      // `time-pos`, so the clock and the frame timestamps stay in one domain.
      // Mixing in `audio-pts` shifted the clock by tens of milliseconds and
      // made the display link settle on the oldest queued frame (flat 24 fps).
      if let position = MPVHelpers.getDoubleProperty(handle, "time-pos") {
        clockBaseSeconds = position
        clockBaseHostTime = now
        clockSampleHostTime = now
      } else if let position = MPVHelpers.getDoubleProperty(handle, "audio-pts") {
        clockBaseSeconds = position
        clockBaseHostTime = now
        clockSampleHostTime = now
      } else {
        clockSampleHostTime = now
        return .invalid
      }
    }

    // Clamp the extrapolation: while paused the clock does not advance, and a
    // stale sample must not produce a runaway value.
    let elapsed = min(max(now - clockBaseHostTime, 0), 0.12)
    return CMTime(seconds: clockBaseSeconds + elapsed, preferredTimescale: 1000)
  }

  // Must be run on the main thread
  private func registerTextureId() {
    // Textures must be registered on the platform thread.
    textureId = registry.register(texture)
    // textureUpdateCallback must run on the main thread
    textureUpdateCallback(textureId, CGSize(width: 0, height: 0))
  }

  private func disposeTextureId(completion: @escaping () -> Void = {}) {
    let registry_ = self.registry
    let textureId_ = self.textureId
    textureId = -1
    DispatchQueue.main.async {
      // Textures must be unregistered on the platform thread
      if textureId_ >= 0 {
        registry_.unregisterTexture(textureId_)
      }
      completion()
    }
  }

  public func updateCallback() {
    worker.enqueue {
      self._updateCallback()
    }
  }

  /// Notifies Flutter that a new frame (produced by the enhancement pipeline)
  /// is ready to be pulled. Unlike \`updateCallback\` this does not render a new
  /// mpv frame.
  public func notifyFrameAvailable() {
    worker.enqueue {
      self._notifyTexture()
    }
  }

  /// Must run on the render worker.
  private func notifyOnMain() {
    guard !disposed else { return }
    tickLock.lock()
    notifyCount += 1
    tickLock.unlock()
    // `async`, never `sync`: this runs on the render worker, and a synchronous
    // hop lets the main thread — Flutter's platform thread, busy with danmaku
    // layout and gestures — stall the whole pipeline. It is also a deadlock
    // waiting to happen when the player is torn down mid-frame.
    DispatchQueue.main.async { [weak self] in
      guard let that = self, !that.disposed else { return }
      // Textures must be marked as available from the main thread
      that.registry.textureFrameAvailable(that.textureId)
    }
  }

  private func _notifyTexture() {
    notifyOnMain()
  }

  private func _updateCallback() {
    guard !disposed else { return }
    // Refresh the per-second rates from the render callback: the display link
    // no longer hops onto the worker, so it cannot drive this any more.
    updateRatesIfNeeded()

    let size = videoSize

    if size.width == 0 || size.height == 0 {
      return
    }

    if currentSize != size {
      currentSize = size

      texture.resize(size)
      DispatchQueue.main.sync { [weak self] in
        guard let that = self else { return }
        // textureUpdateCallback must run on the main thread
        that.textureUpdateCallback(that.textureId, size)
      }
    }

    if disposed {
      return
    }

    texture.render(size)

    #if os(iOS)
      if let hardwareTexture = hardwareTexture,
        hardwareTexture.hasActiveEnhancer
      {
        // When interpolation is active, presentation is driven directly from
        // `handleDisplayTick` on the main-thread CADisplayLink so intermediate
        // frames are never starved behind queued worker renders. For non-
        // interpolated modes (e.g. scaling or Anime4K only), present immediately.
        if !hardwareTexture.needsHalfFrameTick {
          if hardwareTexture.selectFrame(at: .invalid) {
            tickLock.lock()
            selectCount += 1
            tickLock.unlock()
            notifyOnMain()
          }
        }
        return
      }
    #endif

    _notifyTexture()
  }

    private var videoSize: CGSize {
        // fixed size
        if width != nil && height != nil {
            return CGSize(
                width: Double(width!),
                height: Double(height!)
            )
        }
        
        let params = MPVHelpers.getVideoOutParams(handle)
        return CGSize(
            width: Double(width ?? (params.rotate == 0 || params.rotate == 180
                                    ? params.dw
                                    : params.dh)),
            height: Double(height ?? (params.rotate == 0 || params.rotate == 180
                                      ? params.dh
                                      : params.dw))
        )
  }
}
