//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Concrete enhancement pipelines.
//
//  * MetalOnlyEnhancer  - Anime4K on Metal. Works on every Metal capable
//                         device, no VideoToolbox requirement.
//  * IOS26VideoEnhancer - VideoToolbox frame interpolation (and optional
//                         Anime4K). iOS 26.0+ only.
//
//  Both are deliberately conservative: any failure disables the stage and
//  leaves the plain TextureHW path untouched, so enhancement can never break
//  playback.

import CoreMedia
import CoreVideo
import Foundation

// MARK: - Anime4K only

/// Runs Anime4K on Metal for every decoded frame.
final class MetalOnlyEnhancer: FrameEnhancerProtocol {
  private var processor: MetalFrameProcessor?
  private let frames = EnhanceFrameQueue(capacity: 3)

  var onFrameAvailable: (() -> Void)?
  private(set) var isActive: Bool = false

  init() {
    processor = MetalFrameProcessor()
  }

  func configure(
    configuration: VideoEnhanceConfiguration,
    inputSize: CGSize,
    outputSize: CGSize
  ) -> Bool {
    // The VideoToolbox stages belong to IOS26VideoEnhancer.
    guard !configuration.interpolation, configuration.vtScaleFactor <= 0 else {
      return false
    }
    guard !configuration.anime4kShaders.isEmpty else {
      return false
    }
    let width = Int(inputSize.width)
    let height = Int(inputSize.height)
    guard width > 0, height > 0 else {
      return false
    }
    guard let processor = processor else {
      return false
    }

    let configured = processor.configure(
      profile: configuration.anime4kProfile,
      glslSources: configuration.anime4kShaders,
      width: width,
      height: height
    )
    guard configured, processor.hasAnime4K else {
      return false
    }

    frames.clear()
    isActive = true
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

  func invalidate() {
    isActive = false
    frames.clear()
    processor?.invalidate()
  }
}

// MARK: - VideoToolbox (iOS 26+)

/// Frame interpolation through VideoToolbox with an optional Anime4K pass.
///
/// NOTE: this is the experimental stage. Frame pacing is not display clock
/// driven yet (that requires the CADisplayLink scheduling described in the
/// design document), so interpolation is off by default and the caller is
/// expected to gate it behind an explicit user opt-in.
@available(iOS 26.0, *)
final class IOS26VideoEnhancer: FrameEnhancerProtocol {
  private var processor: MetalFrameProcessor?
  private var interpolator: VTFrameInterpolator?
  private let frames = EnhanceFrameQueue(capacity: 4)
  private let processingQueue = DispatchQueue(
    label: "media_kit_video.enhance.videotoolbox",
    qos: .userInteractive
  )

  var onFrameAvailable: (() -> Void)?
  private(set) var isActive: Bool = false

  private var width: Int = 0
  private var height: Int = 0
  private var previousBuffer: CVPixelBuffer?
  private var previousTime: CMTime = .invalid
  private var submittedCount: Int64 = 0

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

    guard VTFrameInterpolator.isSupported(width: width, height: height) else {
      NSLog(
        "IOS26VideoEnhancer: VideoToolbox interpolation is not supported for \(width)x\(height)"
      )
      return false
    }

    self.width = width
    self.height = height

    // The Metal processor is always needed: it produces the scratch buffers
    // that survive across frames and, when shaders are configured, runs
    // Anime4K on both the original and the interpolated frames.
    if let processor = processor {
      let ok = processor.configure(
        profile: configuration.anime4kProfile,
        glslSources: configuration.anime4kShaders,
        width: width,
        height: height
      )
      if !ok {
        NSLog("IOS26VideoEnhancer: Metal processor configuration failed")
        return false
      }
    } else {
      return false
    }

    guard let interpolator = VTFrameInterpolator(width: width, height: height)
    else {
      NSLog("IOS26VideoEnhancer: failed to start the VTFrameProcessor session")
      return false
    }
    self.interpolator = interpolator

    frames.clear()
    previousBuffer = nil
    previousTime = .invalid
    submittedCount = 0
    isActive = true

    NSLog(
      "IOS26VideoEnhancer: active (\(width)x\(height), anime4k: \(configuration.anime4kProfile))"
    )
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

    // 1. Take ownership of the decoded frame. The GL texture buffers are
    //    recycled by TextureHW, so anything kept across frames must be a copy.
    guard let owned = processor.process(source) else {
      return
    }

    // 2. Emit the original frame (Anime4K applied by the processor above).
    frames.push(
      EnhanceFrame(pixelBuffer: owned, presentationTime: presentationTime)
    )
    onFrameAvailable?()

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
    guard let destination = processor.makeScratchBuffer() else {
      return
    }

    let middleTime = CMTimeAdd(
      previousPTS,
      CMTimeMultiplyByFloat64(
        CMTimeSubtract(presentationTime, previousPTS),
        multiplier: 0.5
      )
    )

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
        NSLog("IOS26VideoEnhancer: interpolation failed: \(error)")
      case .success:
        // Optionally run Anime4K on the interpolated frame as well. Without
        // Anime4K the scratch buffer already holds the result, so it is
        // published as is (no extra blit).
        let output: CVPixelBuffer
        if let processor = self.processor, processor.hasAnime4K {
          output = processor.process(destination) ?? destination
        } else {
          output = destination
        }
        self.frames.push(
          EnhanceFrame(pixelBuffer: output, presentationTime: middleTime)
        )
        self.onFrameAvailable?()
      }
    }
  }

  func latestFrame() -> CVPixelBuffer? {
    return frames.latest()?.pixelBuffer
  }

  func invalidate() {
    isActive = false
    frames.clear()
    previousBuffer = nil
    previousTime = .invalid
    interpolator?.invalidate()
    interpolator = nil
    processor?.invalidate()
  }
}

// MARK: - Factory

enum VideoEnhancerFactory {
  /// Builds the best pipeline the current device can run for \`configuration\`.
  /// Returns nil when none of the requested stages can be honoured; callers
  /// must then keep the plain rendering path.
  static func make(
    configuration: VideoEnhanceConfiguration
  ) -> FrameEnhancerProtocol? {
    let wantsVideoToolbox =
      configuration.interpolation || configuration.vtScaleFactor > 0

    if wantsVideoToolbox {
      if #available(iOS 26.0, *) {
        if let enhancer = IOS26VideoEnhancer() {
          return enhancer
        }
      }
      NSLog(
        "VideoEnhancerFactory: VideoToolbox enhancement unavailable, falling back to Anime4K only"
      )
    }

    if !configuration.anime4kShaders.isEmpty {
      return MetalOnlyEnhancer()
    }

    return nil
  }
}
