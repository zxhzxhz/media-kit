//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Metal side of the enhancement pipeline.
//
//  It owns everything that is shared between the "Anime4K only" path and the
//  VideoToolbox path: a Metal device/queue pair, a CVMetalTextureCache that
//  wraps the BGRA CVPixelBuffers produced by TextureHW without a copy, an
//  output CVPixelBufferPool, and the Anime4KMetalRenderer instance.
//
//  When no Anime4K shaders are configured the processor degrades to a plain
//  GPU blit, which is what the VideoToolbox path uses to obtain scratch buffers
//  it is allowed to keep across frames.

import CoreVideo
import Foundation
import Metal

final class MetalFrameProcessor {
  /// Kicks off pipeline compilation. Returns nil when Metal is unavailable.
  init?() {
    guard let device = MTLCreateSystemDefaultDevice(),
      let commandQueue = device.makeCommandQueue()
    else {
      NSLog("MetalFrameProcessor: Metal is unavailable")
      return nil
    }
    self.device = device
    self.commandQueue = commandQueue

    var cache: CVMetalTextureCache?
    let cacheStatus = CVMetalTextureCacheCreate(
      kCFAllocatorDefault,
      nil,
      device,
      nil,
      &cache
    )
    guard cacheStatus == kCVReturnSuccess, let textureCache = cache else {
      NSLog("MetalFrameProcessor: CVMetalTextureCacheCreate failed: \(cacheStatus)")
      return nil
    }
    self.textureCache = textureCache

    guard let library = try? device.makeLibrary(
      source: MetalFrameProcessor.copyKernelSource,
      options: nil
    ),
      let function = library.makeFunction(name: "mkvideo_copy"),
      let pipeline = try? device.makeComputePipelineState(function: function)
    else {
      NSLog("MetalFrameProcessor: failed to compile the copy kernel")
      return nil
    }
    self.copyPipeline = pipeline

    let samplerDescriptor = MTLSamplerDescriptor()
    samplerDescriptor.magFilter = .linear
    samplerDescriptor.minFilter = .linear
    samplerDescriptor.sAddressMode = .clampToEdge
    samplerDescriptor.tAddressMode = .clampToEdge
    guard let sampler = device.makeSamplerState(descriptor: samplerDescriptor)
    else {
      NSLog("MetalFrameProcessor: failed to create the sampler")
      return nil
    }
    self.linearSampler = sampler
  }

  deinit {
    pool = nil
    renderer = nil
    CVMetalTextureCacheFlush(textureCache, 0)
  }

  // MARK: - Configuration

  /// Rebuilds the pipeline when the profile or the frame size changes.
  ///
  /// Compiling Anime4K takes hundreds of milliseconds, so this must never run
  /// on the platform thread. Callers are expected to invoke it from the
  /// TextureHW worker.
  func configure(
    profile: String,
    glslSources: [String],
    width: Int,
    height: Int
  ) -> Bool {
    guard width > 0, height > 0 else {
      return false
    }

    let key = MetalFrameProcessor.pipelineKey(
      profile: profile,
      glslSources: glslSources
    )

    if key == pipelineKey, width == self.width, height == self.height,
      pool != nil
    {
      return true
    }

    let sizeChanged = width != self.width || height != self.height

    self.width = width
    self.height = height
    pipelineKey = key

    if sizeChanged {
      CVMetalTextureCacheFlush(textureCache, 0)
    }

    // Recompile the Anime4K pipeline. An empty source list is legal and simply
    // disables the Anime4K stage, leaving the blit path in place.
    if glslSources.isEmpty {
      renderer = nil
    } else if width * height > MetalFrameProcessor.maxAnime4KPixels {
      // The Anime4K chain ends with an x2 upscale, so the intermediate
      // rgba16Float textures cover 4x the input area (roughly 66 MB per
      // texture at 1080p and 265 MB at 4K). Refuse to run it above 1080p
      // instead of risking a jetsam kill: 4K sources do not need upscaling
      // anyway and simply fall back to the plain path.
      NSLog(
        "MetalFrameProcessor: skipping Anime4K for \(width)x\(height) (above the 1080p budget)"
      )
      renderer = nil
    } else {
      do {
        let renderer = try Anime4KMetalRenderer(
          name: "kazumi-\(profile)",
          device: device,
          glslSources: glslSources
        )
        // OUTPUT is the display size Anime4K assumes. It is what the WHEN
        // expressions compare against, so the value decides the chain:
        //   OUTPUT = 1.0x -> AutoDownscalePre_x2 needs > 1.2: skipped.
        //                    Upscale_CNN_x2 needs OUTPUT/MAIN > 1.2: skipped.
        //                    Result: restore only, no upscale.
        //   OUTPUT = 1.5x -> AutoDownscalePre_x2 runs (1.2 < 1.5 < 2.0) and
        //                    sets MAIN to 1.5x; both upscales then need
        //                    OUTPUT/MAIN > 1.2 and are skipped. Bilinear 1.5x.
        //   OUTPUT = 2.0x -> AutoDownscalePre_x2 needs < 2.0 (false) and x4
        //                    needs > 2.4 (false), so both are skipped;
        //                    Upscale_CNN_x2_M sees OUTPUT/MAIN = 2.0 and runs,
        //                    producing exactly 2x, then _S is skipped again.
        // 2x is therefore the only setting that runs the CNN upscale once and
        // still caps the intermediate rgba16Float textures at 2x the input
        // area (about 66 MB each at 1080p).
        try renderer.compileShaders(
          device: device,
          textureInW: width,
          textureInH: height,
          outputW: width * 2,
          outputH: height * 2
        )
        self.renderer = renderer
        NSLog(
          "MetalFrameProcessor: Anime4K pipeline ready (\(profile), \(renderer.enabledShaders.count) passes)"
        )
      } catch {
        NSLog("MetalFrameProcessor: Anime4K compilation failed: \(error)")
        // Degrade to the blit path instead of failing: the VideoToolbox stage
        // can still work, and TextureHW reports "not active" only when the
        // caller asked for Anime4K alone (it checks hasAnime4K afterwards).
        renderer = nil
      }
    }

    if sizeChanged || pool == nil {
      guard let pool = MetalFrameProcessor.makePool(width: width, height: height)
      else {
        NSLog("MetalFrameProcessor: failed to create the output pool")
        return false
      }
      self.pool = pool
    }

    return true
  }

  var hasAnime4K: Bool {
    return renderer != nil
  }

  // MARK: - Processing

  /// Runs the configured stages on \`source\` and returns a buffer taken from the
  /// internal pool, or nil when the frame could not be processed.
  ///
  /// The call is synchronous: it waits for the GPU to finish so that the
  /// returned buffer can immediately be handed to Flutter.
  func process(_ source: CVPixelBuffer) -> CVPixelBuffer? {
    guard let pool = pool else {
      return nil
    }

    let sourceWidth = CVPixelBufferGetWidth(source)
    let sourceHeight = CVPixelBufferGetHeight(source)
    guard sourceWidth > 0, sourceHeight > 0 else {
      return nil
    }

    var output: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      pool,
      &output
    )
    guard status == kCVReturnSuccess, let outputBuffer = output else {
      // The pool is starved: drop this frame instead of stalling the renderer.
      return nil
    }

    guard
      let input = makeTexture(
        source,
        width: sourceWidth,
        height: sourceHeight
      )
    else {
      return nil
    }
    guard
      let destination = makeTexture(
        outputBuffer,
        width: width,
        height: height
      )
    else {
      return nil
    }

    guard let commandBuffer = commandQueue.makeCommandBuffer() else {
      return nil
    }

    // The CV metal textures must outlive the GPU work that reads them.
    let inputCVTexture = input.cvTexture
    let outputCVTexture = destination.cvTexture
    commandBuffer.addCompletedHandler { _ in
      withExtendedLifetime(inputCVTexture) {}
      withExtendedLifetime(outputCVTexture) {}
    }

    do {
      if let renderer = renderer {
        try renderer.encode(
          commandBuffer: commandBuffer,
          input: input.texture,
          output: destination.texture
        )
      } else {
        encodeCopy(
          commandBuffer: commandBuffer,
          input: input.texture,
          output: destination.texture
        )
      }
    } catch {
      NSLog("MetalFrameProcessor: encode failed: \(error)")
      commandBuffer.commit()
      return nil
    }

    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if commandBuffer.error != nil {
      NSLog("MetalFrameProcessor: command buffer failed: \(String(describing: commandBuffer.error))")
      return nil
    }

    return outputBuffer
  }

  /// Borrows a buffer from the internal pool without running any stage.
  /// Used by the VideoToolbox path for interpolation destinations.
  func makeScratchBuffer() -> CVPixelBuffer? {
    guard let pool = pool else {
      return nil
    }
    var buffer: CVPixelBuffer?
    let status = CVPixelBufferPoolCreatePixelBuffer(
      kCFAllocatorDefault,
      pool,
      &buffer
    )
    guard status == kCVReturnSuccess else {
      return nil
    }
    return buffer
  }

  /// Releases the output pool so that previously returned buffers can be freed.
  func invalidate() {
    pool = nil
    renderer = nil
    pipelineKey = ""
    width = 0
    height = 0
    CVMetalTextureCacheFlush(textureCache, 0)
  }

  // MARK: - Internals

  private func encodeCopy(
    commandBuffer: MTLCommandBuffer,
    input: MTLTexture,
    output: MTLTexture
  ) {
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
      return
    }
    encoder.setComputePipelineState(copyPipeline)
    encoder.setTexture(input, index: 0)
    encoder.setTexture(output, index: 1)
    encoder.setSamplerState(linearSampler, index: 0)
    let width = copyPipeline.threadExecutionWidth
    let height = max(1, copyPipeline.maxTotalThreadsPerThreadgroup / width)
    let threadsPerThreadgroup = MTLSizeMake(width, height, 1)
    let threadgroups = MTLSize(
      width: (output.width + width - 1) / width,
      height: (output.height + height - 1) / height,
      depth: 1
    )
    encoder.dispatchThreadgroups(
      threadgroups,
      threadsPerThreadgroup: threadsPerThreadgroup
    )
    encoder.endEncoding()
  }

  private func makeTexture(
    _ buffer: CVPixelBuffer,
    width: Int,
    height: Int
  ) -> (texture: MTLTexture, cvTexture: CVMetalTexture)? {
    var cvTexture: CVMetalTexture?
    let status = CVMetalTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault,
      textureCache,
      buffer,
      nil,
      .bgra8Unorm,
      width,
      height,
      0,
      &cvTexture
    )
    guard status == kCVReturnSuccess, let cvTexture = cvTexture,
      let texture = CVMetalTextureGetTexture(cvTexture)
    else {
      NSLog("MetalFrameProcessor: failed to wrap a CVPixelBuffer (\(status))")
      return nil
    }
    return (texture, cvTexture)
  }

  private static func makePool(width: Int, height: Int) -> CVPixelBufferPool? {
    let attributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
      kCVPixelBufferMetalCompatibilityKey as String: true,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
      kCVPixelBufferPoolMinimumBufferCountKey as String: 4,
    ]
    var pool: CVPixelBufferPool?
    let status = CVPixelBufferPoolCreate(
      kCFAllocatorDefault,
      nil,
      attributes as CFDictionary,
      &pool
    )
    guard status == kCVReturnSuccess else {
      NSLog("MetalFrameProcessor: CVPixelBufferPoolCreate failed: \(status)")
      return nil
    }
    return pool
  }

  private static func pipelineKey(
    profile: String,
    glslSources: [String]
  ) -> String {
    // Cheap, allocation free identity for the shader set. Hashing the whole
    // GLSL text on every resize would be wasteful and hashValue is only stable
    // inside one process, which is exactly the scope we need.
    var total = 0
    for source in glslSources {
      total = total &+ source.utf8.count
    }
    return "\(profile):\(glslSources.count):\(total)"
  }

  /// Upper bound on the decoded frame size for the Anime4K stage (1080p).
  private static let maxAnime4KPixels = 1920 * 1080

  private static let copyKernelSource = """
    #include <metal_stdlib>
    using namespace metal;

    kernel void mkvideo_copy(texture2d<float, access::sample> src [[texture(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             sampler s [[sampler(0)]],
                             uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) {
            return;
        }
        float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
        dst.write(src.sample(s, uv), gid);
    }
    """
}
