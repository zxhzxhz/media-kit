//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  CVPixelBuffer format conversion for the VideoToolbox stages.
//
//  VTFrameProcessor only accepts the pixel formats it advertises in
//  `frameSupportedPixelFormats` — on an iPad Pro M4 that is `420v`
//  (kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, NV12 video range) — while
//  TextureHW renders into `32BGRA` and Flutter displays `32BGRA` identically
//  without running Impeller's hardcoded-BT.601 YuvToRgbFilter.
//
//  CoreImage (`CIContext` with `CGColorSpace.itur_709`) applies Apple's ~1.96
//  scene gamma curve on every 32BGRA <-> 420v conversion, visibly brightening
//  the image compared to plain playback. This class replaces CoreImage with
//  direct Metal compute kernels using exact ITU-R BT.709 limited-range math and
//  zero gamma re-encoding.
//
//  Buffers handed to VT must still be created from the configuration's own
//  `sourcePixelBufferAttributes` / `destinationPixelBufferAttributes`: the ANE
//  backend has hard alignment / extended-pixel / cache-mode requirements.

import CoreGraphics
import CoreVideo
import Foundation
import Metal
import VideoToolbox

@available(iOS 16.0, *)
private struct MetalPatchParams {
  var srcInnerOrigin: SIMD2<UInt32>
  var dstOrigin: SIMD2<UInt32>
  var copySize: SIMD2<UInt32>
}

@available(iOS 16.0, *)
private final class MetalConversionPipelines {
  let device: MTLDevice
  let commandQueue: MTLCommandQueue
  let bgraToNv12Y: MTLComputePipelineState
  let bgraToNv12UV: MTLComputePipelineState
  let nv12CropY: MTLComputePipelineState
  let nv12CropUV: MTLComputePipelineState
  let nv12ToBgra: MTLComputePipelineState
  let bgraPatch: MTLComputePipelineState

  private static let cacheLock = NSLock()
  private static weak var cachedDevice: AnyObject?
  private static var cachedInstance: MetalConversionPipelines?

  static func shared() -> MetalConversionPipelines? {
    guard let device = MTLCreateSystemDefaultDevice() else {
      return nil
    }
    cacheLock.lock()
    defer { cacheLock.unlock() }
    if cachedDevice === (device as AnyObject), let existing = cachedInstance {
      return existing
    }
    guard let created = MetalConversionPipelines(device: device) else {
      return nil
    }
    cachedDevice = device as AnyObject
    cachedInstance = created
    return created
  }

  private init?(device: MTLDevice) {
    guard let commandQueue = device.makeCommandQueue() else {
      NSLog("PixelBufferConverter: Metal command queue unavailable")
      return nil
    }
    let library: MTLLibrary
    do {
      library = try device.makeLibrary(
        source: MetalConversionPipelines.shaderSource,
        options: nil
      )
    } catch {
      NSLog("PixelBufferConverter: failed to compile Metal kernels: \(error)")
      return nil
    }

    func makePipeline(_ name: String) -> MTLComputePipelineState? {
      guard let fn = library.makeFunction(name: name) else {
        NSLog("PixelBufferConverter: missing Metal kernel \(name)")
        return nil
      }
      do {
        return try device.makeComputePipelineState(function: fn)
      } catch {
        NSLog("PixelBufferConverter: pipeline \(name) failed: \(error)")
        return nil
      }
    }

    guard let bgraToNv12Y = makePipeline("mkconv_bgra_to_nv12_y"),
      let bgraToNv12UV = makePipeline("mkconv_bgra_to_nv12_uv"),
      let nv12CropY = makePipeline("mkconv_nv12_crop_y"),
      let nv12CropUV = makePipeline("mkconv_nv12_crop_uv"),
      let nv12ToBgra = makePipeline("mkconv_nv12_to_bgra"),
      let bgraPatch = makePipeline("mkconv_bgra_patch")
    else {
      return nil
    }

    self.device = device
    self.commandQueue = commandQueue
    self.bgraToNv12Y = bgraToNv12Y
    self.bgraToNv12UV = bgraToNv12UV
    self.nv12CropY = nv12CropY
    self.nv12CropUV = nv12CropUV
    self.nv12ToBgra = nv12ToBgra
    self.bgraPatch = bgraPatch
  }

  // Exact ITU-R BT.709 limited-range matrices with zero gamma / transfer shift:
  // `bgra8Unorm` exposes normalized [0, 1] values directly and `r8Unorm` /
  // `rg8Unorm` carry Y in [16/255, 235/255] and Cb/Cr in [16/255, 240/255].
  private static let shaderSource = """
    #include <metal_stdlib>
    using namespace metal;

    struct PatchParams {
      uint2 srcInnerOrigin;
      uint2 dstOrigin;
      uint2 copySize;
    };

    kernel void mkconv_bgra_to_nv12_y(
      texture2d<float, access::read> src [[texture(0)]],
      texture2d<float, access::write> dstY [[texture(1)]],
      constant uint2 &cropOrigin [[buffer(0)]],
      uint2 gid [[thread_position_in_grid]]
    ) {
      if (gid.x >= dstY.get_width() || gid.y >= dstY.get_height()) {
        return;
      }
      uint2 maxSrc = uint2(src.get_width() - 1, src.get_height() - 1);
      uint2 p = min(cropOrigin + gid, maxSrc);
      float3 rgb = src.read(p).rgb;
      float y = (16.0 + 219.0 * (0.2126 * rgb.r + 0.7152 * rgb.g + 0.0722 * rgb.b)) / 255.0;
      dstY.write(float4(clamp(y, 0.0, 1.0), 0.0, 0.0, 1.0), gid);
    }

    kernel void mkconv_bgra_to_nv12_uv(
      texture2d<float, access::read> src [[texture(0)]],
      texture2d<float, access::write> dstUV [[texture(1)]],
      constant uint2 &cropOrigin [[buffer(0)]],
      uint2 gid [[thread_position_in_grid]]
    ) {
      if (gid.x >= dstUV.get_width() || gid.y >= dstUV.get_height()) {
        return;
      }
      uint2 maxSrc = uint2(src.get_width() - 1, src.get_height() - 1);
      uint2 base = cropOrigin + gid * 2;
      float3 rgb = 0.25 * (
        src.read(min(base, maxSrc)).rgb +
        src.read(min(base + uint2(1, 0), maxSrc)).rgb +
        src.read(min(base + uint2(0, 1), maxSrc)).rgb +
        src.read(min(base + uint2(1, 1), maxSrc)).rgb
      );
      float cb = (128.0 + 224.0 * (-0.114572 * rgb.r - 0.385428 * rgb.g + 0.500000 * rgb.b)) / 255.0;
      float cr = (128.0 + 224.0 * ( 0.500000 * rgb.r - 0.454153 * rgb.g - 0.045847 * rgb.b)) / 255.0;
      dstUV.write(float4(clamp(cb, 0.0, 1.0), clamp(cr, 0.0, 1.0), 0.0, 1.0), gid);
    }

    kernel void mkconv_nv12_crop_y(
      texture2d<float, access::read> srcY [[texture(0)]],
      texture2d<float, access::write> dstY [[texture(1)]],
      constant uint2 &cropOrigin [[buffer(0)]],
      uint2 gid [[thread_position_in_grid]]
    ) {
      if (gid.x >= dstY.get_width() || gid.y >= dstY.get_height()) {
        return;
      }
      uint2 maxSrc = uint2(srcY.get_width() - 1, srcY.get_height() - 1);
      float y = srcY.read(min(cropOrigin + gid, maxSrc)).r;
      dstY.write(float4(y, 0.0, 0.0, 1.0), gid);
    }

    kernel void mkconv_nv12_crop_uv(
      texture2d<float, access::read> srcUV [[texture(0)]],
      texture2d<float, access::write> dstUV [[texture(1)]],
      constant uint2 &cropOrigin [[buffer(0)]],
      uint2 gid [[thread_position_in_grid]]
    ) {
      if (gid.x >= dstUV.get_width() || gid.y >= dstUV.get_height()) {
        return;
      }
      uint2 maxSrc = uint2(srcUV.get_width() - 1, srcUV.get_height() - 1);
      float2 cbcr = srcUV.read(min((cropOrigin >> 1) + gid, maxSrc)).rg;
      dstUV.write(float4(cbcr.x, cbcr.y, 0.0, 1.0), gid);
    }

    kernel void mkconv_nv12_to_bgra(
      texture2d<float, access::read> srcY [[texture(0)]],
      texture2d<float, access::sample> srcUV [[texture(1)]],
      texture2d<float, access::write> dst [[texture(2)]],
      constant PatchParams &p [[buffer(0)]],
      uint2 gid [[thread_position_in_grid]]
    ) {
      if (gid.x >= p.copySize.x || gid.y >= p.copySize.y) {
        return;
      }
      uint2 dstPos = p.dstOrigin + gid;
      if (dstPos.x >= dst.get_width() || dstPos.y >= dst.get_height()) {
        return;
      }
      uint2 maxSrc = uint2(srcY.get_width() - 1, srcY.get_height() - 1);
      uint2 srcPos = min(p.srcInnerOrigin + gid, maxSrc);
      float y = srcY.read(srcPos).r;
      constexpr sampler uvSampler(coord::normalized, address::clamp_to_edge, filter::linear);
      float2 uvCoord = ((float2(srcPos) + 0.5) * 0.5) / float2(srcUV.get_width(), srcUV.get_height());
      float2 cbcr = srcUV.sample(uvSampler, uvCoord).rg;

      float y_  = (y * 255.0 - 16.0) / 219.0;
      float cb_ = (cbcr.x * 255.0 - 128.0) / 224.0;
      float cr_ = (cbcr.y * 255.0 - 128.0) / 224.0;
      float r   = clamp(y_ + 1.574800 * cr_, 0.0, 1.0);
      float g   = clamp(y_ - 0.187324 * cb_ - 0.468124 * cr_, 0.0, 1.0);
      float b   = clamp(y_ + 1.855600 * cb_, 0.0, 1.0);
      dst.write(float4(r, g, b, 1.0), dstPos);
    }

    kernel void mkconv_bgra_patch(
      texture2d<float, access::read> src [[texture(0)]],
      texture2d<float, access::write> dst [[texture(1)]],
      constant PatchParams &p [[buffer(0)]],
      uint2 gid [[thread_position_in_grid]]
    ) {
      if (gid.x >= p.copySize.x || gid.y >= p.copySize.y) {
        return;
      }
      uint2 dstPos = p.dstOrigin + gid;
      if (dstPos.x >= dst.get_width() || dstPos.y >= dst.get_height()) {
        return;
      }
      uint2 maxSrc = uint2(src.get_width() - 1, src.get_height() - 1);
      uint2 srcPos = min(p.srcInnerOrigin + gid, maxSrc);
      dst.write(src.read(srcPos), dstPos);
    }
    """
}

/// Converts CVPixelBuffers between two pixel formats at a fixed size.
@available(iOS 16.0, *)
final class PixelBufferConverter {
  private let pipelines: MetalConversionPipelines
  private let textureCache: CVMetalTextureCache
  private let cacheLock = NSLock()
  private var flushCountdown: Int = 120
  private let pool: CVPixelBufferPool?
  private let vtTransferSession: VTPixelTransferSession?
  private let width: Int
  private let height: Int
  /// Format this converter produces.
  let pixelFormat: OSType

  private static let readUsage: CFDictionary =
    [
      kCVMetalTextureUsage as String: NSNumber(
        value: MTLTextureUsage.shaderRead.rawValue
      )
    ] as CFDictionary

  private static let writeUsage: CFDictionary =
    [
      kCVMetalTextureUsage as String: NSNumber(
        value: MTLTextureUsage.shaderWrite.rawValue
      )
    ] as CFDictionary

  /// - Parameters:
  ///   - providesPool: create a `CVPixelBufferPool` of the target format.
  ///   - vtAttributes: the VT configuration's `sourcePixelBufferAttributes` (or
  ///     `destinationPixelBufferAttributes`), merged into the native pool
  ///     attributes so the buffers satisfy the ANE backend's constraints.
  init?(
    width: Int,
    height: Int,
    pixelFormat: OSType,
    providesPool: Bool,
    vtAttributes: [String: Any]? = nil
  ) {
    guard width > 0, height > 0,
      let pipelines = MetalConversionPipelines.shared()
    else {
      NSLog("PixelBufferConverter: Metal pipelines unavailable")
      return nil
    }

    var cache: CVMetalTextureCache?
    let cacheStatus = CVMetalTextureCacheCreate(
      kCFAllocatorDefault,
      nil,
      pipelines.device,
      nil,
      &cache
    )
    guard cacheStatus == kCVReturnSuccess, let createdCache = cache else {
      NSLog("PixelBufferConverter: CVMetalTextureCacheCreate failed: \(cacheStatus)")
      return nil
    }

    self.pipelines = pipelines
    self.textureCache = createdCache
    self.width = width
    self.height = height
    self.pixelFormat = pixelFormat

    if providesPool {
      guard
        let created = PixelBufferConverter.makeResolvedPool(
          vtAttributes: vtAttributes,
          width: width,
          height: height,
          pixelFormat: pixelFormat
        )
      else {
        NSLog("PixelBufferConverter: pool creation failed")
        return nil
      }
      self.pool = created
    } else {
      self.pool = nil
    }

    var session: VTPixelTransferSession?
    if VTPixelTransferSessionCreate(
      allocator: kCFAllocatorDefault,
      pixelTransferSessionOut: &session
    ) == noErr {
      self.vtTransferSession = session
    } else {
      self.vtTransferSession = nil
    }
  }

  deinit {
    if let session = vtTransferSession {
      VTPixelTransferSessionInvalidate(session)
    }
    CVMetalTextureCacheFlush(textureCache, 0)
  }

  /// Builds a `CVPixelBufferPool` that satisfies both VideoToolbox's required
  /// buffer attributes and ours.
  ///
  /// `kCVPixelBufferPoolMinimumBufferCountKey` belongs to the pool attributes
  /// (second parameter of `CVPixelBufferPoolCreate`), not to the pixel buffer
  /// attributes — putting it in the latter silently does nothing.
  static func makeResolvedPool(
    vtAttributes: [String: Any]?,
    width: Int,
    height: Int,
    pixelFormat: OSType,
    minBufferCount: Int = 6
  ) -> CVPixelBufferPool? {
    let customAttributes: [String: Any] = [
      kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
      kCVPixelBufferWidthKey as String: width,
      kCVPixelBufferHeightKey as String: height,
      kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
      kCVPixelBufferMetalCompatibilityKey as String: true,
    ]

    var resolved: CFDictionary?
    var sources: [CFDictionary] = [customAttributes as CFDictionary]
    if let vtAttributes = vtAttributes, !vtAttributes.isEmpty {
      sources.append(vtAttributes as CFDictionary)
    }
    let status = CVPixelBufferCreateResolvedAttributesDictionary(
      kCFAllocatorDefault,
      sources as CFArray,
      &resolved
    )
    guard status == kCVReturnSuccess, let resolvedAttributes = resolved else {
      NSLog("PixelBufferConverter: resolve attributes failed: \(status)")
      return nil
    }

    let poolAttributes: [String: Any] = [
      kCVPixelBufferPoolMinimumBufferCountKey as String: minBufferCount
    ]

    var pool: CVPixelBufferPool?
    let poolStatus = CVPixelBufferPoolCreate(
      kCFAllocatorDefault,
      poolAttributes as CFDictionary,
      resolvedAttributes as CFDictionary,
      &pool
    )
    guard poolStatus == kCVReturnSuccess, let created = pool else {
      NSLog("PixelBufferConverter: pool create failed: \(poolStatus)")
      return nil
    }
    return created
  }

  /// Returns a pooled buffer in the target format, or `nil` on failure.
  func convert(_ source: CVPixelBuffer) -> CVPixelBuffer? {
    guard let pool = pool else {
      return nil
    }
    var destination: CVPixelBuffer?
    guard
      CVPixelBufferPoolCreatePixelBuffer(
        kCFAllocatorDefault,
        pool,
        &destination
      ) == kCVReturnSuccess,
      let output = destination
    else {
      return nil
    }
    return convert(source, to: output) ? output : nil
  }

  /// Converts `source` into a caller supplied buffer of the destination's format.
  ///
  /// Handles `32BGRA <-> 420v`, `420v -> 420v`, and `32BGRA -> 32BGRA` with exact
  /// ITU-R BT.709 limited-range math and no gamma shift.
  @discardableResult
  func convert(_ source: CVPixelBuffer, to destination: CVPixelBuffer) -> Bool {
    PixelBufferConverter.applyColorAttachments(to: destination)
    return runMetalTransfer(
      source: source,
      destination: destination,
      cropOrigin: .zero
    )
  }

  /// Crops `cropRect` (top-left coordinates) from `source`, converts it to the
  /// converter's target format, and writes it into a pooled `width x height` buffer.
  func convertCropped(_ source: CVPixelBuffer, cropRect: CGRect) -> CVPixelBuffer? {
    guard let pool = pool else {
      return nil
    }
    var destination: CVPixelBuffer?
    guard
      CVPixelBufferPoolCreatePixelBuffer(
        kCFAllocatorDefault,
        pool,
        &destination
      ) == kCVReturnSuccess,
      let output = destination
    else {
      return nil
    }
    PixelBufferConverter.applyColorAttachments(to: output)
    let origin = SIMD2<UInt32>(
      UInt32(max(0, Int(cropRect.origin.x.rounded()))),
      UInt32(max(0, Int(cropRect.origin.y.rounded())))
    )
    return runMetalTransfer(
      source: source,
      destination: output,
      cropOrigin: origin
    ) ? output : nil
  }

  /// Stitches non-overlapping inner patches from upscaled tiles directly into
  /// `destination` (`32BGRA` or `420v`-to-`32BGRA`) in a single Metal pass.
  @discardableResult
  func compositeTiles(
    _ tiles: [(buffer: CVPixelBuffer, innerRect: CGRect, destOrigin: CGPoint)],
    to destination: CVPixelBuffer
  ) -> Bool {
    guard !tiles.isEmpty else {
      return false
    }
    let dstFormat = CVPixelBufferGetPixelFormatType(destination)
    guard dstFormat == kCVPixelFormatType_32BGRA else {
      return false
    }
    let dstW = CVPixelBufferGetWidth(destination)
    let dstH = CVPixelBufferGetHeight(destination)
    guard dstW > 0, dstH > 0,
      let dstTex = makeTexture(
        destination,
        pixelFormat: .bgra8Unorm,
        width: dstW,
        height: dstH,
        planeIndex: 0,
        writable: true
      ),
      let commandBuffer = pipelines.commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
      return false
    }

    var retained: [CVMetalTexture] = [dstTex.cvTexture]
    retained.reserveCapacity(1 + tiles.count * 2)

    for tile in tiles {
      let copyW = max(0, Int(tile.innerRect.width.rounded()))
      let copyH = max(0, Int(tile.innerRect.height.rounded()))
      guard copyW > 0, copyH > 0 else {
        continue
      }
      var params = MetalPatchParams(
        srcInnerOrigin: SIMD2<UInt32>(
          UInt32(max(0, Int(tile.innerRect.origin.x.rounded()))),
          UInt32(max(0, Int(tile.innerRect.origin.y.rounded())))
        ),
        dstOrigin: SIMD2<UInt32>(
          UInt32(max(0, Int(tile.destOrigin.x.rounded()))),
          UInt32(max(0, Int(tile.destOrigin.y.rounded())))
        ),
        copySize: SIMD2<UInt32>(UInt32(copyW), UInt32(copyH))
      )

      if PixelBufferConverter.isYUVFormat(tile.buffer) {
        let srcY = CVPixelBufferGetWidthOfPlane(tile.buffer, 0)
        let srcYH = CVPixelBufferGetHeightOfPlane(tile.buffer, 0)
        let srcUV = CVPixelBufferGetWidthOfPlane(tile.buffer, 1)
        let srcUVH = CVPixelBufferGetHeightOfPlane(tile.buffer, 1)
        guard srcY > 0, srcYH > 0, srcUV > 0, srcUVH > 0,
          let yTex = makeTexture(
            tile.buffer,
            pixelFormat: .r8Unorm,
            width: srcY,
            height: srcYH,
            planeIndex: 0,
            writable: false
          ),
          let uvTex = makeTexture(
            tile.buffer,
            pixelFormat: .rg8Unorm,
            width: srcUV,
            height: srcUVH,
            planeIndex: 1,
            writable: false
          )
        else {
          encoder.endEncoding()
          return false
        }
        retained.append(yTex.cvTexture)
        retained.append(uvTex.cvTexture)
        encoder.setTexture(yTex.texture, index: 0)
        encoder.setTexture(uvTex.texture, index: 1)
        encoder.setTexture(dstTex.texture, index: 2)
        encoder.setBytes(
          &params,
          length: MemoryLayout<MetalPatchParams>.stride,
          index: 0
        )
        dispatch(
          encoder: encoder,
          pipeline: pipelines.nv12ToBgra,
          width: copyW,
          height: copyH
        )
      } else if CVPixelBufferGetPixelFormatType(tile.buffer)
        == kCVPixelFormatType_32BGRA
      {
        let srcW = CVPixelBufferGetWidth(tile.buffer)
        let srcH = CVPixelBufferGetHeight(tile.buffer)
        guard srcW > 0, srcH > 0,
          let srcTex = makeTexture(
            tile.buffer,
            pixelFormat: .bgra8Unorm,
            width: srcW,
            height: srcH,
            planeIndex: 0,
            writable: false
          )
        else {
          encoder.endEncoding()
          return false
        }
        retained.append(srcTex.cvTexture)
        encoder.setTexture(srcTex.texture, index: 0)
        encoder.setTexture(dstTex.texture, index: 1)
        encoder.setBytes(
          &params,
          length: MemoryLayout<MetalPatchParams>.stride,
          index: 0
        )
        dispatch(
          encoder: encoder,
          pipeline: pipelines.bgraPatch,
          width: copyW,
          height: copyH
        )
      } else {
        encoder.endEncoding()
        return false
      }
    }

    encoder.endEncoding()
    let success = commitAndWait(commandBuffer, retaining: retained)
    if success {
      PixelBufferConverter.applyColorAttachments(to: destination)
    }
    return success
  }

  /// Converts `source` using `VTPixelTransferSession` into a pooled buffer of
  /// the target format, or returns `nil` if unavailable or transfer fails.
  func convertViaVT(_ source: CVPixelBuffer) -> CVPixelBuffer? {
    guard let pool = pool, let session = vtTransferSession else {
      return nil
    }
    var destination: CVPixelBuffer?
    guard
      CVPixelBufferPoolCreatePixelBuffer(
        kCFAllocatorDefault,
        pool,
        &destination
      ) == kCVReturnSuccess,
      let output = destination
    else {
      return nil
    }
    PixelBufferConverter.applyColorAttachments(to: output)
    let status = VTPixelTransferSessionTransferImage(
      session,
      from: source,
      to: output
    )
    guard status == noErr else {
      return nil
    }
    return output
  }

  /// Formats a list of FourCC pixel format codes as a comma-separated string.
  static func formatListDescription(_ formats: [OSType]) -> String {
    guard !formats.isEmpty else {
      return "none"
    }
    return formats.map(fourCC).joined(separator: ",")
  }

  /// Extracts a readable pixel format summary from a CVPixelBuffer attributes dictionary.
  static func pixelFormatSummary(from attributes: [String: Any]) -> String {
    let key = kCVPixelBufferPixelFormatTypeKey as String
    guard let value = attributes[key] else {
      return "none"
    }
    if let number = value as? NSNumber {
      return fourCC(number.uint32Value)
    }
    if let numbers = value as? [NSNumber] {
      return formatListDescription(numbers.map { $0.uint32Value })
    }
    return String(describing: value)
  }

  /// Empty pool buffer of the target format.
  func makeBuffer() -> CVPixelBuffer? {
    guard let pool = pool else {
      return nil
    }
    var buffer: CVPixelBuffer?
    guard
      CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        == kCVReturnSuccess,
      let output = buffer
    else {
      return nil
    }
    PixelBufferConverter.applyColorAttachments(to: output)
    return output
  }

  /// `true` when a 420v buffer carries no picture data.
  ///
  /// An unwritten buffer is all zeros; a legitimate black frame keeps luma at
  /// 16. Sampling the Y plane is the only reliable check: converting an empty
  /// 420v to BGRA yields *dark green* with alpha 255 (YCbCr 0,0,0 is not RGB
  /// black), which looks like real content to any BGRA-side test.
  static func isBlankYUV(_ buffer: CVPixelBuffer) -> Bool {
    guard isYUVFormat(buffer),
      CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess
    else {
      return false
    }
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }
    guard let plane = CVPixelBufferGetBaseAddressOfPlane(buffer, 0) else {
      return false
    }
    let width = CVPixelBufferGetWidthOfPlane(buffer, 0)
    let height = CVPixelBufferGetHeightOfPlane(buffer, 0)
    let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
    guard width > 0, height > 0, bytesPerRow > 0 else {
      return false
    }
    let bytes = plane.assumingMemoryBound(to: UInt8.self)
    for row in 0..<3 {
      for col in 0..<3 {
        let x = width * (col * 2 + 1) / 6
        let y = height * (row * 2 + 1) / 6
        // 20 sits above the 16 of a legal black frame.
        if bytes[y * bytesPerRow + x] > 20 {
          return false
        }
      }
    }
    return true
  }

  /// `true` when a 32BGRA buffer has all-zero B/G/R channels at 9 sample points.
  static func isBlankBGRA(_ buffer: CVPixelBuffer) -> Bool {
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
    guard width > 0, height > 0, bytesPerRow >= width * 4 else {
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

  /// Format-aware blankness check across 420v/420f and 32BGRA buffers.
  static func isBlank(_ buffer: CVPixelBuffer) -> Bool {
    let format = CVPixelBufferGetPixelFormatType(buffer)
    switch format {
    case kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange,
      kCVPixelFormatType_420YpCbCr8BiPlanarFullRange:
      return isBlankYUV(buffer)
    case kCVPixelFormatType_32BGRA:
      return isBlankBGRA(buffer)
    default:
      return false
    }
  }

  /// Compact FourCC string for diagnostics.
  static func fourCC(_ code: OSType) -> String {
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

  /// Samples the center pixel for failure diagnostics (never called on the happy path).
  static func sampleCenter(_ buffer: CVPixelBuffer) -> String {
    let format = CVPixelBufferGetPixelFormatType(buffer)
    let tag = fourCC(format)
    guard CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess
    else {
      return "\(tag) lockFailed"
    }
    defer {
      CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
    }

    let width = CVPixelBufferGetWidth(buffer)
    let height = CVPixelBufferGetHeight(buffer)
    guard width > 0, height > 0 else {
      return "\(tag) empty"
    }

    if isYUVFormat(buffer) {
      guard let yPlane = CVPixelBufferGetBaseAddressOfPlane(buffer, 0),
        let uvPlane = CVPixelBufferGetBaseAddressOfPlane(buffer, 1)
      else {
        return "\(tag) noPlanes"
      }
      let yW = CVPixelBufferGetWidthOfPlane(buffer, 0)
      let yH = CVPixelBufferGetHeightOfPlane(buffer, 0)
      let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0)
      let uvW = CVPixelBufferGetWidthOfPlane(buffer, 1)
      let uvH = CVPixelBufferGetHeightOfPlane(buffer, 1)
      let uvStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
      guard yW > 0, yH > 0, yStride >= yW,
        uvW > 0, uvH > 0, uvStride >= uvW * 2
      else {
        return "\(tag) badPlanes"
      }
      let cx = yW / 2
      let cy = yH / 2
      let yVal = yPlane.assumingMemoryBound(to: UInt8.self)[cy * yStride + cx]
      let uvx = min(cx / 2, uvW - 1)
      let uvy = min(cy / 2, uvH - 1)
      let uvBytes = uvPlane.assumingMemoryBound(to: UInt8.self)
      let uvOff = uvy * uvStride + uvx * 2
      let cb = uvBytes[uvOff]
      let cr = uvBytes[uvOff + 1]
      return "\(tag) Y=\(yVal) Cb=\(cb) Cr=\(cr)"
    }

    if format == kCVPixelFormatType_32BGRA,
      let base = CVPixelBufferGetBaseAddress(buffer)
    {
      let stride = CVPixelBufferGetBytesPerRow(buffer)
      guard stride >= width * 4 else {
        return "\(tag) badStride"
      }
      let off = (height / 2) * stride + (width / 2) * 4
      let bytes = base.assumingMemoryBound(to: UInt8.self)
      return
        "\(tag)=\(bytes[off]),\(bytes[off + 1]),\(bytes[off + 2]),\(bytes[off + 3])"
    }

    return tag
  }

  /// Tags YUV buffers BT.709 for VideoToolbox.
  ///
  /// `32BGRA` buffers are intentionally left untagged so Flutter renders them
  /// identically to plain `TextureHW` without transfer-curve brightening.
  static func applyColorAttachments(to buffer: CVPixelBuffer) {
    guard isYUVFormat(buffer) else {
      return
    }
    CVBufferSetAttachment(
      buffer,
      kCVImageBufferColorPrimariesKey,
      kCVImageBufferColorPrimaries_ITU_R_709_2,
      .shouldPropagate
    )
    CVBufferSetAttachment(
      buffer,
      kCVImageBufferTransferFunctionKey,
      kCVImageBufferTransferFunction_ITU_R_709_2,
      .shouldPropagate
    )
    CVBufferSetAttachment(
      buffer,
      kCVImageBufferYCbCrMatrixKey,
      kCVImageBufferYCbCrMatrix_ITU_R_709_2,
      .shouldPropagate
    )
  }

  // MARK: - Metal Internals

  private static func isYUVFormat(_ buffer: CVPixelBuffer) -> Bool {
    let format = CVPixelBufferGetPixelFormatType(buffer)
    return format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
      || format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange
  }

  private func runMetalTransfer(
    source: CVPixelBuffer,
    destination: CVPixelBuffer,
    cropOrigin: SIMD2<UInt32>
  ) -> Bool {
    let srcIsYUV = PixelBufferConverter.isYUVFormat(source)
    let dstIsYUV = PixelBufferConverter.isYUVFormat(destination)
    let srcIsBGRA =
      CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_32BGRA
    let dstIsBGRA =
      CVPixelBufferGetPixelFormatType(destination) == kCVPixelFormatType_32BGRA

    guard (srcIsYUV || srcIsBGRA), (dstIsYUV || dstIsBGRA),
      let commandBuffer = pipelines.commandQueue.makeCommandBuffer(),
      let encoder = commandBuffer.makeComputeCommandEncoder()
    else {
      return false
    }

    var retained: [CVMetalTexture] = []

    if dstIsYUV {
      let dstYW = CVPixelBufferGetWidthOfPlane(destination, 0)
      let dstYH = CVPixelBufferGetHeightOfPlane(destination, 0)
      let dstUVW = CVPixelBufferGetWidthOfPlane(destination, 1)
      let dstUVH = CVPixelBufferGetHeightOfPlane(destination, 1)
      guard dstYW > 0, dstYH > 0, dstUVW > 0, dstUVH > 0,
        let dstYTex = makeTexture(
          destination,
          pixelFormat: .r8Unorm,
          width: dstYW,
          height: dstYH,
          planeIndex: 0,
          writable: true
        ),
        let dstUVTex = makeTexture(
          destination,
          pixelFormat: .rg8Unorm,
          width: dstUVW,
          height: dstUVH,
          planeIndex: 1,
          writable: true
        )
      else {
        encoder.endEncoding()
        return false
      }
      retained.append(dstYTex.cvTexture)
      retained.append(dstUVTex.cvTexture)

      var origin = cropOrigin
      if srcIsBGRA {
        let srcW = CVPixelBufferGetWidth(source)
        let srcH = CVPixelBufferGetHeight(source)
        guard srcW > 0, srcH > 0,
          let srcTex = makeTexture(
            source,
            pixelFormat: .bgra8Unorm,
            width: srcW,
            height: srcH,
            planeIndex: 0,
            writable: false
          )
        else {
          encoder.endEncoding()
          return false
        }
        retained.append(srcTex.cvTexture)
        encoder.setTexture(srcTex.texture, index: 0)
        encoder.setTexture(dstYTex.texture, index: 1)
        encoder.setBytes(
          &origin,
          length: MemoryLayout<SIMD2<UInt32>>.stride,
          index: 0
        )
        dispatch(
          encoder: encoder,
          pipeline: pipelines.bgraToNv12Y,
          width: dstYW,
          height: dstYH
        )

        encoder.setTexture(srcTex.texture, index: 0)
        encoder.setTexture(dstUVTex.texture, index: 1)
        encoder.setBytes(
          &origin,
          length: MemoryLayout<SIMD2<UInt32>>.stride,
          index: 0
        )
        dispatch(
          encoder: encoder,
          pipeline: pipelines.bgraToNv12UV,
          width: dstUVW,
          height: dstUVH
        )
      } else {
        let srcYW = CVPixelBufferGetWidthOfPlane(source, 0)
        let srcYH = CVPixelBufferGetHeightOfPlane(source, 0)
        let srcUVW = CVPixelBufferGetWidthOfPlane(source, 1)
        let srcUVH = CVPixelBufferGetHeightOfPlane(source, 1)
        guard srcYW > 0, srcYH > 0, srcUVW > 0, srcUVH > 0,
          let srcYTex = makeTexture(
            source,
            pixelFormat: .r8Unorm,
            width: srcYW,
            height: srcYH,
            planeIndex: 0,
            writable: false
          ),
          let srcUVTex = makeTexture(
            source,
            pixelFormat: .rg8Unorm,
            width: srcUVW,
            height: srcUVH,
            planeIndex: 1,
            writable: false
          )
        else {
          encoder.endEncoding()
          return false
        }
        retained.append(srcYTex.cvTexture)
        retained.append(srcUVTex.cvTexture)

        encoder.setTexture(srcYTex.texture, index: 0)
        encoder.setTexture(dstYTex.texture, index: 1)
        encoder.setBytes(
          &origin,
          length: MemoryLayout<SIMD2<UInt32>>.stride,
          index: 0
        )
        dispatch(
          encoder: encoder,
          pipeline: pipelines.nv12CropY,
          width: dstYW,
          height: dstYH
        )

        encoder.setTexture(srcUVTex.texture, index: 0)
        encoder.setTexture(dstUVTex.texture, index: 1)
        encoder.setBytes(
          &origin,
          length: MemoryLayout<SIMD2<UInt32>>.stride,
          index: 0
        )
        dispatch(
          encoder: encoder,
          pipeline: pipelines.nv12CropUV,
          width: dstUVW,
          height: dstUVH
        )
      }
    } else {
      let dstW = CVPixelBufferGetWidth(destination)
      let dstH = CVPixelBufferGetHeight(destination)
      guard dstW > 0, dstH > 0,
        let dstTex = makeTexture(
          destination,
          pixelFormat: .bgra8Unorm,
          width: dstW,
          height: dstH,
          planeIndex: 0,
          writable: true
        )
      else {
        encoder.endEncoding()
        return false
      }
      retained.append(dstTex.cvTexture)

      var params = MetalPatchParams(
        srcInnerOrigin: cropOrigin,
        dstOrigin: .zero,
        copySize: SIMD2<UInt32>(UInt32(dstW), UInt32(dstH))
      )

      if srcIsYUV {
        let srcYW = CVPixelBufferGetWidthOfPlane(source, 0)
        let srcYH = CVPixelBufferGetHeightOfPlane(source, 0)
        let srcUVW = CVPixelBufferGetWidthOfPlane(source, 1)
        let srcUVH = CVPixelBufferGetHeightOfPlane(source, 1)
        guard srcYW > 0, srcYH > 0, srcUVW > 0, srcUVH > 0,
          let srcYTex = makeTexture(
            source,
            pixelFormat: .r8Unorm,
            width: srcYW,
            height: srcYH,
            planeIndex: 0,
            writable: false
          ),
          let srcUVTex = makeTexture(
            source,
            pixelFormat: .rg8Unorm,
            width: srcUVW,
            height: srcUVH,
            planeIndex: 1,
            writable: false
          )
        else {
          encoder.endEncoding()
          return false
        }
        retained.append(srcYTex.cvTexture)
        retained.append(srcUVTex.cvTexture)
        encoder.setTexture(srcYTex.texture, index: 0)
        encoder.setTexture(srcUVTex.texture, index: 1)
        encoder.setTexture(dstTex.texture, index: 2)
        encoder.setBytes(
          &params,
          length: MemoryLayout<MetalPatchParams>.stride,
          index: 0
        )
        dispatch(
          encoder: encoder,
          pipeline: pipelines.nv12ToBgra,
          width: dstW,
          height: dstH
        )
      } else {
        let srcW = CVPixelBufferGetWidth(source)
        let srcH = CVPixelBufferGetHeight(source)
        guard srcW > 0, srcH > 0,
          let srcTex = makeTexture(
            source,
            pixelFormat: .bgra8Unorm,
            width: srcW,
            height: srcH,
            planeIndex: 0,
            writable: false
          )
        else {
          encoder.endEncoding()
          return false
        }
        retained.append(srcTex.cvTexture)
        encoder.setTexture(srcTex.texture, index: 0)
        encoder.setTexture(dstTex.texture, index: 1)
        encoder.setBytes(
          &params,
          length: MemoryLayout<MetalPatchParams>.stride,
          index: 0
        )
        dispatch(
          encoder: encoder,
          pipeline: pipelines.bgraPatch,
          width: dstW,
          height: dstH
        )
      }
    }

    encoder.endEncoding()
    return commitAndWait(commandBuffer, retaining: retained)
  }

  private func makeTexture(
    _ buffer: CVPixelBuffer,
    pixelFormat: MTLPixelFormat,
    width: Int,
    height: Int,
    planeIndex: Int,
    writable: Bool
  ) -> (texture: MTLTexture, cvTexture: CVMetalTexture)? {
    var cvTexture: CVMetalTexture?
    let attrs =
      writable
      ? PixelBufferConverter.writeUsage
      : PixelBufferConverter.readUsage
    cacheLock.lock()
    let status = CVMetalTextureCacheCreateTextureFromImage(
      kCFAllocatorDefault,
      textureCache,
      buffer,
      attrs,
      pixelFormat,
      width,
      height,
      planeIndex,
      &cvTexture
    )
    cacheLock.unlock()
    guard status == kCVReturnSuccess,
      let unwrapped = cvTexture,
      let texture = CVMetalTextureGetTexture(unwrapped)
    else {
      NSLog("PixelBufferConverter: CVMetalTexture create failed: \(status)")
      return nil
    }
    return (texture, unwrapped)
  }

  private func dispatch(
    encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState,
    width: Int,
    height: Int
  ) {
    guard width > 0, height > 0 else {
      return
    }
    let tw = max(1, pipeline.threadExecutionWidth)
    let th = max(1, pipeline.maxTotalThreadsPerThreadgroup / tw)
    let threadsPerThreadgroup = MTLSize(width: tw, height: th, depth: 1)
    let threadgroups = MTLSize(
      width: (width + tw - 1) / tw,
      height: (height + th - 1) / th,
      depth: 1
    )
    encoder.setComputePipelineState(pipeline)
    encoder.dispatchThreadgroups(
      threadgroups,
      threadsPerThreadgroup: threadsPerThreadgroup
    )
  }

  private func commitAndWait(
    _ commandBuffer: MTLCommandBuffer,
    retaining retained: [CVMetalTexture]
  ) -> Bool {
    withExtendedLifetime(retained) {
      commandBuffer.commit()
      commandBuffer.waitUntilCompleted()
    }
    cacheLock.lock()
    flushCountdown -= 1
    if flushCountdown <= 0 {
      flushCountdown = 120
      CVMetalTextureCacheFlush(textureCache, 0)
    }
    cacheLock.unlock()

    if let error = commandBuffer.error {
      NSLog("PixelBufferConverter: Metal command failed: \(error)")
      return false
    }
    return true
  }
}

/// Decides when the blankness probe may run.
///
/// `PixelBufferConverter.isBlankYUV` has to lock the base address, and for an
/// IOSurface backed buffer written by the GPU or the ANE that is a
/// synchronisation point: the CPU waits for the frame to finish. Doing it on
/// every frame — twice, once inside the VT stage and once before publishing —
/// costs several milliseconds per frame in exactly the path that is already
/// latency bound.
///
/// The probe still matters: it is the only way to tell "VideoToolbox reported
/// success but wrote nothing" apart from a legitimate black frame. So it runs
/// on the first frames, where a broken session shows up immediately, and then
/// samples periodically.
@available(iOS 16.0, *)
struct BlankCheckBudget {
  private let initialWindow: Int
  private var initialChecks: Int
  private var countdown: Int = 0
  private let period: Int

  init(initialChecks: Int = 30, period: Int = 240) {
    self.initialWindow = max(initialChecks, 1)
    self.initialChecks = self.initialWindow
    self.period = max(period, 1)
  }

  /// `true` when the caller should pay for a probe on this frame.
  func needsCheck() -> Bool {
    return initialChecks > 0 || countdown == 0
  }

  /// Records a probe result; any blank frame resets the startup window so a
  /// broken session can never leak unprobed green/black buffers to Flutter.
  mutating func recordCheckResult(isBlank: Bool) {
    if isBlank {
      initialChecks = initialWindow
      countdown = 0
    } else if initialChecks > 0 {
      initialChecks -= 1
    } else {
      countdown = period
    }
  }

  /// Advances the periodic skip countdown when `needsCheck()` returned `false`.
  mutating func recordSkipped() {
    if countdown > 0 {
      countdown -= 1
    }
  }

  /// Probes `buffer` on schedule and updates the budget.
  mutating func checkIfBlank(_ buffer: CVPixelBuffer) -> Bool {
    guard needsCheck() else {
      recordSkipped()
      return false
    }
    let blank = PixelBufferConverter.isBlank(buffer)
    recordCheckResult(isBlank: blank)
    return blank
  }

  /// Restores the startup window, e.g. after a session restart or resize.
  mutating func reset() {
    initialChecks = initialWindow
    countdown = 0
  }
}
