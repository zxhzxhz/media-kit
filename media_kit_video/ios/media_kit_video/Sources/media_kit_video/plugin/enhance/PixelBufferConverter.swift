//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  CVPixelBuffer format conversion for the VideoToolbox stages.
//
//  VTFrameProcessor only accepts the pixel formats it advertises in
//  `frameSupportedPixelFormats` — on an iPad Pro M4 that is `420v`
//  (kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, NV12 video range) — while
//  the OpenGL renderer produces `32BGRA`.
//
//  Buffers handed to VT must be created from the configuration's own
//  `sourcePixelBufferAttributes` / `destinationPixelBufferAttributes`: the ANE
//  backend has hard alignment / extended-pixel / cache-mode requirements, and a
//  pool that merely looks compatible (same format and size) can make VT report
//  success while writing nothing.

import CoreImage
import CoreVideo
import Foundation
import Metal

/// Converts CVPixelBuffers between two pixel formats at a fixed size.
@available(iOS 16.0, *)
final class PixelBufferConverter {
  private let context: CIContext
  private let pool: CVPixelBufferPool?
  private let width: Int
  private let height: Int
  /// Format this converter produces.
  let pixelFormat: OSType

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
      let device = MTLCreateSystemDefaultDevice()
    else {
      NSLog("PixelBufferConverter: Metal unavailable")
      return nil
    }

    // Explicit BT.709 rather than NSNull(): telling CoreImage "no colour space"
    // makes it skip the colour management step, and rendering a 420v source
    // into a 4K BGRA destination then silently produces an empty buffer.
    let colorSpace = CGColorSpace(name: CGColorSpace.itur_709) as Any
    self.context = CIContext(
      mtlDevice: device,
      options: [
        .workingColorSpace: colorSpace,
        .outputColorSpace: colorSpace,
        .cacheIntermediates: false,
      ]
    )
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

  /// Converts `source` into a caller supplied buffer of the target format.
  ///
  /// CoreImage picks the pixel format from the destination buffer, so the same
  /// call handles BGRA -> 420v and 420v -> BGRA.
  @discardableResult
  func convert(_ source: CVPixelBuffer, to destination: CVPixelBuffer) -> Bool {
    let image = CIImage(cvPixelBuffer: source)
    context.render(image, to: destination)
    PixelBufferConverter.applyColorAttachments(to: destination)
    return true
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
    let format = CVPixelBufferGetPixelFormatType(buffer)
    guard
      format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        || format == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
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

  /// Tags a buffer BT.709.
  ///
  /// Without a YCbCr matrix VideoToolbox and `CVPixelTransferSession` can
  /// produce empty or mis-coloured output without reporting an error, because
  /// neither knows how to interpret the planes.
  static func applyColorAttachments(to buffer: CVPixelBuffer) {
    let isYUV =
      CVPixelBufferGetPixelFormatType(buffer)
      == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
      || CVPixelBufferGetPixelFormatType(buffer)
        == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange

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
    if isYUV {
      CVBufferSetAttachment(
        buffer,
        kCVImageBufferYCbCrMatrixKey,
        kCVImageBufferYCbCrMatrix_ITU_R_709_2,
        .shouldPropagate
      )
    }
  }
}
