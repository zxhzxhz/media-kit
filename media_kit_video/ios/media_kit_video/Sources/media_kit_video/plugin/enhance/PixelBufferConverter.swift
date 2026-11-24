//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  CVPixelBuffer format conversion for the VideoToolbox stages.
//
//  VTFrameProcessor only accepts the pixel formats it advertises in
//  `frameSupportedPixelFormats` — on an iPad Pro M4 that is `420v`
//  (kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, NV12 video range) — while
//  the OpenGL renderer produces `32BGRA`.
//
//  `VTPixelTransferSession` was tried first: it returned noErr for the
//  BGRA -> 420v direction yet produced a completely empty buffer on this
//  device, which surfaced as a black picture (`scaled frame is blank`). The
//  same conversion through CoreImage/Metal does produce valid output.

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

  /// - Parameter providesPool: when `true` a `CVPixelBufferPool` of the target
  ///   format is created, so callers can obtain destination buffers cheaply.
  ///   Pass `false` when the caller always supplies its own destination.
  init?(width: Int, height: Int, pixelFormat: OSType, providesPool: Bool) {
    guard width > 0, height > 0,
      let device = MTLCreateSystemDefaultDevice()
    else {
      NSLog("PixelBufferConverter: Metal unavailable")
      return nil
    }

    // No colour space conversion: the frames already carry the display's
    // colour information and only the pixel format has to change.
    self.context = CIContext(
      mtlDevice: device,
      options: [
        .workingColorSpace: NSNull(),
        .outputColorSpace: NSNull(),
        .cacheIntermediates: false,
      ]
    )
    self.width = width
    self.height = height
    self.pixelFormat = pixelFormat

    if providesPool {
      let attributes: [String: Any] = [
        kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
        kCVPixelBufferWidthKey as String: width,
        kCVPixelBufferHeightKey as String: height,
        kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
        kCVPixelBufferMetalCompatibilityKey as String: true,
        kCVPixelBufferPoolMinimumBufferCountKey as String: 4,
      ]
      var pool: CVPixelBufferPool?
      guard
        CVPixelBufferPoolCreate(
          kCFAllocatorDefault,
          nil,
          attributes as CFDictionary,
          &pool
        ) == kCVReturnSuccess,
        let createdPool = pool
      else {
        NSLog("PixelBufferConverter: pool creation failed")
        return nil
      }
      self.pool = createdPool
    } else {
      self.pool = nil
    }
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
        == kCVReturnSuccess
    else {
      return nil
    }
    return buffer
  }
}
