//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  CVPixelBuffer format conversion for the VideoToolbox stages.
//
//  VTFrameProcessor only accepts the pixel formats it advertises in
//  `frameSupportedPixelFormats` — on an iPad Pro M4 that is `420v`
//  (kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange, NV12 video range) — while
//  the OpenGL renderer produces `32BGRA`. Every frame therefore needs one
//  conversion in each direction. `VTPixelTransferSession` runs that on the
//  hardware converter, so 1080p costs a few milliseconds per frame.

import CoreVideo
import Foundation
import VideoToolbox

/// Converts CVPixelBuffers between two pixel formats at a fixed size.
///
/// `VTPixelTransferSession` is iOS 16+; every caller lives behind the
/// `@available(iOS 26.0, *)` VideoToolbox restriction, so gating here is enough.
@available(iOS 16.0, *)
final class PixelBufferConverter {
  private let session: VTPixelTransferSession
  private let pool: CVPixelBufferPool?
  private let width: Int
  private let height: Int
  /// Format this converter produces.
  let pixelFormat: OSType

  /// - Parameter providesPool: when `true` a `CVPixelBufferPool` of the target
  ///   format is created, so callers can obtain destination buffers cheaply.
  ///   Pass `false` when the caller always supplies its own destination.
  init?(width: Int, height: Int, pixelFormat: OSType, providesPool: Bool) {
    guard width > 0, height > 0 else {
      return nil
    }

    var session: VTPixelTransferSession?
    guard
      VTPixelTransferSessionCreate(
        allocator: kCFAllocatorDefault,
        pixelTransferSessionOut: &session
      ) == noErr,
      let created = session
    else {
      NSLog("PixelBufferConverter: VTPixelTransferSessionCreate failed")
      return nil
    }
    self.session = created
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

  deinit {
    VTPixelTransferSessionInvalidate(session)
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
  @discardableResult
  func convert(_ source: CVPixelBuffer, to destination: CVPixelBuffer) -> Bool {
    let status = VTPixelTransferSessionTransferImage(
      session,
      from: source,
      to: destination
    )
    guard status == noErr else {
      NSLog("PixelBufferConverter: transfer failed: \(status)")
      return false
    }
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
