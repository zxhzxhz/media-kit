//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Shared plumbing for the iOS video enhancement pipeline: the frame contract,
//  a small bounded queue, and the protocol the TextureHW renderer talks to.

import CoreMedia
import CoreVideo
import Foundation

/// One enhanced frame produced by an enhancer.
final class EnhanceFrame {
  let pixelBuffer: CVPixelBuffer
  let presentationTime: CMTime

  init(pixelBuffer: CVPixelBuffer, presentationTime: CMTime) {
    self.pixelBuffer = pixelBuffer
    self.presentationTime = presentationTime
  }
}

/// A tiny thread safe FIFO that keeps the most recent N frames.
///
/// The enhancer writes from its own queue, \`TextureHW.copyPixelBuffer\` reads
/// from the Flutter raster thread. Older frames are dropped instead of blocking
/// the producer: dropping a stale frame is always preferable to stalling the
/// renderer.
final class EnhanceFrameQueue {
  private let lock = NSLock()
  private var storage: [EnhanceFrame] = []
  private let capacity: Int

  init(capacity: Int) {
    self.capacity = max(1, capacity)
  }

  func push(_ frame: EnhanceFrame) {
    lock.lock()
    storage.append(frame)
    while storage.count > capacity {
      storage.removeFirst()
    }
    lock.unlock()
  }

  /// Returns the newest frame without consuming it, so a single frame can be
  /// pulled by multiple Flutter raster passes.
  func latest() -> EnhanceFrame? {
    lock.lock()
    defer {
      lock.unlock()
    }
    return storage.last
  }

  func clear() {
    lock.lock()
    storage.removeAll()
    lock.unlock()
  }
}

/// The renderer facing contract of an enhancement pipeline.
///
/// Implementations must be safe to call from the \`TextureHW\` worker thread
/// (submit / configure / invalidate) and from the Flutter raster thread
/// (latestFrame). They must never block the raster thread.
protocol FrameEnhancerProtocol: AnyObject {
  /// \`true\` when the pipeline is configured and able to produce frames.
  var isActive: Bool { get }

  /// Invoked (on an internal queue) whenever a new frame becomes available
  /// asynchronously. \`TextureHW\` uses it to notify Flutter.
  var onFrameAvailable: (() -> Void)? { get set }

  /// Builds or rebuilds the pipeline. Returns \`false\` when the device cannot
  /// run the requested pipeline, in which case the caller must fall back to the
  /// plain texture path.
  ///
  /// - Parameters:
  ///   - configuration: the requested stages.
  ///   - inputSize: pixel size of the buffers produced by mpv.
  ///   - outputSize: pixel size reported to \`OUTPUT\` in the Anime4K \`WHEN\`
  ///     expressions (the size Flutter will eventually draw at).
  func configure(
    configuration: VideoEnhanceConfiguration,
    inputSize: CGSize,
    outputSize: CGSize
  ) -> Bool

  /// Submits one decoded frame. \`source\` is owned by the caller and may be
  /// recycled as soon as this call returns, so implementations that need to
  /// keep it must copy it.
  func submit(
    _ source: CVPixelBuffer,
    presentationTime: CMTime,
    frameIndex: Int64
  )

  /// Newest enhanced frame, or \`nil\` when nothing is ready yet.
  func latestFrame() -> CVPixelBuffer?

  /// Releases every resource. Safe to call more than once.
  func invalidate()
}
