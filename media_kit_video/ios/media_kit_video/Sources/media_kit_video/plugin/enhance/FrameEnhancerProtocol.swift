//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Shared plumbing for the iOS video enhancement pipeline: the frame contract,
//  a small bounded queue with PTS lookup, and the protocol the TextureHW
//  renderer talks to.

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
/// The enhancer writes from its own queue, `TextureHW.copyPixelBuffer` reads
/// from the Flutter raster thread and the display link picks a frame on the
/// `Worker` thread. Older frames are dropped instead of blocking the producer:
/// dropping a stale frame is always preferable to stalling the renderer.
final class EnhanceFrameQueue {
  private let lock = NSLock()
  private var storage: [EnhanceFrame] = []
  private let capacity: Int

  init(capacity: Int) {
    self.capacity = max(2, capacity)
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

  /// Returns the frame whose presentation time is closest to `time`.
  ///
  /// This is what makes interpolated frames visible: the display link asks for
  /// the frame that belongs to the current clock instead of always taking the
  /// newest one (which would let a freshly decoded frame hide the interpolated
  /// frame that came before it).
  func nearest(to time: CMTime) -> EnhanceFrame? {
    lock.lock()
    defer {
      lock.unlock()
    }
    guard !storage.isEmpty else {
      return nil
    }
    guard time.isValid else {
      return storage.last
    }

    var best = storage[0]
    var bestDistance = EnhanceFrameQueue.distance(best.presentationTime, time)
    for frame in storage.dropFirst() {
      let distance = EnhanceFrameQueue.distance(frame.presentationTime, time)
      if distance < bestDistance {
        best = frame
        bestDistance = distance
      }
    }
    return best
  }

  /// Drops frames that fall behind `time` by more than `tolerance`.
  func prune(before time: CMTime, tolerance: CMTime) {
    guard time.isValid else {
      return
    }
    lock.lock()
    let threshold = CMTimeSubtract(time, tolerance)
    storage.removeAll { frame in
      frame.presentationTime.isValid
        && CMTimeCompare(frame.presentationTime, threshold) < 0
    }
    lock.unlock()
  }

  func clear() {
    lock.lock()
    storage.removeAll()
    lock.unlock()
  }

  var count: Int {
    lock.lock()
    defer {
      lock.unlock()
    }
    return storage.count
  }

  private static func distance(_ lhs: CMTime, _ rhs: CMTime) -> Double {
    guard lhs.isValid, rhs.isValid else {
      return .greatestFiniteMagnitude
    }
    return abs(CMTimeGetSeconds(CMTimeSubtract(lhs, rhs)))
  }
}

/// The renderer facing contract of an enhancement pipeline.
///
/// Implementations must be safe to call from the `TextureHW` worker thread
/// (submit / configure / invalidate), from the display link tick on that same
/// worker (frame(nearestTo:)) and from the Flutter raster thread
/// (latestFrame). They must never block the raster thread.
protocol FrameEnhancerProtocol: AnyObject {
  /// `true` when the pipeline is configured and able to produce frames.
  var isActive: Bool { get }

  /// Human readable state, surfaced to Dart so that "enhancement is on but
  /// nothing happens" becomes diagnosable from inside the app.
  var statusDescription: String { get }

  /// Invoked (on an internal queue) whenever a new frame becomes available
  /// asynchronously. `TextureHW` uses it to notify Flutter.
  var onFrameAvailable: (() -> Void)? { get set }

  /// Builds or rebuilds the pipeline. Returns `false` when the device cannot
  /// run the requested pipeline, in which case the caller must fall back to the
  /// plain texture path.
  ///
  /// - Parameters:
  ///   - configuration: the requested stages.
  ///   - inputSize: pixel size of the buffers produced by mpv.
  ///   - outputSize: pixel size reported to `OUTPUT` in the Anime4K `WHEN`
  ///     expressions (the size Flutter will eventually draw at).
  func configure(
    configuration: VideoEnhanceConfiguration,
    inputSize: CGSize,
    outputSize: CGSize
  ) -> Bool

  /// Submits one decoded frame. `source` is owned by the caller and may be
  /// recycled as soon as this call returns, so implementations that need to
  /// keep it must copy it.
  func submit(
    _ source: CVPixelBuffer,
    presentationTime: CMTime,
    frameIndex: Int64
  )

  /// Newest enhanced frame, or `nil` when nothing is ready yet.
  func latestFrame() -> CVPixelBuffer?

  /// Frame whose presentation time is closest to `time`. Used by the display
  /// link to present the right frame for the current clock.
  func frame(nearestTo time: CMTime) -> CVPixelBuffer?

  /// Releases every resource. Safe to call more than once.
  func invalidate()
}
