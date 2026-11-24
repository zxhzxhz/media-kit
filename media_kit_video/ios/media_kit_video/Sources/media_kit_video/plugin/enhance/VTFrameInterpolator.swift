//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Low-latency 2x frame interpolation via VideoToolbox's VTFrameProcessor (iOS
//  26+): one frame at phase 0.5 between two consecutive frames, written into a
//  caller supplied CVPixelBuffer. Interpolation only: no super resolution, no
//  HDR, no Metal.
//
//  The processor forbids concurrent submissions per session, so calls are
//  serialised on a private queue; the caller owns all buffers and the
//  destination must be pooled and match `init(width:height:)`.

import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

/// Failures surfaced through `VTFrameInterpolator.interpolate(...)`.
enum VTFrameInterpolatorError: Error {
  /// No session is running: never started, or `invalidate()` was called.
  case notReady
  /// A buffer does not match the configured frame size.
  case dimensionMismatch(
    expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int)
  /// `VTFrameProcessorFrame` rejected a buffer.
  case frameCreationFailed
  case interpolationFailed(Error)
}

/// Availability probe that is callable from iOS 13 deployment targets.
///
/// `VTFrameInterpolator` is `@available(iOS 26.0, *)`, and Swift gates the whole
/// type with it, so iOS 13 code can not name the class at all - not even its
/// static members. This shim mirrors `VTFrameInterpolator.isSupported`.
enum VTFrameInterpolationSupport {
  static func isSupported(width: Int, height: Int) -> Bool {
    guard width > 0, height > 0 else { return false }
    if #available(iOS 26.0, *) {
      return VTFrameInterpolationSupport.probe(width: width, height: height)
    }
    return false
  }

  @available(iOS 26.0, *)
  private static func probe(width: Int, height: Int) -> Bool {
    // A nil configuration means this size is unsupported on this hardware.
    return VTLowLatencyFrameInterpolationConfiguration(
      frameWidth: width, frameHeight: height, numberOfInterpolatedFrames: 1
    ) != nil
  }
}

/// Wraps a single `VTFrameProcessor` low-latency interpolation session.
@available(iOS 26.0, *)
final class VTFrameInterpolator {
  private let frameWidth: Int
  private let frameHeight: Int
  /// Serialises submissions: the processor forbids concurrent work.
  private let queue: DispatchQueue
  /// Guards `processor` for readers outside `queue` (namely `isReady`).
  private let lock = NSLock()
  private var processor: VTFrameProcessor?

  /// Starts the session; nil when the size is rejected or `startSession` throws.
  init?(width: Int, height: Int) {
    self.frameWidth = width
    self.frameHeight = height
    self.queue = DispatchQueue(label: "media_kit_video.vt.interpolation",
                               qos: .userInteractive)
    guard width > 0, height > 0,
      let configuration = VTLowLatencyFrameInterpolationConfiguration(
        frameWidth: width, frameHeight: height, numberOfInterpolatedFrames: 1
      )
    else { return nil }

    // UNCERTAIN: assumes the no-argument `VTFrameProcessor()` initialiser.
    let processor = VTFrameProcessor()
    do {
      try processor.startSession(configuration: configuration)
    } catch {
      return nil
    }
    self.processor = processor
  }

  deinit {
    processor?.endSession()  // best effort; `invalidate()` is the real path
  }

  var isReady: Bool {
    lock.lock()
    defer { lock.unlock() }
    return processor != nil
  }

  /// Generates one frame at phase 0.5 between `previous` and `source`, written
  /// into `destination`. All three buffers must be `frameWidth` x `frameHeight`;
  /// `destination` is normally drained from the caller's `CVPixelBufferPool`.
  /// VideoToolbox forbids touching the buffers (attachments included) until
  /// `completion` fires; it always runs asynchronously on the internal queue.
  func interpolate(
    previous: CVPixelBuffer,
    previousTime: CMTime,
    source: CVPixelBuffer,
    sourceTime: CMTime,
    destination: CVPixelBuffer,
    completion: @escaping (Result<Void, Error>) -> Void
  ) {
    queue.async {
      self.lock.lock()
      let processor = self.processor
      self.lock.unlock()

      guard let processor = processor else {
        self.finish(completion, .failure(VTFrameInterpolatorError.notReady))
        return
      }

      for buffer in [previous, source, destination] {
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        if width != self.frameWidth || height != self.frameHeight {
          let error = VTFrameInterpolatorError.dimensionMismatch(
            expectedWidth: self.frameWidth, expectedHeight: self.frameHeight,
            actualWidth: width, actualHeight: height
          )
          self.finish(completion, .failure(error))
          return
        }
      }

      // UNCERTAIN: the docs do not state whether the processor derives output
      // timing from the destination timestamp or ignores it; for phase 0.5 the
      // midpoint of the two source frames is the natural value.
      let delta = CMTimeSubtract(sourceTime, previousTime)
      let destinationTime = CMTimeAdd(previousTime,
                                      CMTimeMultiplyByFloat64(delta, multiplier: 0.5))

      guard
        let previousFrame = VTFrameProcessorFrame(
          buffer: previous, presentationTimeStamp: previousTime
        ),
        let sourceFrame = VTFrameProcessorFrame(
          buffer: source, presentationTimeStamp: sourceTime
        ),
        let destinationFrame = VTFrameProcessorFrame(
          buffer: destination, presentationTimeStamp: destinationTime
        )
      else {
        self.finish(completion, .failure(VTFrameInterpolatorError.frameCreationFailed))
        return
      }

      // UNCERTAIN: documented with exactly these four arguments, but the frame
      // rate conversion sibling also takes `submissionMode:`; add
      // `.sequential` here if this initialiser turns out to require it.
      let parameters = VTLowLatencyFrameInterpolationParameters(
        sourceFrame: sourceFrame, previousFrame: previousFrame,
        interpolationPhase: [0.5], destinationFrames: [destinationFrame]
      )

      // UNCERTAIN: documented as `(VTFrameProcessorParameters?, Error?) -> Void`,
      // yet some header dumps of the low latency variant show `(Error?) -> Void`.
      // Types are inferred to match the first shape; on an arity mismatch replace
      // `{ _, error in` with `{ error in`.
      processor.process(parameters: parameters) { _, error in
        guard let error = error else {
          self.finish(completion, .success(()))
          return
        }
        self.finish(completion,
                    .failure(VTFrameInterpolatorError.interpolationFailed(error)))
      }
    }
  }

  /// Ends the session and drops the processor; pending submissions fail with
  /// `notReady`. Call it only after the last completion fired.
  func invalidate() {
    queue.async {
      self.lock.lock()
      let processor = self.processor
      self.processor = nil
      self.lock.unlock()

      processor?.endSession()
    }
  }

  /// Mirrors `VTFrameInterpolationSupport.isSupported(width:height:)`, but only
  /// reachable behind `#available(iOS 26.0, *)` since the type itself is gated.
  static func isSupported(width: Int, height: Int) -> Bool {
    return VTFrameInterpolationSupport.isSupported(width: width, height: height)
  }

  private func finish(_ completion: @escaping (Result<Void, Error>) -> Void,
                      _ result: Result<Void, Error>) {
    queue.async { completion(result) }
  }
}
