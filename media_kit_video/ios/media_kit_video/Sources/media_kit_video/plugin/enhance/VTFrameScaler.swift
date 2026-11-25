//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Low-latency super resolution via VideoToolbox's VTFrameProcessor (iOS 26+):
//  - Runs directly when the source dimensions fit the device's
//    `VTLowLatencySuperResolutionScalerConfiguration` ceiling (e.g. `1280x1280`
//    on iPad Pro M4).
//  - Splits larger frames (e.g. 1920x1080 -> 3840x2160) into minimum-count
//    overlapped tiles within those hardware limits (trying `2x1` before `2x2`
//    with automatic runtime fallback to `2x2`), overlaps GPU cropping of the
//    next tile with ANE inference, and stitches their non-overlapping cores back
//    in a single GPU pass with zero seams.
//
//  The processor forbids concurrent submissions per session, so calls are
//  serialised on a private queue; the caller owns all buffers.

import CoreGraphics
import CoreMedia
import CoreVideo
import Foundation
import Metal
import VideoToolbox

// VideoToolbox marks the whole VTFrameProcessor family as
// `#if !TARGET_OS_SIMULATOR`, so this file is device only.
#if !targetEnvironment(simulator)

  /// Failures surfaced through `VTFrameScaler.scale(...)`.
  enum VTFrameScalerError: Error {
    /// No session is running: never started, or `invalidate()` was called.
    case notReady
    /// A buffer does not match the configured frame size.
    case dimensionMismatch(
      expectedWidth: Int, expectedHeight: Int, actualWidth: Int, actualHeight: Int)
    /// `VTFrameProcessorFrame` rejected a buffer.
    case frameCreationFailed
    /// The scale factor is not in `supportedScaleFactors`.
    case unsupportedScaleFactor(Float)
    case scalingFailed(Error)
    /// `process` neither completed nor reported an error in time.
    case timedOut
    /// BGRA <-> the VideoToolbox format conversion failed.
    case formatConversionFailed
    /// VT reported success but the destination planes are empty.
    case blankOutput(String)
  }

  struct ScaleTileSpec {
    let srcCropRect: CGRect
    let outInnerRect: CGRect
    let destOrigin: CGPoint
  }

  struct ScaleTilePlan {
    let cols: Int
    let rows: Int
    let tileWidth: Int
    let tileHeight: Int
    let outTileWidth: Int
    let outTileHeight: Int
    let tiles: [ScaleTileSpec]

    var isTiled: Bool {
      return cols > 1 || rows > 1
    }
  }

  /// Availability probe callable from iOS 13 deployment targets.
  enum VTFrameScalingSupport {
    static func isSupported(
      width: Int,
      height: Int,
      scaleFactor: Float
    ) -> Bool {
      guard width > 0, height > 0, scaleFactor > 1.0 else {
        return false
      }
      if #available(iOS 26.0, *) {
        return VTFrameScalingSupport.probe(
          width: width,
          height: height,
          scaleFactor: scaleFactor
        )
      }
      return false
    }

    /// Human readable probe result, including hardware limits, tile geometry, and formats.
    static func describe(width: Int, height: Int, scaleFactor: Float) -> String {
      guard width > 0, height > 0 else { return "invalid size" }
      if #available(iOS 26.0, *) {
        return VTFrameScalingSupport.describe26(
          width: width,
          height: height,
          scaleFactor: scaleFactor
        )
      }
      return "needs iOS 26"
    }

    @available(iOS 26.0, *)
    static func supportsDirectScaler(
      width: Int,
      height: Int,
      scaleFactor: Float
    ) -> Bool {
      guard VTLowLatencySuperResolutionScalerConfiguration.isSupported else {
        return false
      }
      if let maxDim = VTLowLatencySuperResolutionScalerConfiguration.maximumDimensions,
        width > Int(maxDim.width) || height > Int(maxDim.height)
      {
        return false
      }
      if let minDim = VTLowLatencySuperResolutionScalerConfiguration.minimumDimensions,
        width < Int(minDim.width) || height < Int(minDim.height)
      {
        return false
      }
      let scales = VTLowLatencySuperResolutionScalerConfiguration
        .__supportedScaleFactors(forFrameWidth: width, frameHeight: height)
        .map { $0.floatValue }
      guard scales.contains(where: { abs($0 - scaleFactor) < 0.01 }) else {
        return false
      }
      let configuration = VTLowLatencySuperResolutionScalerConfiguration(
        frameWidth: width,
        frameHeight: height,
        scaleFactor: scaleFactor
      )
      return !configuration.__frameSupportedPixelFormats.isEmpty
    }

    @available(iOS 26.0, *)
    static func resolveTilePlan(
      width: Int,
      height: Int,
      scaleFactor: Float,
      minRowsOverride: Int = 1
    ) -> ScaleTilePlan? {
      guard width > 0, height > 0, scaleFactor > 1.0,
        VTLowLatencySuperResolutionScalerConfiguration.isSupported
      else {
        return nil
      }

      if minRowsOverride <= 1,
        supportsDirectScaler(
          width: width,
          height: height,
          scaleFactor: scaleFactor
        )
      {
        let outW = Int((Float(width) * scaleFactor).rounded())
        let outH = Int((Float(height) * scaleFactor).rounded())
        let tile = ScaleTileSpec(
          srcCropRect: CGRect(x: 0, y: 0, width: width, height: height),
          outInnerRect: CGRect(x: 0, y: 0, width: outW, height: outH),
          destOrigin: .zero
        )
        return ScaleTilePlan(
          cols: 1,
          rows: 1,
          tileWidth: width,
          tileHeight: height,
          outTileWidth: outW,
          outTileHeight: outH,
          tiles: [tile]
        )
      }

      let maxW = Int(
        VTLowLatencySuperResolutionScalerConfiguration.maximumDimensions?.width
          ?? 1280
      )
      let maxH = Int(
        VTLowLatencySuperResolutionScalerConfiguration.maximumDimensions?.height
          ?? 1280
      )
      guard maxW >= 96, maxH >= 96 else {
        return nil
      }

      let minCols = max(1, Int(ceil(Double(width) / Double(maxW))))
      let minRows = max(
        minRowsOverride,
        max(1, Int(ceil(Double(height) / Double(maxH))))
      )

      var candidateGrids: [(cols: Int, rows: Int)] = []
      for c in minCols...4 {
        for r in minRows...4 {
          if (width + c - 1) / c <= maxW && (height + r - 1) / r <= maxH {
            candidateGrids.append((c, r))
          }
        }
      }
      candidateGrids.sort { lhs, rhs in
        let lCount = lhs.cols * lhs.rows
        let rCount = rhs.cols * rhs.rows
        if lCount != rCount {
          return lCount < rCount
        }
        return lhs.rows < rhs.rows
      }

      let overlapCandidates: [(overlap: Int, align: Int)] = [
        (32, 16),
        (16, 16),
        (0, 16),
        (0, 2),
      ]

      for grid in candidateGrids {
        let cols = grid.cols
        let rows = grid.rows
        let baseW = (width + cols - 1) / cols
        let baseH = (height + rows - 1) / rows

        for candidate in overlapCandidates {
          let overlap = candidate.overlap
          let align = candidate.align
          let rawW =
            cols == 1
            ? width
            : min(width, ((baseW + overlap + align - 1) / align) * align)
          let rawH =
            rows == 1
            ? height
            : min(height, ((baseH + overlap + align - 1) / align) * align)
          guard rawW >= baseW, rawH >= baseH,
            rawW <= maxW, rawH <= maxH,
            rawW % 2 == 0, rawH % 2 == 0
          else {
            continue
          }
          let outTileW = Int((Float(rawW) * scaleFactor).rounded())
          let outTileH = Int((Float(rawH) * scaleFactor).rounded())
          guard outTileW % 2 == 0, outTileH % 2 == 0 else {
            continue
          }
          guard
            supportsDirectScaler(
              width: rawW,
              height: rawH,
              scaleFactor: scaleFactor
            )
          else {
            continue
          }
          var tiles: [ScaleTileSpec] = []
          tiles.reserveCapacity(rows * cols)

          for r in 0..<rows {
            let sliceStartY = (height * r) / rows
            let sliceEndY = (height * (r + 1)) / rows
            let sliceH = sliceEndY - sliceStartY
            var cropY =
              rows == 1
              ? 0
              : (max(0, min(height - rawH, sliceStartY - (rawH - sliceH) / 2)) & ~1)
            if cropY > sliceStartY {
              cropY = max(0, sliceStartY & ~1)
            }
            if cropY + rawH < sliceEndY {
              cropY = min(height - rawH, sliceEndY - rawH)
            }

            let destY = Int((Float(sliceStartY) * scaleFactor).rounded())
            let destEndY = Int((Float(sliceEndY) * scaleFactor).rounded())
            let innerY = Int((Float(sliceStartY - cropY) * scaleFactor).rounded())

            for c in 0..<cols {
              let sliceStartX = (width * c) / cols
              let sliceEndX = (width * (c + 1)) / cols
              let sliceW = sliceEndX - sliceStartX
              var cropX =
                cols == 1
                ? 0
                : (max(0, min(width - rawW, sliceStartX - (rawW - sliceW) / 2)) & ~1)
              if cropX > sliceStartX {
                cropX = max(0, sliceStartX & ~1)
              }
              if cropX + rawW < sliceEndX {
                cropX = min(width - rawW, sliceEndX - rawW)
              }

              let destX = Int((Float(sliceStartX) * scaleFactor).rounded())
              let destEndX = Int((Float(sliceEndX) * scaleFactor).rounded())
              let innerX = Int((Float(sliceStartX - cropX) * scaleFactor).rounded())

              tiles.append(
                ScaleTileSpec(
                  srcCropRect: CGRect(x: cropX, y: cropY, width: rawW, height: rawH),
                  outInnerRect: CGRect(
                    x: innerX,
                    y: innerY,
                    width: destEndX - destX,
                    height: destEndY - destY
                  ),
                  destOrigin: CGPoint(x: destX, y: destY)
                )
              )
            }
          }

          return ScaleTilePlan(
            cols: cols,
            rows: rows,
            tileWidth: rawW,
            tileHeight: rawH,
            outTileWidth: outTileW,
            outTileHeight: outTileH,
            tiles: tiles
          )
        }
      }

      return nil
    }

    @available(iOS 26.0, *)
    private static func probe(
      width: Int,
      height: Int,
      scaleFactor: Float
    ) -> Bool {
      guard
        let plan = resolveTilePlan(
          width: width,
          height: height,
          scaleFactor: scaleFactor
        )
      else {
        return false
      }
      let configuration = VTLowLatencySuperResolutionScalerConfiguration(
        frameWidth: plan.tileWidth,
        frameHeight: plan.tileHeight,
        scaleFactor: scaleFactor
      )
      let processor = VTFrameProcessor()
      do {
        try processor.startSession(configuration: configuration)
      } catch {
        VTFrameInterpolationSupport.recordStartError(error)
        NSLog("VTFrameScaler: probe startSession failed: \(error)")
        return false
      }
      processor.endSession()
      return true
    }

    @available(iOS 26.0, *)
    static func summarizeConfiguration(
      _ configuration: VTLowLatencySuperResolutionScalerConfiguration,
      tilePlan: ScaleTilePlan,
      scaleFactor: Float
    ) -> String {
      let maxDim = VTLowLatencySuperResolutionScalerConfiguration.maximumDimensions
      let minDim = VTLowLatencySuperResolutionScalerConfiguration.minimumDimensions
      let maxDesc = maxDim.map { "\($0.width)x\($0.height)" } ?? "nil"
      let minDesc = minDim.map { "\($0.width)x\($0.height)" } ?? "nil"
      let scales = VTLowLatencySuperResolutionScalerConfiguration
        .__supportedScaleFactors(
          forFrameWidth: tilePlan.tileWidth,
          frameHeight: tilePlan.tileHeight
        )
        .map { String(format: "%.2g", $0.floatValue) }
        .joined(separator: ",")
      let formats = configuration.__frameSupportedPixelFormats.map {
        $0.uint32Value
      }
      let fmts = PixelBufferConverter.formatListDescription(formats)
      let srcF = PixelBufferConverter.pixelFormatSummary(
        from: configuration.sourcePixelBufferAttributes
      )
      let dstF = PixelBufferConverter.pixelFormatSummary(
        from: configuration.destinationPixelBufferAttributes
      )
      return "max=\(maxDesc) min=\(minDesc) "
        + "tile=\(tilePlan.cols)x\(tilePlan.rows)(\(tilePlan.tileWidth)x\(tilePlan.tileHeight)) "
        + "scales=[\(scales)] fmts=[\(fmts)] srcF=\(srcF) dstF=\(dstF)"
    }

    @available(iOS 26.0, *)
    private static func describe26(
      width: Int,
      height: Int,
      scaleFactor: Float
    ) -> String {
      var parts: [String] = []
      parts.append("scaleFactor \(scaleFactor)")
      parts.append(
        "processorSupported=\(VTLowLatencySuperResolutionScalerConfiguration.isSupported)"
      )

      if let plan = resolveTilePlan(
        width: width,
        height: height,
        scaleFactor: scaleFactor
      ) {
        let configuration = VTLowLatencySuperResolutionScalerConfiguration(
          frameWidth: plan.tileWidth,
          frameHeight: plan.tileHeight,
          scaleFactor: scaleFactor
        )
        parts.append(
          summarizeConfiguration(
            configuration,
            tilePlan: plan,
            scaleFactor: scaleFactor
          )
        )
      } else {
        let maxDim = VTLowLatencySuperResolutionScalerConfiguration.maximumDimensions
        let minDim = VTLowLatencySuperResolutionScalerConfiguration.minimumDimensions
        let maxDesc = maxDim.map { "\($0.width)x\($0.height)" } ?? "nil"
        let minDesc = minDim.map { "\($0.width)x\($0.height)" } ?? "nil"
        parts.append("no tile plan for \(width)x\(height) (max=\(maxDesc) min=\(minDesc))")
      }

      let error = VTFrameInterpolationSupport.lastStartError()
      if !error.isEmpty {
        parts.append(error)
      }
      return parts.joined(separator: "; ")
    }
  }

  /// Wraps a single `VTFrameProcessor` low-latency super resolution session.
  @available(iOS 26.0, *)
  final class VTFrameScaler {
    private enum ScaleMode: String {
      case yuvAsync
      case yuvMetal
      case yuvVT
      case bgraAsync
      case bgraMetal
      case bgraDirectAsync

      var pixelFormat: OSType {
        switch self {
        case .yuvAsync, .yuvMetal, .yuvVT:
          return kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        case .bgraAsync, .bgraMetal, .bgraDirectAsync:
          return kCVPixelFormatType_32BGRA
        }
      }

      var usesMetalCommandBuffer: Bool {
        switch self {
        case .yuvMetal, .bgraMetal:
          return true
        case .yuvAsync, .yuvVT, .bgraAsync, .bgraDirectAsync:
          return false
        }
      }
    }

    /// Per-tile timeout (`process` must finish before the next tile/frame).
    private static let processTimeout = DispatchTimeInterval.milliseconds(200)

    private let frameWidth: Int
    private let frameHeight: Int
    private let outputWidth: Int
    private let outputHeight: Int
    private let scaleFactor: Float
    private var tilePlan: ScaleTilePlan
    private var configuration: VTLowLatencySuperResolutionScalerConfiguration
    private let metalCommandQueue: MTLCommandQueue?
    /// Serialises submissions: the processor forbids concurrent work.
    private let queue: DispatchQueue
    /// Guards `processor`, stats, and active converters during mode/grid fallback.
    private let lock = NSLock()
    private var processor: VTFrameProcessor?

    /// Pixel formats VideoToolbox advertises for this tile configuration.
    private(set) var supportedFormats: [OSType] = []
    /// Target format negotiated for VT tile inputs and tile outputs.
    private(set) var targetPixelFormat: OSType = kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
    /// `false` when the advertised formats are known and do not include BGRA.
    private(set) var acceptsBGRASource = true
    /// `true` when BGRA <-> VT format conversion is set up.
    private(set) var canConvertPixelFormat = false

    private var candidateModes: [ScaleMode] = []
    private var modeIndex: Int = 0
    private var activeMode: ScaleMode = .yuvAsync
    private var modeLocked: Bool = false
    private var probeInfo: String
    private var triedModesSummary: String = ""
    private var recordedTriedModes: Set<String> = []

    /// Full frame (or cropped tile) -> VT tile format (`tileWidth x tileHeight`).
    private var inputConverter: PixelBufferConverter?
    /// VT tile destination pool (`outTileWidth x outTileHeight`).
    private var tileOutputConverter: PixelBufferConverter?
    /// Full output pool (`outputWidth x outputHeight`). Equals `tileOutputConverter` when untiled.
    private var finalDestinationConverter: PixelBufferConverter?
    /// Sampling budget for the blankness probe (see `BlankCheckBudget`).
    private var blankBudget = BlankCheckBudget()

    private var submitted: Int = 0
    private var succeeded: Int = 0
    private var failed: Int = 0
    private var timedOut: Int = 0
    /// Most recent failure, surfaced so "nothing submitted" can be told apart
    /// from "submitted and rejected".
    private var lastError: String = ""

    var backendLabel: String {
      lock.lock()
      let cols = tilePlan.cols
      let rows = tilePlan.rows
      let isTiled = tilePlan.isTiled
      let modeName = activeMode.rawValue
      lock.unlock()
      if isTiled {
        return "\(cols)x\(rows):\(modeName)"
      }
      return modeName
    }

    /// Compact counters, surfaced through the enhancer status string.
    var statsDescription: String {
      lock.lock()
      let modeName =
        tilePlan.isTiled
        ? "\(tilePlan.cols)x\(tilePlan.rows):\(activeMode.rawValue)"
        : activeMode.rawValue
      let fmt = targetPixelFormat
      let s = submitted
      let ok = succeeded
      let f = failed
      let to = timedOut
      let err = lastError
      let probe = probeInfo
      let tried = triedModesSummary
      lock.unlock()
      var text =
        "scale[\(modeName):\(PixelBufferConverter.fourCC(fmt))] submit=\(s) ok=\(ok) fail=\(f) to=\(to)"
      if !err.isEmpty {
        text += " last=\(err)"
      }
      if ok == 0 {
        text += " \(probe)"
        if !tried.isEmpty {
          text += " tried=[\(tried)]"
        }
      }
      return text
    }

    /// Output size for a given input, exposed so the caller can size its pool.
    static func outputSize(
      width: Int,
      height: Int,
      scaleFactor: Float
    ) -> CGSize {
      // Ensure even output dimensions for 4:2:0 chroma planes at fractional scales (e.g. 1.5x).
      let w = (Int((Float(width) * scaleFactor).rounded()) + 1) & ~1
      let h = (Int((Float(height) * scaleFactor).rounded()) + 1) & ~1
      return CGSize(
        width: CGFloat(w),
        height: CGFloat(h)
      )
    }

    init?(width: Int, height: Int, scaleFactor: Float) {
      guard width > 0, height > 0, scaleFactor > 1.0,
        let plan = VTFrameScalingSupport.resolveTilePlan(
          width: width,
          height: height,
          scaleFactor: scaleFactor
        )
      else {
        return nil
      }
      self.frameWidth = width
      self.frameHeight = height
      self.scaleFactor = scaleFactor
      self.tilePlan = plan

      let output = VTFrameScaler.outputSize(
        width: width,
        height: height,
        scaleFactor: scaleFactor
      )
      self.outputWidth = Int(output.width)
      self.outputHeight = Int(output.height)

      self.queue = DispatchQueue(
        label: "media_kit_video.vt.scaler",
        qos: .userInteractive
      )
      self.metalCommandQueue = MTLCreateSystemDefaultDevice()?.makeCommandQueue()

      let configuration = VTLowLatencySuperResolutionScalerConfiguration(
        frameWidth: plan.tileWidth,
        frameHeight: plan.tileHeight,
        scaleFactor: scaleFactor
      )
      self.configuration = configuration
      self.probeInfo = VTFrameScalingSupport.summarizeConfiguration(
        configuration,
        tilePlan: plan,
        scaleFactor: scaleFactor
      )

      guard
        let finalPool = PixelBufferConverter(
          width: self.outputWidth,
          height: self.outputHeight,
          pixelFormat: kCVPixelFormatType_32BGRA,
          providesPool: true,
          vtAttributes: nil
        )
      else {
        NSLog("VTFrameScaler: cannot create 32BGRA output pool")
        return nil
      }
      self.finalDestinationConverter = finalPool

      self.buildCandidateModes(for: configuration, isTiled: plan.isTiled)

      var initialConfigured = false
      let formats = self.supportedFormats
      for (index, mode) in self.candidateModes.enumerated() {
        let overrideFormat: OSType? =
          (self.candidateModes.count == 1
            && !formats.contains(kCVPixelFormatType_32BGRA)
            && !formats.contains(kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange))
          ? formats.first
          : nil
        if self.configurePools(for: mode, explicitFormat: overrideFormat) {
          self.modeIndex = index
          initialConfigured = true
          break
        }
      }
      guard initialConfigured else {
        NSLog("VTFrameScaler: cannot create scaler pools for any candidate mode")
        return nil
      }

      NSLog(
        "VTFrameScaler: mode=\(backendLabel) target=\(PixelBufferConverter.fourCC(targetPixelFormat)) \(probeInfo)"
      )

      let processor = VTFrameProcessor()
      do {
        try processor.startSession(configuration: configuration)
      } catch {
        VTFrameInterpolationSupport.recordStartError(error)
        NSLog("VTFrameScaler: startSession failed: \(error)")
        return nil
      }
      self.processor = processor
    }

    deinit {
      processor?.endSession()
    }

    private func buildCandidateModes(
      for configuration: VTLowLatencySuperResolutionScalerConfiguration,
      isTiled: Bool
    ) {
      let formats = configuration.__frameSupportedPixelFormats.map {
        $0.uint32Value
      }
      self.supportedFormats = formats
      if !formats.isEmpty {
        self.acceptsBGRASource = formats.contains(kCVPixelFormatType_32BGRA)
      }

      var modes: [ScaleMode] = []
      var seenModes: Set<ScaleMode> = []
      func appendMode(_ mode: ScaleMode) {
        if seenModes.insert(mode).inserted {
          modes.append(mode)
        }
      }
      for format in formats {
        if format == kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange {
          appendMode(.yuvAsync)
          appendMode(.yuvMetal)
          if !isTiled {
            appendMode(.yuvVT)
          }
        } else if format == kCVPixelFormatType_32BGRA {
          appendMode(.bgraAsync)
          appendMode(.bgraMetal)
          if !isTiled {
            appendMode(.bgraDirectAsync)
          }
        }
      }
      if modes.isEmpty {
        modes = [.bgraAsync]
      }
      self.candidateModes = modes
    }

    private func configurePools(
      for mode: ScaleMode,
      explicitFormat: OSType? = nil
    ) -> Bool {
      let format = explicitFormat ?? mode.pixelFormat
      guard
        let tileOut = PixelBufferConverter(
          width: tilePlan.outTileWidth,
          height: tilePlan.outTileHeight,
          pixelFormat: format,
          providesPool: true,
          vtAttributes: configuration.destinationPixelBufferAttributes
        ),
        let input = PixelBufferConverter(
          width: tilePlan.tileWidth,
          height: tilePlan.tileHeight,
          pixelFormat: format,
          providesPool: true,
          vtAttributes: configuration.sourcePixelBufferAttributes
        )
      else {
        return false
      }

      lock.lock()
      self.activeMode = mode
      self.targetPixelFormat = format
      self.tileOutputConverter = tileOut
      self.inputConverter = input
      self.canConvertPixelFormat = true
      lock.unlock()
      return true
    }

    var isReady: Bool {
      lock.lock()
      defer {
        lock.unlock()
      }
      return processor != nil
    }

    /// A buffer at the full output size (`outputWidth` x `outputHeight`) ready for Flutter.
    func makeDestinationBuffer() -> CVPixelBuffer? {
      lock.lock()
      let converter = finalDestinationConverter
      lock.unlock()
      return converter?.makeBuffer()
    }

    /// Upscales `source` into `destination`.
    ///
    /// `destination` must be `outputWidth` x `outputHeight` and come from a
    /// pool: VideoToolbox forbids touching either buffer until `completion`
    /// fires. Runs asynchronously on the internal queue.
    func scale(
      source: CVPixelBuffer,
      sourceTime: CMTime,
      destination: CVPixelBuffer,
      completion: @escaping (Result<Void, Error>) -> Void
    ) {
      queue.async {
        self.lock.lock()
        let processor = self.processor
        let mode = self.activeMode
        let expectedFormat = self.targetPixelFormat
        let inputConverter = self.inputConverter
        let tileOutputConverter = self.tileOutputConverter
        let finalConverter = self.finalDestinationConverter
        let plan = self.tilePlan
        self.lock.unlock()

        guard let processor = processor else {
          self.finish(completion, .failure(VTFrameScalerError.notReady))
          return
        }

        let sourceWidth = CVPixelBufferGetWidth(source)
        let sourceHeight = CVPixelBufferGetHeight(source)
        if sourceWidth != self.frameWidth || sourceHeight != self.frameHeight {
          self.finish(
            completion,
            .failure(
              VTFrameScalerError.dimensionMismatch(
                expectedWidth: self.frameWidth,
                expectedHeight: self.frameHeight,
                actualWidth: sourceWidth,
                actualHeight: sourceHeight
              )
            )
          )
          return
        }

        let destWidth = CVPixelBufferGetWidth(destination)
        let destHeight = CVPixelBufferGetHeight(destination)
        if destWidth != self.outputWidth || destHeight != self.outputHeight {
          self.finish(
            completion,
            .failure(
              VTFrameScalerError.dimensionMismatch(
                expectedWidth: self.outputWidth,
                expectedHeight: self.outputHeight,
                actualWidth: destWidth,
                actualHeight: destHeight
              )
            )
          )
          return
        }

        let pts = sourceTime.isValid ? sourceTime : .zero

        if !plan.isTiled {
          guard
            CVPixelBufferGetPixelFormatType(destination)
              == kCVPixelFormatType_32BGRA,
            let tileOutputConverter = tileOutputConverter,
            let finalConverter = finalConverter
          else {
            self.finish(
              completion,
              .failure(VTFrameScalerError.formatConversionFailed)
            )
            return
          }

          let sourceBuffer: CVPixelBuffer
          switch mode {
          case .bgraDirectAsync:
            if CVPixelBufferGetPixelFormatType(source) == kCVPixelFormatType_32BGRA {
              sourceBuffer = source
            } else if let converted = inputConverter?.convert(source) {
              sourceBuffer = converted
            } else {
              self.finish(
                completion,
                .failure(VTFrameScalerError.formatConversionFailed)
              )
              return
            }
          case .yuvVT:
            guard let converted = inputConverter?.convertViaVT(source) else {
              self.finish(
                completion,
                .failure(VTFrameScalerError.formatConversionFailed)
              )
              return
            }
            sourceBuffer = converted
          case .yuvAsync, .yuvMetal, .bgraAsync, .bgraMetal:
            guard let converted = inputConverter?.convert(source) else {
              self.finish(
                completion,
                .failure(VTFrameScalerError.formatConversionFailed)
              )
              return
            }
            sourceBuffer = converted
          }

          let vtDestBuffer: CVPixelBuffer
          if mode == .bgraDirectAsync {
            vtDestBuffer = destination
          } else {
            guard let pooledOut = tileOutputConverter.makeBuffer() else {
              self.finish(completion, .failure(VTFrameScalerError.notReady))
              return
            }
            vtDestBuffer = pooledOut
          }

          let outcome = self.runSingleVTScale(
            processor: processor,
            mode: mode,
            sourceBuffer: sourceBuffer,
            destinationBuffer: vtDestBuffer,
            pts: pts,
            whileWaiting: nil
          )
          if let err = outcome.error {
            self.finish(completion, .failure(err))
            return
          }

          PixelBufferConverter.applyColorAttachments(to: vtDestBuffer)

          // Probe the 420v buffer before BGRA conversion so an unwritten
          // all-zero output is caught rather than turned into dark green.
          if self.blankBudget.checkIfBlank(vtDestBuffer) {
            let srcSample = PixelBufferConverter.sampleCenter(sourceBuffer)
            let dstSample = PixelBufferConverter.sampleCenter(vtDestBuffer)
            let detail = "\(mode.rawValue) src=\(srcSample) dst=\(dstSample)"
            if !self.modeLocked && !PixelBufferConverter.isBlank(source) {
              self.advanceModeAfterBlank(
                failedMode: mode,
                srcSample: srcSample,
                dstSample: dstSample
              )
            }
            self.finish(
              completion,
              .failure(VTFrameScalerError.blankOutput(detail))
            )
            return
          }

          if mode != .bgraDirectAsync {
            guard finalConverter.convert(vtDestBuffer, to: destination) else {
              self.finish(
                completion,
                .failure(VTFrameScalerError.formatConversionFailed)
              )
              return
            }
          }

          self.modeLocked = true
          self.lock.lock()
          self.succeeded += 1
          self.lock.unlock()
          self.finish(completion, .success(()))
          return
        }

        // Tiled path: crop tile 0 first, then overlap tile[i+1] GPU crop
        // with async ANE execution of tile[i].
        guard let inputConverter = inputConverter,
          let tileOutputConverter = tileOutputConverter,
          let finalConverter = finalConverter
        else {
          self.finish(completion, .failure(VTFrameScalerError.notReady))
          return
        }

        var completedTiles:
          [(buffer: CVPixelBuffer, innerRect: CGRect, destOrigin: CGPoint)] = []
        completedTiles.reserveCapacity(plan.tiles.count)

        var currentTileIn = inputConverter.convertCropped(
          source,
          cropRect: plan.tiles[0].srcCropRect
        )

        for (index, tile) in plan.tiles.enumerated() {
          guard let tileIn = currentTileIn else {
            self.finish(
              completion,
              .failure(VTFrameScalerError.formatConversionFailed)
            )
            return
          }
          guard let tileOut = tileOutputConverter.makeBuffer() else {
            self.finish(completion, .failure(VTFrameScalerError.notReady))
            return
          }

          let prepareNext: (() -> CVPixelBuffer?)?
          if index + 1 < plan.tiles.count {
            let nextRect = plan.tiles[index + 1].srcCropRect
            prepareNext = {
              inputConverter.convertCropped(source, cropRect: nextRect)
            }
          } else {
            prepareNext = nil
          }

          let outcome = self.runSingleVTScale(
            processor: processor,
            mode: mode,
            sourceBuffer: tileIn,
            destinationBuffer: tileOut,
            pts: pts,
            whileWaiting: prepareNext
          )

          if let err = outcome.error {
            // If 2x1 (single row) hit a runtime hardware limit on frame 0
            // (e.g. -19730), fall back to 2x2 tiles immediately.
            if index == 0, !self.modeLocked, plan.rows == 1,
              case .scalingFailed(let underlying) = err
            {
              _ = self.fallbackToMultiRowGrid(
                failedLabel: "\(plan.cols)x\(plan.rows)(\(plan.tileWidth)x\(plan.tileHeight))",
                error: underlying
              )
            }
            self.finish(completion, .failure(err))
            return
          }

          currentTileIn = outcome.sideResult

          PixelBufferConverter.applyColorAttachments(to: tileOut)

          // While probing modes on tile 0, fail fast instead of running remaining tiles.
          if index == 0, !self.modeLocked,
            PixelBufferConverter.isBlank(tileOut),
            !PixelBufferConverter.isBlank(tileIn)
          {
            let srcSample = PixelBufferConverter.sampleCenter(tileIn)
            let dstSample = PixelBufferConverter.sampleCenter(tileOut)
            self.advanceModeAfterBlank(
              failedMode: mode,
              srcSample: srcSample,
              dstSample: dstSample
            )
            self.finish(
              completion,
              .failure(
                VTFrameScalerError.blankOutput(
                  "\(mode.rawValue) tile0 src=\(srcSample) dst=\(dstSample)"
                )
              )
            )
            return
          }

          completedTiles.append(
            (buffer: tileOut, innerRect: tile.outInnerRect, destOrigin: tile.destOrigin)
          )
        }

        // Probe the 420v tiles before compositing into 32BGRA so an unwritten
        // all-zero ANE session cannot leak dark green frames to Flutter.
        let needTileBlankCheck = self.blankBudget.needsCheck()
        if !needTileBlankCheck {
          self.blankBudget.recordSkipped()
        } else {
          let allTilesBlank =
            !completedTiles.isEmpty
            && completedTiles.allSatisfy {
              PixelBufferConverter.isBlank($0.buffer)
            }
          self.blankBudget.recordCheckResult(isBlank: allTilesBlank)
          if allTilesBlank {
            let srcSample = PixelBufferConverter.sampleCenter(source)
            let dstSample = PixelBufferConverter.sampleCenter(
              completedTiles[0].buffer
            )
            let detail =
              "\(plan.cols)x\(plan.rows):\(mode.rawValue) src=\(srcSample) dst=\(dstSample)"
            if !self.modeLocked && !PixelBufferConverter.isBlank(source) {
              self.advanceModeAfterBlank(
                failedMode: mode,
                srcSample: srcSample,
                dstSample: dstSample
              )
            }
            self.finish(
              completion,
              .failure(VTFrameScalerError.blankOutput(detail))
            )
            return
          }
        }

        guard finalConverter.compositeTiles(completedTiles, to: destination) else {
          self.finish(
            completion,
            .failure(VTFrameScalerError.formatConversionFailed)
          )
          return
        }

        self.modeLocked = true
        self.lock.lock()
        self.succeeded += 1
        self.lock.unlock()
        self.finish(completion, .success(()))
      }
    }

    private func runSingleVTScale(
      processor: VTFrameProcessor,
      mode: ScaleMode,
      sourceBuffer: CVPixelBuffer,
      destinationBuffer: CVPixelBuffer,
      pts: CMTime,
      whileWaiting: (() -> CVPixelBuffer?)?
    ) -> (error: VTFrameScalerError?, sideResult: CVPixelBuffer?) {
      guard
        let sourceFrame = VTFrameProcessorFrame(
          buffer: sourceBuffer,
          presentationTimeStamp: pts
        ),
        let destinationFrame = VTFrameProcessorFrame(
          buffer: destinationBuffer,
          presentationTimeStamp: pts
        )
      else {
        return (.frameCreationFailed, nil)
      }

      let parameters = VTLowLatencySuperResolutionScalerParameters(
        sourceFrame: sourceFrame,
        destinationFrame: destinationFrame
      )

      self.lock.lock()
      self.submitted += 1
      self.lock.unlock()

      if mode.usesMetalCommandBuffer {
        guard let commandQueue = self.metalCommandQueue,
          let commandBuffer = commandQueue.makeCommandBuffer()
        else {
          return (.notReady, nil)
        }
        processor.process(with: commandBuffer, parameters: parameters)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
          return (.scalingFailed(error), nil)
        }
        return (nil, whileWaiting?())
      }

      let semaphore = DispatchSemaphore(value: 0)
      let errorLock = NSLock()
      var processError: Error?
      processor.process(parameters: parameters) { _, error in
        errorLock.lock()
        processError = error
        errorLock.unlock()
        semaphore.signal()
      }

      // Run next tile's GPU crop while ANE is executing the current tile.
      let sideResult = whileWaiting?()

      if semaphore.wait(timeout: .now() + VTFrameScaler.processTimeout)
        == .timedOut
      {
        self.lock.lock()
        self.timedOut += 1
        self.lock.unlock()
        return (.timedOut, nil)
      }

      errorLock.lock()
      let error = processError
      errorLock.unlock()

      if let error = error {
        return (.scalingFailed(error), nil)
      }
      return (nil, sideResult)
    }

    /// Runs on `self.queue`. Rebuilds the VT session with at least 2 rows
    /// if a `2x1` tile exceeded an undocumented runtime hardware pixel limit.
    private func fallbackToMultiRowGrid(
      failedLabel: String,
      error: Error
    ) -> Bool {
      guard
        let fallbackPlan = VTFrameScalingSupport.resolveTilePlan(
          width: frameWidth,
          height: frameHeight,
          scaleFactor: scaleFactor,
          minRowsOverride: 2
        )
      else {
        return false
      }

      let nextConfig = VTLowLatencySuperResolutionScalerConfiguration(
        frameWidth: fallbackPlan.tileWidth,
        frameHeight: fallbackPlan.tileHeight,
        scaleFactor: scaleFactor
      )
      let nextProcessor = VTFrameProcessor()
      do {
        try nextProcessor.startSession(configuration: nextConfig)
      } catch {
        return false
      }

      lock.lock()
      let oldProcessor = self.processor
      self.processor = nil
      self.tilePlan = fallbackPlan
      self.configuration = nextConfig
      self.probeInfo = VTFrameScalingSupport.summarizeConfiguration(
        nextConfig,
        tilePlan: fallbackPlan,
        scaleFactor: scaleFactor
      )
      let note = "\(failedLabel):\(error.localizedDescription)"
      if triedModesSummary.isEmpty {
        triedModesSummary = note
      } else {
        triedModesSummary += ", \(note)"
      }
      lock.unlock()

      oldProcessor?.endSession()

      buildCandidateModes(for: nextConfig, isTiled: fallbackPlan.isTiled)
      var configured = false
      for (index, mode) in candidateModes.enumerated() {
        if configurePools(for: mode) {
          lock.lock()
          modeIndex = index
          lock.unlock()
          configured = true
          break
        }
      }
      guard configured else {
        nextProcessor.endSession()
        return false
      }

      lock.lock()
      self.processor = nextProcessor
      lock.unlock()

      NSLog(
        "VTFrameScaler: \(failedLabel) runtime failed (\(error)), falling back to "
          + "\(fallbackPlan.cols)x\(fallbackPlan.rows)(\(fallbackPlan.tileWidth)x\(fallbackPlan.tileHeight))"
      )
      return true
    }

    private func advanceModeAfterBlank(
      failedMode: ScaleMode,
      srcSample: String,
      dstSample: String
    ) {
      let note = "\(failedMode.rawValue)(src=\(srcSample),dst=\(dstSample))"
      lock.lock()
      if !recordedTriedModes.contains(failedMode.rawValue) {
        recordedTriedModes.insert(failedMode.rawValue)
        if triedModesSummary.isEmpty {
          triedModesSummary = note
        } else {
          triedModesSummary += ", \(note)"
        }
      }
      var nextIndex = modeIndex + 1
      let modes = candidateModes
      lock.unlock()

      while nextIndex < modes.count {
        let candidate = modes[nextIndex]
        if configurePools(for: candidate) {
          lock.lock()
          modeIndex = nextIndex
          lock.unlock()
          NSLog(
            "VTFrameScaler: \(failedMode.rawValue) blank (\(srcSample) -> \(dstSample)), switching to \(candidate.rawValue)"
          )
          return
        }
        nextIndex += 1
      }
    }

    /// Tears the session down. Safe to call more than once.
    func invalidate() {
      lock.lock()
      let processor = self.processor
      self.processor = nil
      lock.unlock()

      guard let processor = processor else {
        return
      }
      queue.async {
        processor.endSession()
      }
    }

    /// Every failure funnels through here so the counters and the last error
    /// stay accurate even for branches that fail before submission.
    private func finish(
      _ completion: @escaping (Result<Void, Error>) -> Void,
      _ result: Result<Void, Error>
    ) {
      if case .failure(let error) = result {
        lock.lock()
        failed += 1
        lastError = "\(error)"
        lock.unlock()
      }
      queue.async { completion(result) }
    }
  }

#endif  // !targetEnvironment(simulator)
