//  Part of media_kit_video (Kazumi fork). See LICENSE.
//
//  Video enhancement pipeline configuration, shared between the Dart side and
//  the native renderer. Pure data: no platform specific API is referenced here
//  so both the iOS and the macOS targets can compile this file.

import Foundation

/// Describes the requested video enhancement pipeline.
///
/// The native side decides what it can actually do with it:
/// * VideoToolbox frame interpolation / super resolution: iOS 26.0+ only.
/// * Anime4K through Metal: any iOS version with a Metal capable GPU.
///
/// An instance is only created when at least one stage is enabled, so a non-nil
/// value always means "the user wants enhancement".
public class VideoEnhanceConfiguration {
  /// Whether VideoToolbox low latency frame interpolation (2x) is requested.
  public let interpolation: Bool

  /// VideoToolbox low latency super resolution scale factor.
  /// `0` disables the stage. Legal values must be validated at runtime against
  /// `VTLowLatencySuperResolutionScalerConfiguration.supportedScaleFactors`.
  public let vtScaleFactor: Double

  /// Anime4K GLSL sources (raw `.glsl` file contents), in pipeline order.
  /// An empty array disables the Metal Anime4K stage.
  public let anime4kShaders: [String]

  /// Human readable profile name ("efficiency" / "quality" / ...), used for
  /// logging and for caching compiled pipelines.
  public let anime4kProfile: String

  init(
    interpolation: Bool,
    vtScaleFactor: Double,
    anime4kShaders: [String],
    anime4kProfile: String
  ) {
    self.interpolation = interpolation
    self.vtScaleFactor = vtScaleFactor
    self.anime4kShaders = anime4kShaders
    self.anime4kProfile = anime4kProfile
  }

  /// `true` when at least one stage is requested.
  public var isEnabled: Bool {
    return interpolation || vtScaleFactor > 0 || !anime4kShaders.isEmpty
  }

  public var description: String {
    return
      "VideoEnhanceConfiguration(interpolation: \(interpolation), vtScaleFactor: \(vtScaleFactor), anime4kProfile: \(anime4kProfile), shaders: \(anime4kShaders.count))"
  }

  /// Builds a configuration from the Dart method channel dictionary.
  ///
  /// Returns `nil` when the dictionary is missing, malformed, or describes a
  /// disabled pipeline, so callers can treat `nil` as "no enhancement".
  public static func fromDict(
    _ dict: [String: Any]?
  ) -> VideoEnhanceConfiguration? {
    guard let dict = dict else {
      return nil
    }

    let interpolation = dict["interpolation"] as? Bool ?? false
    // Dart doubles arrive as NSNumber; `as? Double` is unreliable for those.
    let vtScaleFactor = (dict["vtScaleFactor"] as? NSNumber)?.doubleValue ?? 0
    let anime4kShaders = dict["anime4kShaders"] as? [String] ?? []
    let anime4kProfile = dict["anime4kProfile"] as? String ?? "off"

    let configuration = VideoEnhanceConfiguration(
      interpolation: interpolation,
      vtScaleFactor: vtScaleFactor,
      anime4kShaders: anime4kShaders,
      anime4kProfile: anime4kProfile
    )

    if !configuration.isEnabled {
      return nil
    }

    NSLog("VideoEnhanceConfiguration: \(configuration.description)")

    return configuration
  }
}
