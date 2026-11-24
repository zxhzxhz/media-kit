/// This file is a part of media_kit (https://github.com/media-kit/media-kit).
///
/// Copyright © 2021 & onwards, Hitesh Kumar Saini <saini123hitesh@gmail.com>.
/// All rights reserved.
/// Use of this source code is governed by MIT license that can be found in the LICENSE file.
import 'dart:async';
import 'package:flutter/widgets.dart';

import 'package:media_kit/media_kit.dart';

import 'package:media_kit_video/src/video_controller/video_controller.dart';

/// {@template platform_video_controller}
///
/// PlatformVideoController
/// -----------------------
///
/// This class provides the interface for platform specific [VideoController] implementations.
/// The platform specific implementations are expected to implement the methods accordingly.
///
/// The subclasses are then used in composition with the [VideoController] class, based on the platform the application is running on.
///
/// {@endtemplate}
abstract class PlatformVideoController {
  /// The [Player] instance associated with this instance.
  final Player player;

  /// User defined configuration for [VideoController].
  final VideoControllerConfiguration configuration;

  /// Texture ID of the video output, registered with Flutter engine by the native implementation.
  final ValueNotifier<int?> id = ValueNotifier<int?>(null);

  /// [Rect] of the video output, received from the native implementation.
  final ValueNotifier<Rect?> rect = ValueNotifier<Rect?>(null);

  /// {@macro platform_video_controller}
  PlatformVideoController(
    this.player,
    this.configuration,
  );

  /// Sets the required size of the video output.
  /// This may yield substantial performance improvements if a small [width] & [height] is specified.
  ///
  /// Remember:
  /// * “Premature optimization is the root of all evil”
  /// * “With great power comes great responsibility”
  Future<void> setSize({
    int? width,
    int? height,
  });

  /// A [Future] that completes when the first video frame has been rendered.
  Future<void> get waitUntilFirstFrameRendered =>
      waitUntilFirstFrameRenderedCompleter.future;

  /// [Completer] used to signal the decoding & rendering of the first video frame.
  /// Use [waitUntilFirstFrameRendered] to wait for the first frame to be rendered.
  @protected
  final waitUntilFirstFrameRenderedCompleter = Completer<void>();

  void dispose() {
    id.dispose();
    rect.dispose();
  }
}

/// {@template video_enhance_configuration}
///
/// VideoEnhanceConfiguration
/// -------------------------
/// Configurable options for the native video enhancement pipeline.
///
/// These options are currently honored by the iOS implementation only
/// (iOS 26.0 & above). On every other platform, the native implementations
/// ignore them & the video output is rendered as-is.
///
/// {@endtemplate}
class VideoEnhanceConfiguration {
  /// Whether to enable low-latency 2x frame interpolation through
  /// `VTFrameProcessor` on native backend.
  ///
  /// Default: `false`
  final bool interpolation;

  /// The scale factor for the low-latency super resolution through
  /// `VTFrameProcessor` on native backend.
  ///
  /// `0` disables super resolution.
  ///
  /// Default: `0.0`
  final double vtScaleFactor;

  /// List of Anime4K GLSL shader sources, applied in the given order.
  ///
  /// An empty list disables Anime4K shader-based upscaling.
  ///
  /// Default: `const []`
  final List<String> anime4kShaders;

  /// Name of the Anime4K profile.
  ///
  /// Only used by the native implementation for caching & logging purposes.
  ///
  /// Default: `'off'`
  final String anime4kProfile;

  /// Whether at least one enhancement stage is enabled.
  bool get isEnabled =>
      interpolation || vtScaleFactor > 0 || anime4kShaders.isNotEmpty;

  /// {@macro video_enhance_configuration}
  const VideoEnhanceConfiguration({
    this.interpolation = false,
    this.vtScaleFactor = 0.0,
    this.anime4kShaders = const [],
    this.anime4kProfile = 'off',
  });

  /// Returns a map representation of this class, used for passing the
  /// configuration to the native implementation.
  Map<String, dynamic> toMap() => {
        'interpolation': interpolation,
        'vtScaleFactor': vtScaleFactor,
        'anime4kShaders': anime4kShaders,
        'anime4kProfile': anime4kProfile,
      };

  /// Returns a copy of this class with the given fields replaced by the new values.
  VideoEnhanceConfiguration copyWith({
    bool? interpolation,
    double? vtScaleFactor,
    List<String>? anime4kShaders,
    String? anime4kProfile,
  }) =>
      VideoEnhanceConfiguration(
        interpolation: interpolation ?? this.interpolation,
        vtScaleFactor: vtScaleFactor ?? this.vtScaleFactor,
        anime4kShaders: anime4kShaders ?? this.anime4kShaders,
        anime4kProfile: anime4kProfile ?? this.anime4kProfile,
      );
}

/// {@template video_controller_configuration}
///
/// VideoControllerConfiguration
/// ----------------------------
/// Configurable options for customizing the [VideoController] behavior.
///
/// {@endtemplate}
class VideoControllerConfiguration {
  /// Sets the [`--vo`](https://mpv.io/manual/stable/#options-vo) property on native backend.
  ///
  /// Default: Platform specific.
  /// * Windows, GNU/Linux, macOS & iOS: `libmpv`
  /// * Android: `gpu`
  /// * Ohos: `gpu-next`
  final String? vo;

  /// Sets the [`--hwdec`](https://mpv.io/manual/stable/#options-hwdec) property on native backend.
  ///
  /// Default: Platform specific.
  /// * Windows, GNU/Linux, macOS & iOS, Ohos : `auto`
  /// * Android: `auto-safe`
  final String? hwdec;

  /// The scale for the video output.
  /// This may be used for performance reasons. Specifying this option will cause [width] & [height] to be ignored.
  ///
  /// Default: `1.0`
  final double scale;

  /// The fixed width for the video output.
  /// This may be used for performance reasons.
  ///
  /// Default: `null`
  final int? width;

  /// The fixed height for the video output.
  /// This may be used for performance reasons.
  ///
  /// Default: `null`
  final int? height;

  /// Whether to enable hardware acceleration.
  ///
  /// DO NOT DISABLE THIS OPTION MEANINGLESSLY.
  /// THE BATTERY WILL DRAIN, THE DEVICE MAY HEAT UP & CPU USAGE WILL BE HIGH.
  ///
  /// Default: `true`
  final bool enableHardwareAcceleration;

  /// Whether to use Flutter's `SurfaceProducer` API on Android.
  ///
  /// This option only has effect on Android. If disabled, the Android
  /// implementation uses the `SurfaceTexture` code path instead. The
  /// `SurfaceTexture` code path is only effective with Android's Skia backend.
  ///
  /// Default: `true`
  final bool enableAndroidSurfaceProducer;

  /// Whether to attach `android.view.Surface` after video parameters are known.
  ///
  /// Default:
  /// * [vo] == gpu : `true`
  /// * [vo] != gpu : `false`
  final bool? androidAttachSurfaceAfterVideoParameters;

  /// Native video enhancement pipeline configuration.
  ///
  /// This option only has effect on iOS 26.0 & above. On other platforms
  /// the value is ignored.
  ///
  /// Default: `null`
  final VideoEnhanceConfiguration? enhance;

  /// {@macro video_controller_configuration}
  const VideoControllerConfiguration({
    this.vo,
    this.hwdec,
    this.width,
    this.height,
    this.scale = 1.0,
    this.enableHardwareAcceleration = true,
    this.enableAndroidSurfaceProducer = true,
    this.androidAttachSurfaceAfterVideoParameters,
    this.enhance,
  });

  /// Returns a copy of this class with the given fields replaced by the new values.
  VideoControllerConfiguration copyWith({
    String? vo,
    String? hwdec,
    double? scale,
    int? width,
    int? height,
    bool? enableHardwareAcceleration,
    bool? enableAndroidSurfaceProducer,
    bool? androidAttachSurfaceAfterVideoParameters,
    VideoEnhanceConfiguration? enhance,
  }) =>
      VideoControllerConfiguration(
        vo: vo ?? this.vo,
        hwdec: hwdec ?? this.hwdec,
        scale: scale ?? this.scale,
        width: width ?? this.width,
        height: height ?? this.height,
        enableHardwareAcceleration:
            enableHardwareAcceleration ?? this.enableHardwareAcceleration,
        enableAndroidSurfaceProducer:
            enableAndroidSurfaceProducer ?? this.enableAndroidSurfaceProducer,
        androidAttachSurfaceAfterVideoParameters:
            androidAttachSurfaceAfterVideoParameters ??
                this.androidAttachSurfaceAfterVideoParameters,
        enhance: enhance ?? this.enhance,
      );
}
