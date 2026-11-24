public class VideoOutputConfiguration {
  public let width: Int64?
  public let height: Int64?
  public let enableHardwareAcceleration: Bool
  /// Requested iOS video enhancement pipeline. \`nil\` means "plain rendering".
  public let enhance: VideoEnhanceConfiguration?

  init(
    width: Int64?,
    height: Int64?,
    enableHardwareAcceleration: Bool,
    enhance: VideoEnhanceConfiguration? = nil
  ) {
    self.width = width
    self.height = height
    self.enableHardwareAcceleration = enableHardwareAcceleration
    self.enhance = enhance
  }

  public static func fromDict(_ dict: [String: Any])
    -> VideoOutputConfiguration
  {
    let widthStr = dict["width"] as! String
    let heightStr = dict["height"] as! String
    let enableHardwareAcceleration =
      dict["enableHardwareAcceleration"] as! Bool
    // The key is optional so that other embedders of the plugin keep working.
    let enhance = VideoEnhanceConfiguration.fromDict(
      dict["enhance"] as? [String: Any]
    )

    let width: Int64? = Int64(widthStr)
    let height: Int64? = Int64(heightStr)

    return VideoOutputConfiguration(
      width: width,
      height: height,
      enableHardwareAcceleration: enableHardwareAcceleration,
      enhance: enhance
    )
  }
}
