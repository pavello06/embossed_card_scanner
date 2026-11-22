import 'package:camera/camera.dart';

import '../api/card_frame.dart';

/// Converts a frame from [CameraController.startImageStream]. Android
/// streams (NV21 / YUV420) use the Y plane directly, iOS BGRA8888 is reduced
/// to luma. [rotation] is usually `CameraDescription.sensorOrientation` for
/// a portrait-locked preview.
CardFrame cardFrameFromCameraImage(CameraImage image, int rotation) {
  final plane = image.planes.first;
  return image.format.group == ImageFormatGroup.bgra8888
      ? CardFrame.bgra(
          plane.bytes,
          width: image.width,
          height: image.height,
          rowStride: plane.bytesPerRow,
          rotation: rotation,
        )
      : CardFrame.luma(
          plane.bytes,
          width: image.width,
          height: image.height,
          rowStride: plane.bytesPerRow,
          rotation: rotation,
        );
}
