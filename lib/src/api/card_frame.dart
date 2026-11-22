import 'dart:typed_data';

import 'package:meta/meta.dart';

import '../reader/frame_crop.dart';

/// One camera frame as delivered by the sensor (before rotation). Only
/// brightness is used, so any format reduces to a gray plane.
class CardFrame {
  CardFrame._(this.luma);

  /// A gray (luma) plane, one byte per pixel; row `y` starts at
  /// `y * rowStride` (default [width]). This is the Y plane of NV21,
  /// YUV_420_888 and other YUV formats, so pass `planes.first` as is.
  ///
  /// [rotation] is the clockwise rotation (0, 90, 180, 270) that makes the
  /// frame upright, e.g. the camera's sensor orientation for portrait.
  factory CardFrame.luma(Uint8List bytes, {required int width, required int height, int? rowStride, int rotation = 0}) {
    final stride = rowStride ?? width;
    if (width <= 0 || height <= 0 || stride < width) {
      throw ArgumentError('Invalid frame size ${width}x$height, row stride $stride');
    }
    if (bytes.length < stride * (height - 1) + width) {
      throw ArgumentError('Frame has ${bytes.length} bytes, expected at least ${stride * (height - 1) + width}');
    }
    if (rotation % 90 != 0) throw ArgumentError.value(rotation, 'rotation', 'must be 0, 90, 180 or 270');
    return CardFrame._(
      LumaFrame(bytes: bytes, width: width, height: height, rowStride: stride, rotation: rotation % 360),
    );
  }

  /// A BGRA8888 frame (iOS camera), [rowStride] in bytes (default `width * 4`).
  factory CardFrame.bgra(
    Uint8List bytes, {
    required int width,
    required int height,
    int? rowStride,
    int rotation = 0,
  }) => CardFrame.luma(
    LumaFrame.lumaFromBgra(bytes, width, height, rowStride ?? width * 4),
    width: width,
    height: height,
    rotation: rotation,
  );

  @internal
  final LumaFrame luma;

  /// Size of the upright frame: the coordinate space of guide rectangles.
  int get uprightWidth => luma.uprightWidth;
  int get uprightHeight => luma.uprightHeight;
}
