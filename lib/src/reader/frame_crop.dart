import 'dart:math' as math;
import 'dart:typed_data';

import 'embossed_reader.dart';

/// A gray camera frame as delivered by the sensor (before rotation).
class LumaFrame {
  const LumaFrame({
    required this.bytes,
    required this.width,
    required this.height,
    required this.rowStride,
    required this.rotation,
  });

  /// One byte per pixel; row `y` starts at `y * rowStride`.
  final Uint8List bytes;
  final int width;
  final int height;
  final int rowStride;

  /// Clockwise rotation (0, 90, 180, 270) that makes the frame upright.
  final int rotation;

  /// Size of the upright image, the coordinate space of the on-screen guide.
  int get uprightWidth => rotation % 180 == 0 ? width : height;
  int get uprightHeight => rotation % 180 == 0 ? height : width;

  /// Luma from a BGRA frame (iOS), with the weights used in training.
  static Uint8List lumaFromBgra(Uint8List bgra, int width, int height, int rowStride) {
    final out = Uint8List(width * height);
    for (var y = 0; y < height; y++) {
      var i = y * rowStride;
      for (var x = 0; x < width; x++, i += 4) {
        out[y * width + x] = ((bgra[i + 2] * 299 + bgra[i + 1] * 587 + bgra[i] * 114) / 1000).round();
      }
    }
    return out;
  }

  /// Crops [guide] (in upright coordinates), rotates and resamples it to the
  /// normalized card frame in one pass, without intermediate images.
  Uint8List normalizeCard(math.Rectangle<double> guide) {
    final outW = CardGeometry.width, outH = CardGeometry.height;
    final out = Uint8List(outW * outH);
    final sx = guide.width / outW, sy = guide.height / outH;
    // One bilinear tap per pixel near 1:1 (the usual camera case); average
    // several only when shrinking a lot, to avoid aliasing.
    final shrink = math.max(sx, sy);
    final taps = shrink > 1.5 ? shrink.ceil() : 1;
    final step = 1.0 / taps;
    for (var oy = 0; oy < outH; oy++) {
      for (var ox = 0; ox < outW; ox++) {
        var sum = 0.0;
        for (var ty = 0; ty < taps; ty++) {
          final v = guide.top + (oy + (ty + 0.5) * step) * sy - 0.5;
          for (var tx = 0; tx < taps; tx++) {
            final u = guide.left + (ox + (tx + 0.5) * step) * sx - 0.5;
            sum += _sampleUpright(u, v);
          }
        }
        out[oy * outW + ox] = (sum / (taps * taps)).round().clamp(0, 255);
      }
    }
    return out;
  }

  /// Bilinear sample at upright coordinates (u, v).
  double _sampleUpright(double u, double v) {
    final double x, y;
    switch (rotation) {
      case 90:
        x = v;
        y = height - 1 - u;
      case 180:
        x = width - 1 - u;
        y = height - 1 - v;
      case 270:
        x = width - 1 - v;
        y = u;
      default:
        x = u;
        y = v;
    }
    final cx = x.clamp(0.0, width - 1.0), cy = y.clamp(0.0, height - 1.0);
    final x0 = cx.floor(), y0 = cy.floor();
    final x1 = math.min(x0 + 1, width - 1), y1 = math.min(y0 + 1, height - 1);
    final fx = cx - x0, fy = cy - y0;
    final r0 = y0 * rowStride, r1 = y1 * rowStride;
    final top = bytes[r0 + x0] * (1 - fx) + bytes[r0 + x1] * fx;
    final bottom = bytes[r1 + x0] * (1 - fx) + bytes[r1 + x1] * fx;
    return top * (1 - fy) + bottom * fy;
  }
}
