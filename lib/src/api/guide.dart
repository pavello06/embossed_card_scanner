import 'dart:math' as math;

/// ISO/IEC 7810 ID-1 card proportions (85.60 x 53.98 mm).
const cardAspectRatio = 85.6 / 53.98;

/// Card outline the user aligns the card with, as a rectangle inside an
/// upright image (or preview) of [width] x [height]. The widget's overlay and
/// the frame crop both use it, so what the user sees is what gets read.
math.Rectangle<double> cardGuideRect(double width, double height) {
  final w = math.min(width * 0.86, height * 0.7 * cardAspectRatio);
  final h = w / cardAspectRatio;
  return math.Rectangle((width - w) / 2, (height - h) / 2, w, h);
}
