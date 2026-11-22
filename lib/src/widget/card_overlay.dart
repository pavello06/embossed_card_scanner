import 'package:material_ui/material_ui.dart';

import '../api/guide.dart';

/// Darkens everything except a card-shaped cutout ([cardGuideRect]) and
/// outlines the cutout.
class CardOverlay extends StatelessWidget {
  const CardOverlay({
    super.key,
    this.overlayColor = Colors.black54,
    this.frameColor = Colors.white,
    this.frameRadius = 12,
    this.frameStrokeWidth = 2,
    this.child,
  });

  /// Color of the area around the cutout.
  final Color overlayColor;

  /// Color of the cutout outline.
  final Color frameColor;

  /// Corner radius of the cutout.
  final double frameRadius;

  /// Width of the cutout outline; 0 hides it.
  final double frameStrokeWidth;

  /// Painted above the overlay; fills the available space by default.
  final Widget? child;

  @override
  Widget build(BuildContext context) => CustomPaint(
    painter: CardOverlayPainter(
      overlayColor: overlayColor,
      frameColor: frameColor,
      frameRadius: frameRadius,
      frameStrokeWidth: frameStrokeWidth,
    ),
    child: child ?? const SizedBox.expand(),
  );
}

/// Paints [CardOverlay]; usable directly in a [CustomPaint].
class CardOverlayPainter extends CustomPainter {
  const CardOverlayPainter({
    required this.overlayColor,
    required this.frameColor,
    required this.frameRadius,
    required this.frameStrokeWidth,
  });

  final Color overlayColor;
  final Color frameColor;
  final double frameRadius;
  final double frameStrokeWidth;

  @override
  void paint(Canvas canvas, Size size) {
    final g = cardGuideRect(size.width, size.height);
    final card = RRect.fromRectAndRadius(Rect.fromLTWH(g.left, g.top, g.width, g.height), Radius.circular(frameRadius));
    canvas.drawDRRect(RRect.fromRectAndRadius(Offset.zero & size, Radius.zero), card, Paint()..color = overlayColor);
    if (frameStrokeWidth > 0) {
      canvas.drawRRect(
        card,
        Paint()
          ..color = frameColor
          ..style = PaintingStyle.stroke
          ..strokeWidth = frameStrokeWidth,
      );
    }
  }

  @override
  bool shouldRepaint(CardOverlayPainter oldDelegate) =>
      overlayColor != oldDelegate.overlayColor ||
      frameColor != oldDelegate.frameColor ||
      frameRadius != oldDelegate.frameRadius ||
      frameStrokeWidth != oldDelegate.frameStrokeWidth;
}
