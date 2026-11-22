import 'dart:typed_data';

/// Class probabilities on the 2 px output grid of the digit CNN
/// (bundled in assets/models). Cell (v, u) of an image whose top-left pixel lies at
/// (ax, ay) is the window centred at (ax + [centreX] + 2u, ay + [centreY] + 2v).
class DigitMap {
  DigitMap(this.rows, this.cols, this.probs);

  static const classes = 11;

  /// Class "not centred on a digit"; 0-9 are the digits.
  static const background = 10;
  static const centreX = 13;
  static const centreY = 17;

  /// Receptive field: the image must be at least this large.
  static const windowWidth = 29;
  static const windowHeight = 37;

  final int rows;
  final int cols;

  /// rows x cols x [classes], row-major.
  final Float32List probs;

  Float32List at(int v, int u) =>
      Float32List.sublistView(probs, (v * cols + u) * classes, (v * cols + u + 1) * classes);
}

/// Computes a [DigitMap] for a gray image.
abstract interface class DigitMapClassifier {
  DigitMap classify(Uint8List pixels, int width, int height);
}

/// Creates the classifier inside the isolate that runs it.
typedef ClassifierFactory = DigitMapClassifier Function();
