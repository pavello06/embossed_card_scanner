import 'dart:typed_data';

import 'package:tflite_flutter/tflite_flutter.dart';

import 'digit_map.dart';

/// The digit CNN run by TensorFlow Lite (`assets/models/embossed_digits_cnn.tflite`):
/// input [1, h, w, 1] raw gray pixels, output [1, rows, cols, 11] probabilities.
class TfliteDigitMapClassifier implements DigitMapClassifier {
  TfliteDigitMapClassifier._(this._interpreter);

  factory TfliteDigitMapClassifier.fromBuffer(Uint8List model, {int threads = 4}) {
    final options = InterpreterOptions()..threads = threads;
    return TfliteDigitMapClassifier._(Interpreter.fromBuffer(model, options: options));
  }

  final Interpreter _interpreter;
  var _shape = (0, 0);

  @override
  DigitMap classify(Uint8List pixels, int width, int height) {
    // Full searches and tracking alternate between two band sizes; resizing
    // only on change keeps re-allocation rare.
    if (_shape != (width, height)) {
      _interpreter.resizeInputTensor(0, [1, height, width, 1]);
      _interpreter.allocateTensors();
      _shape = (width, height);
    }
    final input = Float32List(pixels.length);
    for (var i = 0; i < pixels.length; i++) {
      input[i] = pixels[i].toDouble();
    }
    _interpreter.getInputTensor(0).data = input.buffer.asUint8List();
    _interpreter.invoke();
    final output = _interpreter.getOutputTensor(0);
    final shape = output.shape;
    return DigitMap(shape[1], shape[2], Float32List.fromList(Float32List.sublistView(output.data)));
  }

  void close() => _interpreter.close();
}
