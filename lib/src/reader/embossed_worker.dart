import 'dart:async';
import 'dart:isolate';
import 'dart:math' as math;

import 'embossed_reader.dart';
import 'frame_crop.dart';
import '../model/digit_map.dart';

/// A background isolate that keeps the classifier loaded and reads embossed
/// rows from camera frames, so the UI isolate never runs the heavy part and
/// the model isn't copied per frame.
class EmbossedWorker {
  EmbossedWorker._(this._isolate, this._requests, this._responses, this.backend);

  final Isolate _isolate;
  final SendPort _requests;
  final ReceivePort _responses;

  /// Which classifier the worker runs (e.g. "TfliteDigitMapClassifier").
  final String backend;
  final _pending = <int, Completer<SlotReading?>>{};
  var _nextId = 0;

  /// [create] is called in the worker isolate.
  static Future<EmbossedWorker> spawn(ClassifierFactory create) async {
    final responses = ReceivePort();
    final ready = Completer<(SendPort, String)>();
    final isolate = await Isolate.spawn(_main, (responses.sendPort, create));
    late final EmbossedWorker worker;
    responses.listen((message) {
      if (message is (SendPort, String)) {
        ready.complete(message);
      } else if (message is String) {
        // The classifier failed to load; the isolate has exited.
        ready.completeError(StateError(message));
      } else if (message is (int, SlotReading?)) {
        worker._pending.remove(message.$1)?.complete(message.$2);
      } else if (message is (int, String)) {
        worker._pending.remove(message.$1)?.completeError(StateError(message.$2));
      }
    });
    final (SendPort, String) handshake;
    try {
      handshake = await ready.future;
    } catch (_) {
      responses.close();
      rethrow;
    }
    final (requests, backend) = handshake;
    worker = EmbossedWorker._(isolate, requests, responses, backend);
    return worker;
  }

  /// Crops [guide] (upright coordinates) out of [frame] and reads the row,
  /// re-using [hint] geometry when given.
  Future<SlotReading?> read(LumaFrame frame, math.Rectangle<double> guide, SlotReading? hint) {
    final id = _nextId++;
    final completer = _pending[id] = Completer<SlotReading?>();
    _requests.send((
      id,
      TransferableTypedData.fromList([frame.bytes]),
      frame.width,
      frame.height,
      frame.rowStride,
      frame.rotation,
      guide,
      hint,
    ));
    return completer.future;
  }

  void dispose() {
    _isolate.kill(priority: Isolate.immediate);
    _responses.close();
    for (final c in _pending.values) {
      c.complete(null);
    }
    _pending.clear();
  }

  static void _main((SendPort, ClassifierFactory) args) {
    final (replies, create) = args;
    final DigitMapClassifier classifier;
    try {
      classifier = create();
    } catch (e) {
      replies.send('Failed to load the classifier: $e');
      return;
    }
    final reader = EmbossedReader(classifier);
    final requests = ReceivePort();
    replies.send((requests.sendPort, classifier.runtimeType.toString()));
    requests.listen((message) {
      final (
        int id,
        TransferableTypedData bytes,
        int width,
        int height,
        int stride,
        int rotation,
        math.Rectangle<double> guide,
        SlotReading? hint,
      ) = message as dynamic;
      try {
        final frame = LumaFrame(
          bytes: bytes.materialize().asUint8List(),
          width: width,
          height: height,
          rowStride: stride,
          rotation: rotation,
        );
        final card = frame.normalizeCard(guide);
        replies.send((
          id,
          hint == null
              ? reader.readSlots(
                  card,
                  searchTop: CardGeometry.guidedRowSearchTop,
                  searchBottom: CardGeometry.guidedRowSearchBottom,
                )
              : reader.readKnownRow(card, hint),
        ));
      } catch (e) {
        replies.send((id, '$e'));
      }
    });
  }
}
