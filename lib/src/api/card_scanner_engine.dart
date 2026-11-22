import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:meta/meta.dart';

import '../model/digit_map.dart';
import '../model/tflite_classifier.dart';
import '../reader/embossed_reader.dart';
import '../reader/embossed_worker.dart';
import '../reader/frame_tracker.dart';
import 'card_frame.dart';
import 'card_scan_result.dart';
import 'guide.dart';

/// Reads embossed card numbers from photos and camera frames, without UI.
///
/// ```dart
/// final engine = await CardScannerEngine.load();
/// final result = await engine.scanFile(path);           // one photo
/// final session = engine.startSession();                // or a frame stream
/// final update = await session.addFrame(CardFrame.luma(...));
/// ```
/// Inference runs with TFLite in background isolates, never on the calling
/// isolate.
class CardScannerEngine {
  /// [tflite] is the digit CNN for TFLite ([tfliteAsset]).
  CardScannerEngine.fromModels({required ByteData tflite}) : _createClassifier = _tfliteFactory(tflite);

  /// Runs [create]'s classifier instead of the bundled model.
  @visibleForTesting
  CardScannerEngine.withClassifier(ClassifierFactory create) : _createClassifier = create;

  static const _modelDir = 'packages/embossed_card_scanner/assets/models';

  /// Asset key of the bundled model.
  static const tfliteAsset = '$_modelDir/embossed_digits_cnn.tflite';

  /// Loads the bundled model from [bundle] (default [rootBundle]).
  static Future<CardScannerEngine> load({AssetBundle? bundle}) async =>
      CardScannerEngine.fromModels(tflite: await (bundle ?? rootBundle).load(tfliteAsset));

  // A static function, so the closure captures only the model bytes when it
  // is sent to an isolate.
  static ClassifierFactory _tfliteFactory(ByteData model) {
    final bytes = model.buffer.asUint8List(model.offsetInBytes, model.lengthInBytes);
    return () => TfliteDigitMapClassifier.fromBuffer(bytes);
  }

  final ClassifierFactory _createClassifier;
  Future<EmbossedWorker>? _worker;
  bool _disposed = false;

  /// The frame worker starts on first use, so photo-only users never pay for it.
  Future<EmbossedWorker> get _frameWorker {
    if (_disposed) throw StateError('CardScannerEngine is disposed');
    return _worker ??= EmbossedWorker.spawn(_createClassifier);
  }

  /// Reads the number from an encoded image (JPEG, PNG, ...). Returns null
  /// when no number is read confidently.
  ///
  /// [cardRect] is the card's bounding box in the upright image. Without it
  /// the whole image is taken as the card, so the card should fill the photo
  /// (as in a cropped photo or a capture through the guide).
  Future<CardScanResult?> scanImage(Uint8List encoded, {math.Rectangle<int>? cardRect}) async {
    if (_disposed) throw StateError('CardScannerEngine is disposed');
    final create = _createClassifier;
    final result = await Isolate.run(() => _readPhoto(create, encoded, cardRect));
    return result == null || !result.confident ? null : CardScanResult.fromEmbossed(result);
  }

  /// [scanImage] on the file at [path].
  Future<CardScanResult?> scanFile(String path, {math.Rectangle<int>? cardRect}) async =>
      scanImage(await File(path).readAsBytes(), cardRect: cardRect);

  /// Starts combining camera frames of one card; see [CardScanSession].
  CardScanSession startSession() => CardScanSession._(_frameWorker);

  /// Stops the background worker. Sessions of this engine stop working.
  void dispose() {
    _disposed = true;
    _worker?.then((w) => w.dispose()).ignore();
  }

  static EmbossedResult? _readPhoto(ClassifierFactory create, Uint8List bytes, math.Rectangle<int>? box) {
    final classifier = create();
    final decoded = img.decodeImage(bytes);
    if (decoded == null) return null;
    final image = img.bakeOrientation(decoded);
    final reader = EmbossedReader(classifier);
    final card =
        box?.intersection(math.Rectangle(0, 0, image.width, image.height)) ??
        math.Rectangle(0, 0, image.width, image.height);
    final readings = [
      for (final crop in EmbossedReader.augmentedCrops(card, image.width, image.height))
        ?reader.readSlots(EmbossedReader.normalize(image, crop: crop)),
    ];
    return EmbossedReader.decode(readings);
  }
}

/// Combines consecutive camera frames of one card into a confirmed number.
///
/// Embossed digits look different under every lighting angle, so a number
/// is reported only once several frames agree. Feed frames as they arrive;
/// while one is processed, further frames are dropped ([ScanUpdate.skipped]),
/// so there is never a backlog.
class CardScanSession {
  CardScanSession._(this._worker);

  final Future<EmbossedWorker> _worker;
  final _tracker = EmbossedFrameTracker();
  final _results = StreamController<CardScanResult>.broadcast();
  bool _busy = false;
  bool _closed = false;

  /// Every confirmed number. After a result the session starts over, ready
  /// for the next card.
  Stream<CardScanResult> get results => _results.stream;

  /// True while a frame is being processed.
  bool get isBusy => _busy;

  /// Reads the card inside [guide] (upright frame coordinates, default
  /// [cardGuideRect] of the whole frame).
  Future<ScanUpdate> addFrame(CardFrame frame, {math.Rectangle<double>? guide}) async {
    if (_closed) throw StateError('CardScanSession is closed');
    if (_busy) return ScanUpdate.dropped;
    _busy = true;
    try {
      final worker = await _worker;
      final g = guide ?? cardGuideRect(frame.uprightWidth.toDouble(), frame.uprightHeight.toDouble());
      final reading = await worker.read(frame.luma, g, _tracker.geometryHint);
      if (_closed) return ScanUpdate.dropped;
      final detected = reading != null && reading.rowScore >= EmbossedReader.minRowScore;
      final number = _tracker.add(reading);
      if (number != null) {
        final result = CardScanResult.fromEmbossed(_tracker.latest!);
        _tracker.reset();
        _results.add(result);
        return ScanUpdate(result: result, partial: number, cardDetected: true);
      }
      return ScanUpdate(partial: _tracker.latest?.digits, cardDetected: detected);
    } finally {
      _busy = false;
    }
  }

  /// Feeds a whole frame stream through [addFrame] and emits the outcome of
  /// every processed frame; dropped frames are left out. Cancelling the
  /// returned stream cancels the subscription to [frames].
  ///
  /// ```dart
  /// session.process(myFrames).listen((update) {
  ///   if (update.result != null) print(update.result!.formatted);
  /// });
  /// ```
  Stream<ScanUpdate> process(Stream<CardFrame> frames, {math.Rectangle<double>? guide}) {
    late final StreamController<ScanUpdate> out;
    StreamSubscription<CardFrame>? input;
    var pending = 0;
    var inputDone = false;
    void closeIfDone() {
      if (inputDone && pending == 0) out.close();
    }

    out = StreamController<ScanUpdate>(
      onListen: () => input = frames.listen(
        (frame) {
          if (_busy || _closed) return;
          pending++;
          addFrame(frame, guide: guide)
              .then(
                (update) {
                  if (!update.skipped && !out.isClosed) out.add(update);
                },
                onError: (Object e, StackTrace s) {
                  if (!out.isClosed) out.addError(e, s);
                },
              )
              .whenComplete(() {
                pending--;
                closeIfDone();
              });
        },
        onError: out.addError,
        onDone: () {
          inputDone = true;
          closeIfDone();
        },
      ),
      onPause: () => input?.pause(),
      onResume: () => input?.resume(),
      onCancel: () => input?.cancel(),
    );
    return out.stream;
  }

  /// Forgets the frames collected so far.
  void reset() => _tracker.reset();

  /// Ends the session: [results] completes and further frames are rejected.
  void close() {
    _closed = true;
    _results.close();
  }
}
