import 'package:meta/meta.dart';

import '../reader/embossed_reader.dart';
import 'card_number.dart';

/// A card number read with enough confidence to be reported.
class CardScanResult {
  const CardScanResult({required this.number, required this.confidence, required this.readings});

  @internal
  CardScanResult.fromEmbossed(EmbossedResult result)
    : this(number: result.digits, confidence: result.minDigitProb, readings: result.readings);

  /// 16 digits, no spaces.
  final String number;

  /// Lowest averaged probability among the 16 digits (0..1).
  final double confidence;

  /// Number of frames (or crops of one photo) that were combined.
  final int readings;

  /// `1234 5678 9012 3456`.
  String get formatted => CardNumber.format(number);

  /// Whether [number] passes the Luhn checksum.
  bool get passesLuhn => CardNumber.passesLuhn(number);

  @override
  String toString() => 'CardScanResult($formatted, confidence ${confidence.toStringAsFixed(2)}, $readings readings)';
}

/// Outcome of one frame fed to a scan session.
class ScanUpdate {
  const ScanUpdate({this.result, this.partial, this.cardDetected = false, this.skipped = false});

  /// The frame was dropped because the previous one is still being processed.
  static const dropped = ScanUpdate(skipped: true);

  /// Set once the number is confirmed across frames.
  final CardScanResult? result;

  /// Current best guess (16 digits) while still collecting frames, for UI feedback.
  final String? partial;

  /// A digit row was found in the guide on this frame.
  final bool cardDetected;

  /// The frame was not processed (see [dropped]); other fields are empty.
  final bool skipped;
}
