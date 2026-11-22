import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as img;

import '../api/card_number.dart';
import '../model/digit_map.dart';

/// Card geometry in the normalized frame the classifier was trained on.
/// ISO/IEC 7810 ID-1 card, ISO/IEC 7811 embossing: 3.63 mm character pitch.
class CardGeometry {
  static const pxPerMm = 7.0;
  static final width = (85.6 * pxPerMm).round(); // 599
  static final height = (53.98 * pxPerMm).round(); // 378
  static const pitch = 3.63 * pxPerMm;

  /// Vertical band searched for the number row, as a fraction of height.
  static const rowSearchTop = 0.40;
  static const rowSearchBottom = 0.80;

  /// Narrower band for camera frames, where the card is held in the guide.
  static const guidedRowSearchTop = 0.47;
  static const guidedRowSearchBottom = 0.75;

  /// Slot index of each of the 16 digits: 4 groups, one empty pitch between.
  static const slots = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18];
}

/// Probabilities of the 16 digit positions from one image, plus geometry.
class SlotReading {
  const SlotReading({
    required this.slotProbs,
    required this.rowScore,
    required this.y,
    required this.x0,
    required this.pitch,
  });

  /// 16 x 11 class probabilities, row-major.
  final Float32List slotProbs;

  /// Sum over slots of log P(not background); near 0 means a clear digit row.
  final double rowScore;
  final double y;
  final double x0;
  final double pitch;
}

/// Decoded number with confidence details.
class EmbossedResult {
  const EmbossedResult({
    required this.digits,
    required this.logProb,
    required this.bestLogProb,
    required this.rowScore,
    required this.minDigitProb,
    required this.readings,
  });

  final String digits;

  /// Lowest averaged probability among the 16 chosen digits: one doubtful
  /// digit is enough for Luhn to "fix" the number into a wrong one.
  final double minDigitProb;

  /// Number of frames/crops averaged.
  final int readings;

  /// Log-probability of [digits]; [bestLogProb] is that of the unconstrained argmax.
  final double logProb;
  final double bestLogProb;
  final double rowScore;

  /// Luhn-valid number close to the most likely reading on a clear digit row,
  /// with no very doubtful digit. Used for a single photo, which gets no
  /// second chance: a doubtful photo falls back to text recognition.
  bool get confident =>
      rowScore > EmbossedReader.minRowScore &&
      bestLogProb - logProb < EmbossedReader.maxLuhnPenalty &&
      minDigitProb >= EmbossedReader.minPhotoDigitProb;

  /// Stricter rule for camera streams, which can afford to wait for better
  /// frames: additionally no doubtful digit. Without it, simulated hand-held
  /// streams of a hard card produced wrong Luhn-valid numbers.
  bool get confidentForStream => confident && minDigitProb >= EmbossedReader.minDigitProb;
}

/// Reads 16-digit numbers embossed in the ISO 7811 font (Farrington 7B style).
class EmbossedReader {
  EmbossedReader(this.classifier);

  final DigitMapClassifier classifier;

  static const minRowScore = -12.0;

  /// Camera streams: every chosen digit needs at least this averaged
  /// probability (see [EmbossedResult.confidentForStream]).
  static const minDigitProb = 0.4;

  /// Single photos: a lower bar than [minDigitProb]. In the simulated camera
  /// streams, every wrong Luhn-valid decode had a digit below 0.23.
  static const minPhotoDigitProb = 0.3;

  static const maxLuhnPenalty = 2.5;
  static const _step = 2;
  static const _halfW = DigitMap.windowWidth ~/ 2;

  /// Crops [source] to [crop] (or the whole image) and converts it to the
  /// normalized gray card frame.
  static Uint8List normalize(img.Image source, {math.Rectangle<int>? crop}) {
    final c = crop ?? math.Rectangle(0, 0, source.width, source.height);
    final cropped = img.copyCrop(source, x: c.left, y: c.top, width: c.width, height: c.height);
    final shrink = CardGeometry.width / cropped.width;
    final resized = img.copyResize(
      cropped,
      width: CardGeometry.width,
      height: CardGeometry.height,
      interpolation: shrink < 0.75 ? img.Interpolation.average : img.Interpolation.linear,
    );
    final gray = Uint8List(CardGeometry.width * CardGeometry.height);
    var i = 0;
    for (final p in resized) {
      // Same weights as PIL's "L" conversion used in training.
      gray[i++] = ((p.r * 299 + p.g * 587 + p.b * 114) / 1000).round().clamp(0, 255);
    }
    return gray;
  }

  /// The card box plus slightly zoomed and shifted copies. Averaging their
  /// readings (test-time augmentation) stands in for multiple video frames.
  static List<math.Rectangle<int>> augmentedCrops(math.Rectangle<int> box, int imageWidth, int imageHeight) {
    const variants = [(0.0, 0.0), (0.0, 0.03), (0.0, -0.03), (0.02, 0.0), (-0.02, 0.0)]; // (shift x, zoom)
    return [
      for (final (dx, zoom) in variants)
        () {
          final cx = box.left + box.width / 2 + dx * box.width, cy = box.top + box.height / 2;
          final hw = box.width * (1 + zoom) / 2, hh = box.height * (1 + zoom) / 2;
          final l = math.max(cx - hw, 0).round(), t = math.max(cy - hh, 0).round();
          final r = math.min(cx + hw, imageWidth).round(), b = math.min(cy + hh, imageHeight).round();
          return math.Rectangle(l, t, r - l, b - t);
        }(),
    ];
  }

  /// Finds the number row on a normalized card and scores its 16 slots.
  ///
  /// [searchTop] and [searchBottom] bound the row centre as a fraction of
  /// the card height; the camera guide allows a narrower band than a photo.
  SlotReading? readSlots(
    Uint8List card, {
    double searchTop = CardGeometry.rowSearchTop,
    double searchBottom = CardGeometry.rowSearchBottom,
  }) {
    final h = CardGeometry.height;
    final top = (h * searchTop / _step).ceil() * _step, bottom = (h * searchBottom / _step).floor() * _step;
    final rows = _rows(card, top, bottom);

    // Coarse pass: full grid search on every other row.
    final candidates = [
      for (final MapEntry(key: y, value: row) in rows.entries)
        if ((y - top) % (2 * _step) == 0)
          if (_bestGrid(row, _step) case (final score, final x0, final pitch))
            SlotReading(slotProbs: Float32List(0), rowScore: score, y: y.toDouble(), x0: x0, pitch: pitch),
    ]..sort((a, b) => b.rowScore.compareTo(a.rowScore));

    // Refine the two best rows and their neighbours.
    SlotReading? best;
    for (final geometry in candidates.take(2)) {
      for (var y = geometry.y.round() - _step; y <= geometry.y.round() + _step; y += _step) {
        final row = rows[y];
        if (row == null) continue;
        final r = _fit(row, y, geometry);
        if (best == null || r.rowScore > best.rowScore) best = r;
      }
    }
    return best;
  }

  static const _trackY = 6;
  static const _trackX = 8;

  /// Re-scores a row found in an earlier frame (fast path for video): only
  /// a few pixels around [geometry] are searched, on a narrow band.
  SlotReading? readKnownRow(Uint8List card, SlotReading geometry) {
    final yGuess = (geometry.y / _step).round() * _step;
    SlotReading? best;
    for (final MapEntry(key: y, value: row) in _rows(card, yGuess - _trackY, yGuess + _trackY).entries) {
      final r = _fit(row, y, geometry);
      if (best == null || r.rowScore > best.rowScore) best = r;
    }
    return best;
  }

  /// Grid fit near [geometry] on one row, with the 16 slot probabilities.
  SlotReading _fit(List<Float32List?> row, int y, SlotReading geometry) {
    final (score, x0, pitch) = _bestGrid(row, _step, near: (geometry.x0, geometry.pitch));
    return SlotReading(
      slotProbs: _slotProbs(row, x0, pitch, _step),
      rowScore: score,
      y: y.toDouble(),
      x0: x0,
      pitch: pitch,
    );
  }

  /// Class probabilities of windows centred on rows y = [yLo]..[yHi] (even),
  /// in one classifier run over the band they cover. Each row maps centre
  /// x / [_step] to probabilities, null where the window leaves the card.
  Map<int, List<Float32List?>> _rows(Uint8List card, int yLo, int yHi) {
    final w = CardGeometry.width, h = CardGeometry.height;
    // Odd origins put the window centres on even pixels.
    const ax = 1;
    final ay = math.max(yLo - DigitMap.centreY, 1) | 1;
    final bottom = math.min(h, yHi + DigitMap.windowHeight - DigitMap.centreY);
    final bw = w - ax, bh = bottom - ay;
    if (bh < DigitMap.windowHeight) return const {};
    final band = Uint8List(bw * bh);
    for (var y = 0; y < bh; y++) {
      band.setRange(y * bw, (y + 1) * bw, card, (ay + y) * w + ax);
    }
    final map = classifier.classify(band, bw, bh);
    return {
      for (var v = 0; v < map.rows; v++)
        if (ay + DigitMap.centreY + _step * v case final y when y >= yLo && y <= yHi)
          y: List<Float32List?>.generate(w ~/ _step + 1, (i) {
            final u = (i * _step - ax - DigitMap.centreX) ~/ _step;
            return i * _step - ax - DigitMap.centreX >= 0 && u < map.cols ? map.at(v, u) : null;
          }),
    };
  }

  /// Best (score, x0, pitch) of the 16-slot grid over a row of probabilities.
  static (double, double, double) _bestGrid(List<Float32List?> row, int xStep, {(double, double)? near}) {
    final logDigit = Float64List(row.length);
    for (var i = 0; i < row.length; i++) {
      final p = row[i];
      logDigit[i] = math.log(p == null ? 1e-4 : math.max(1 - p[DigitMap.background], 1e-4));
    }
    var best = (-1e9, 0.0, 0.0);
    final minPitch = near == null ? CardGeometry.pitch * 0.85 : near.$2 - 0.5;
    final maxPitch = near == null ? CardGeometry.pitch * 1.15 : near.$2 + 0.5;
    for (var pitch = minPitch; pitch <= maxPitch + 1e-9; pitch += 0.25) {
      final span = pitch * CardGeometry.slots.last;
      final lo = near == null ? _halfW.toDouble() : near.$1 - _trackX;
      final hi = near == null ? CardGeometry.width - _halfW - span : near.$1 + _trackX;
      for (var x0 = lo; x0 <= hi; x0 += 1) {
        var s = 0.0;
        for (final slot in CardGeometry.slots) {
          final i = ((x0 + pitch * slot) / xStep).round();
          s += i >= 0 && i < logDigit.length ? logDigit[i] : math.log(1e-4);
        }
        if (s > best.$1) best = (s, x0, pitch);
      }
    }
    return best;
  }

  static Float32List _slotProbs(List<Float32List?> row, double x0, double pitch, int xStep) {
    final out = Float32List(16 * DigitMap.classes);
    for (var k = 0; k < 16; k++) {
      final i = ((x0 + pitch * CardGeometry.slots[k]) / xStep).round();
      final p = i >= 0 && i < row.length ? row[i] : null;
      if (p == null) {
        out[k * DigitMap.classes + DigitMap.background] = 1;
      } else {
        out.setRange(k * DigitMap.classes, (k + 1) * DigitMap.classes, p);
      }
    }
    return out;
  }

  /// Decodes averaged slot probabilities, preferring a Luhn-valid number.
  static EmbossedResult? decode(List<SlotReading> readings) {
    if (readings.isEmpty) return null;
    const c = DigitMap.classes;
    final logp = List.generate(16, (_) => Float64List(10));
    for (var k = 0; k < 16; k++) {
      final avg = Float64List(10);
      for (final r in readings) {
        var digitMass = 0.0;
        for (var d = 0; d < 10; d++) {
          digitMass += r.slotProbs[k * c + d];
        }
        for (var d = 0; d < 10; d++) {
          avg[d] += r.slotProbs[k * c + d] / math.max(digitMass, 1e-9) / readings.length;
        }
      }
      for (var d = 0; d < 10; d++) {
        logp[k][d] = math.log(math.max(avg[d], 1e-6));
      }
    }
    final rowScore = readings.map((r) => r.rowScore).reduce((a, b) => a + b) / readings.length;

    // Beam search over the 4 most likely digits per slot.
    var beams = <(double, String)>[(0.0, '')];
    for (var k = 0; k < 16; k++) {
      final order = List.generate(10, (d) => d)..sort((a, b) => logp[k][b].compareTo(logp[k][a]));
      final next = <(double, String)>[
        for (final (s, ds) in beams)
          for (final d in order.take(4)) (s + logp[k][d], '$ds$d'),
      ]..sort((a, b) => b.$1.compareTo(a.$1));
      beams = next.take(300).toList();
    }
    final bestLogProb = beams.first.$1;
    final valid = beams.where((b) => CardNumber.passesLuhn(b.$2)).firstOrNull;
    final chosen = valid ?? beams.first;
    var minDigitProb = 1.0;
    for (var k = 0; k < 16; k++) {
      minDigitProb = math.min(minDigitProb, math.exp(logp[k][chosen.$2.codeUnitAt(k) - 48]));
    }
    return EmbossedResult(
      digits: chosen.$2,
      logProb: valid == null ? double.negativeInfinity : chosen.$1,
      bestLogProb: bestLogProb,
      rowScore: rowScore,
      minDigitProb: minDigitProb,
      readings: readings.length,
    );
  }
}
