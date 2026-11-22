import 'embossed_reader.dart';

/// Combines embossed readings from consecutive camera frames.
///
/// Embossed digits look different under every lighting angle, so averaging
/// several frames is much more reliable than any single one. A blurred or
/// misaligned frame is skipped rather than discarding what was collected:
/// the row is re-searched in full only after repeated misses, and the
/// accumulated readings are dropped only when the card is clearly gone.
class EmbossedFrameTracker {
  static const window = 8;
  static const stableDecodes = 2;

  /// Misses after which the next frame gets a full row search.
  static const relocateAfterMisses = 2;

  /// Misses after which collected readings are discarded (card removed).
  static const resetAfterMisses = 4;

  final _readings = <SlotReading>[];
  SlotReading? _geometry;
  String? _lastDigits;
  int _stable = 0;
  int _misses = 0;

  /// Geometry to re-use for the next frame, or null when a full search is due.
  SlotReading? get geometryHint => _misses >= relocateAfterMisses ? null : _geometry;

  /// Latest decode of the accumulated frames, for UI feedback.
  EmbossedResult? latest;

  /// Adds one frame's reading. Returns the number once it is confident and
  /// the same in [stableDecodes] consecutive decodes.
  String? add(SlotReading? reading) {
    if (reading == null || reading.rowScore < EmbossedReader.minRowScore) {
      _misses++;
      if (_misses >= resetAfterMisses) reset();
      return null;
    }
    _misses = 0;
    _geometry = reading;
    _readings.add(reading);
    if (_readings.length > window) _readings.removeAt(0);
    final result = latest = EmbossedReader.decode(_readings);
    if (result == null || !result.confidentForStream) {
      _stable = 0;
      return null;
    }
    _stable = result.digits == _lastDigits ? _stable + 1 : 1;
    _lastDigits = result.digits;
    return _stable >= stableDecodes ? result.digits : null;
  }

  void reset() {
    _readings.clear();
    _geometry = null;
    _lastDigits = null;
    _stable = 0;
    _misses = 0;
    latest = null;
  }
}
