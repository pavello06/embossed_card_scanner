/// Helpers for card number strings (digits only, no spaces).
class CardNumber {
  /// Formats digits in groups of four: `1234 5678 9012 3456`.
  static String format(String digits) => digits.replaceAllMapped(RegExp(r'.{4}'), (m) => '${m[0]} ').trim();

  /// Whether [digits] pass the Luhn checksum used by payment cards.
  static bool passesLuhn(String digits) {
    var sum = 0;
    var doubleIt = false;
    for (var i = digits.length - 1; i >= 0; i--) {
      var d = int.parse(digits[i]);
      if (doubleIt) {
        d *= 2;
        if (d > 9) d -= 9;
      }
      sum += d;
      doubleIt = !doubleIt;
    }
    return sum % 10 == 0;
  }
}
