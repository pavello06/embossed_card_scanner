# embossed_card_scanner

Reads **embossed** (raised) 16-digit payment card numbers on the device with a
small CNN trained on the ISO 7811 embossing font. No network, no cloud OCR:
images never leave the device.

The package has two parts:

| | Import | What it is |
|---|---|---|
| **Scanner engine** | `package:embossed_card_scanner/embossed_card_scanner_core.dart` | Headless. You give it photos or your own camera frames and get the number back. No UI. |
| **Camera widget** | `package:embossed_card_scanner/embossed_card_scanner.dart` | `CardScannerView`: a full-screen camera with a card frame, a back button and a torch button. Also exports the engine. |

Flat printed numbers (the kind on most newer cards' backs) are out of scope.

## Install

```sh
flutter pub add embossed_card_scanner
```

### iOS

Add a camera usage description to `ios/Runner/Info.plist` (required by the widget):

```xml
<key>NSCameraUsageDescription</key>
<string>The camera is used to read the card number.</string>
```

### Android

The camera permission is added by the `camera` plugin. `minSdk` 24 (Flutter's default) is fine.

`tflite_flutter` 0.12 sets different JVM targets for Java and Kotlin, which
fails the build with *Inconsistent JVM Target Compatibility*. Add this to
`android/gradle.properties`:

```properties
kotlin.jvm.target.validation.mode=warning
```

## Camera widget

A full-screen scanner with a back button at the top left, a torch at the top
right and a card frame in the middle. There is no text on the screen.

```dart
import 'package:embossed_card_scanner/embossed_card_scanner.dart';

final result = await Navigator.of(context).push<CardScanResult>(
  MaterialPageRoute(
    builder: (context) => Scaffold(
      body: CardScannerView(
        onResult: (result) => Navigator.of(context).pop(result),
      ),
    ),
  ),
);
print(result?.formatted); // 1234 5678 9012 3456
```

After `onResult` the view pauses scanning. Call `controller.resume()` to scan
another card. The camera is released while the app is in the background.

### Buttons

Both buttons can be hidden or replaced with your own:

```dart
CardScannerView(
  onResult: ...,

  // Back (top left). Pops the route by default.
  showBackButton: true,
  onBack: () => ...,                                   // instead of Navigator.maybePop
  backButtonBuilder: (context) => CloseButton(color: Colors.white),

  // Torch (top right). Hidden on devices without a torch.
  showTorchButton: true,
  initialTorch: false,                                 // true: start with the torch on
  torchButtonBuilder: (context, c) => Switch(
    value: c.torchOn,
    onChanged: c.torchAvailable ? c.setTorch : null,
  ),
);
```

The default buttons, `CardScannerBackButton` and
`CardScannerTorchButton(controller: ...)`, can be used anywhere else in your UI.

A `CardScannerController` controls the torch and scanning from code:

```dart
final controller = CardScannerController();
CardScannerView(controller: controller, onResult: ...);

controller.toggleTorch(); // or setTorch(true / false)
controller.pause();
controller.resume();
```

### Frame and camera

```dart
CardScannerView(
  onResult: ...,
  onError: (e) => ...,              // camera errors; otherwise shown in place of the preview
  engine: sharedEngine,             // reuse a loaded CardScannerEngine
  resolution: ResolutionPreset.high,
  overlayColor: Colors.black54,
  frameColor: Colors.white,
  frameRadius: 12,
  frameStrokeWidth: 2,
);
```

`CardOverlay` (the darkened cutout) is exported too, if you build your own camera screen.

## Scanner engine (headless)

```dart
import 'package:embossed_card_scanner/embossed_card_scanner_core.dart';

final engine = await CardScannerEngine.load(); // loads the bundled model
// ...
engine.dispose();
```

Inference runs with TFLite in background isolates, never on the calling
isolate.

### One photo

```dart
final result = await engine.scanFile(path);                  // or scanImage(jpegOrPngBytes)
final boxed = await engine.scanImage(bytes, cardRect: rect); // card bounding box, if known
if (result != null) print(result.number);                    // '1234567890123456'
```

Without `cardRect` the whole image is treated as the card, so the card should
fill the photo. The result is `null` when the number isn't read confidently.

### Your own frame stream

Use this when you already have a camera pipeline (your own `CameraController`,
a native plugin, a video file, and so on). Embossed digits look different
under every lighting angle, so a number is reported only after several frames
agree.

Wrap each frame in a `CardFrame`. Only brightness is used:

```dart
// The Y plane of NV21 / YUV_420_888 / any YUV format, passed as is:
CardFrame.luma(yPlane, width: w, height: h, rowStride: stride, rotation: 90);
// iOS BGRA8888:
CardFrame.bgra(bytes, width: w, height: h, rowStride: bytesPerRow, rotation: 0);
```

`rotation` is the clockwise rotation (0, 90, 180, 270) that makes the frame
upright. For a portrait camera preview it's usually the sensor orientation.

Then feed the frames to a session, one at a time:

```dart
final session = engine.startSession();

final update = await session.addFrame(frame);
if (update.result != null) print(update.result!.formatted); // confirmed number
// update.partial: current best guess, for UI feedback
// update.cardDetected: a digit row was found on this frame
// update.skipped: dropped because the previous frame is still processing
```

Or hand over a whole `Stream<CardFrame>`:

```dart
session.process(frames).listen((update) {
  if (update.result != null) print(update.result!.formatted);
});
session.results.listen(print); // only confirmed numbers
```

Frames that arrive while one is still being processed are dropped, so the
stream never builds up a backlog. After a result, the session starts over for
the next card. `session.reset()` forgets what was collected, and `session.close()` ends the session.

The engine reads the card inside a **guide rectangle**, in upright frame
coordinates. The default is `cardGuideRect(width, height)`, a centred
card-shaped box that `CardOverlay` also draws. If your UI shows a different
outline, pass it as `addFrame(frame, guide: rect)`. Users should hold the card
so that it fills that rectangle.

With the `camera` package, `cardFrameFromCameraImage(image, sensorOrientation)`
(from the full library) does the conversion for you.

## Results

`CardScanResult` contains:
- `number`: 16 digits, no spaces;
- `formatted`: `1234 5678 9012 3456`;
- `confidence`: the lowest per-digit probability, from 0 to 1;
- `readings`: how many frames or crops were combined;
- `passesLuhn`: whether the number passes the Luhn checksum.

The decoder prefers Luhn-valid numbers and reports only those, so `passesLuhn`
is always true for results from the engine.

## Limitations

- Only embossed 16-digit numbers in the standard 4-4-4-4 layout.
- The card should be roughly aligned with the guide and reasonably lit. Use the torch in dim light.
- Android and iOS only.
