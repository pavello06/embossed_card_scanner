## 1.0.0

- Headless engine: `scanImage` / `scanFile` for photos; `CardScanSession` (`addFrame`, `process`) for your own frame streams.
- `CardFrame.luma` / `CardFrame.bgra` for frames from any camera source.
- `CardScannerView`: full-screen camera with a card frame (no text), a back button at the top left and a torch at the top right. Both can be hidden (`showBackButton`, `showTorchButton`) or replaced (`backButtonBuilder`, `torchButtonBuilder`).
- `CardScannerController` for the torch and pausing.
- Embossed digit CNN v5 (TFLite).
