import 'dart:async';
import 'dart:io';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../api/card_scan_result.dart';
import '../api/card_scanner_engine.dart';
import 'camera_frame.dart';
import 'card_overlay.dart';

/// Builds a replacement for the torch button. Use [CardScannerController.torchOn],
/// [CardScannerController.torchAvailable] and [CardScannerController.toggleTorch].
typedef CardScanTorchButtonBuilder = Widget Function(BuildContext context, CardScannerController controller);

/// Builds a replacement for the back button.
typedef CardScanBackButtonBuilder = Widget Function(BuildContext context);

/// Controls a [CardScannerView] from outside: torch and pausing.
///
/// ```dart
/// final controller = CardScannerController();
/// CardScannerView(controller: controller, onResult: ...);
/// controller.toggleTorch();
/// controller.resume(); // scan the next card
/// ```
class CardScannerController extends ChangeNotifier {
  _CardScannerViewState? _view;
  bool _torchOn = false;
  bool _torchAvailable = true;
  bool _paused = false;

  /// Whether the torch is currently on.
  bool get torchOn => _torchOn;

  /// False once the device reported it has no torch.
  bool get torchAvailable => _torchAvailable;

  /// Scanning is paused: by [pause], or automatically after a result.
  bool get isPaused => _paused;

  /// Turns the torch on or off. Before the camera starts, the choice is
  /// applied once it does.
  Future<void> setTorch(bool on) async => _view?._setTorch(on);

  /// Flips the torch.
  Future<void> toggleTorch() => setTorch(!_torchOn);

  /// Stops reading frames; the preview keeps running.
  void pause() => _set(paused: true);

  /// Continues scanning (also after a result), starting from scratch.
  void resume() {
    _view?._session?.reset();
    _set(paused: false);
  }

  void _set({bool? torchOn, bool? torchAvailable, bool? paused}) {
    _torchOn = torchOn ?? _torchOn;
    _torchAvailable = torchAvailable ?? _torchAvailable;
    _paused = paused ?? _paused;
    notifyListeners();
  }
}

/// Full-screen camera preview that reads the embossed number of the card
/// held in the cutout: a back button at the top left, a torch button at the
/// top right and a darkened card frame in the middle, with no text.
///
/// [onResult] is called once per confirmed number; scanning then pauses
/// until [CardScannerController.resume]. The app must declare camera
/// permission (NSCameraUsageDescription on iOS).
class CardScannerView extends StatefulWidget {
  const CardScannerView({
    super.key,
    required this.onResult,
    this.onError,
    this.engine,
    this.controller,
    this.showBackButton = true,
    this.onBack,
    this.backButtonBuilder,
    this.showTorchButton = true,
    this.torchButtonBuilder,
    this.initialTorch = false,
    this.resolution = ResolutionPreset.high,
    this.overlayColor = Colors.black54,
    this.frameColor = Colors.white,
    this.frameRadius = 12,
    this.frameStrokeWidth = 2,
  });

  /// Called once per confirmed number.
  final ValueChanged<CardScanResult> onResult;

  /// Camera errors (no camera, permission denied, ...). Without it the error
  /// is shown in place of the preview.
  final ValueChanged<Object>? onError;

  /// Shared engine; by default the view loads its own and disposes it.
  final CardScannerEngine? engine;

  /// Torch and pause control; the view creates its own when omitted.
  final CardScannerController? controller;

  /// Shows the back button at the top left.
  final bool showBackButton;

  /// Called by the default back button; defaults to [Navigator.maybePop].
  final VoidCallback? onBack;

  /// Replaces the default back button (shown only if [showBackButton]).
  final CardScanBackButtonBuilder? backButtonBuilder;

  /// Shows the torch button at the top right (hidden on devices without a torch).
  final bool showTorchButton;

  /// Replaces the default torch button (shown only if [showTorchButton]).
  final CardScanTorchButtonBuilder? torchButtonBuilder;

  /// Turns the torch on as soon as the camera starts.
  final bool initialTorch;

  /// Camera preview resolution. `high` (720p) is plenty for embossed digits.
  final ResolutionPreset resolution;

  /// Color of the area around the card cutout.
  final Color overlayColor;

  /// Color of the cutout outline.
  final Color frameColor;

  /// Corner radius of the cutout.
  final double frameRadius;

  /// Width of the cutout outline; 0 hides it.
  final double frameStrokeWidth;

  @override
  State<CardScannerView> createState() => _CardScannerViewState();
}

class _CardScannerViewState extends State<CardScannerView> with WidgetsBindingObserver {
  CardScannerController? _ownController;
  CardScannerEngine? _ownEngine;
  CardScanSession? _session;
  CameraController? _camera;
  Future<void>? _starting;
  Object? _error;
  bool _disposed = false;

  CardScannerController get _controller => widget.controller ?? (_ownController ??= CardScannerController());

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _controller._view = this;
    _controller.addListener(_onControllerChanged);
    _init();
  }

  @override
  void didUpdateWidget(CardScannerView oldWidget) {
    super.didUpdateWidget(oldWidget);
    final old = oldWidget.controller ?? _ownController;
    if (old != _controller) {
      old?.removeListener(_onControllerChanged);
      old?._view = null;
      _controller._view = this;
      _controller.addListener(_onControllerChanged);
    }
  }

  Future<void> _init() async {
    try {
      final engine = widget.engine ?? (_ownEngine = await CardScannerEngine.load());
      if (_disposed) return;
      _session = engine.startSession();
      if (widget.initialTorch) _controller._set(torchOn: true);
      await _startCamera();
    } catch (e) {
      _fail(e);
    }
  }

  void _onControllerChanged() {
    if (mounted) setState(() {});
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    // Release the camera in the background, as the platforms expect.
    if (state == AppLifecycleState.inactive) {
      _stopCamera();
    } else if (state == AppLifecycleState.resumed && _session != null && _camera == null) {
      _startCamera().catchError(_fail);
    }
  }

  Future<void> _startCamera() => _starting ??= () async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) throw CameraException('none', 'No camera found');
      final back = cameras.firstWhere((c) => c.lensDirection == CameraLensDirection.back, orElse: () => cameras.first);
      final camera = CameraController(
        back,
        widget.resolution,
        enableAudio: false,
        imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
      );
      await camera.initialize();
      if (_disposed) return camera.dispose();
      await camera.lockCaptureOrientation(DeviceOrientation.portraitUp);
      setState(() => _camera = camera);
      if (_controller.torchOn) await _setTorch(true);
      await camera.startImageStream(_onFrame);
    } finally {
      _starting = null;
    }
  }();

  Future<void> _stopCamera() async {
    final camera = _camera;
    if (camera == null) return;
    final torchOn = _controller.torchOn;
    _camera = null;
    if (!_disposed) setState(() {});
    try {
      if (camera.value.isStreamingImages) await camera.stopImageStream();
      if (torchOn) await camera.setFlashMode(FlashMode.off);
    } on CameraException catch (_) {
      // The camera is being released anyway.
    }
    await camera.dispose();
  }

  /// Called for every preview frame; frames arriving while the previous one
  /// is still being processed are dropped by the session.
  void _onFrame(CameraImage image) {
    final session = _session, camera = _camera;
    if (session == null || camera == null || session.isBusy || _controller.isPaused) return;
    session
        .addFrame(cardFrameFromCameraImage(image, camera.description.sensorOrientation))
        .then(
          _onUpdate,
          // A failed frame is simply skipped; the next one retries.
          onError: (Object e) => debugPrint('CardScannerView frame failed: $e'),
        );
  }

  void _onUpdate(ScanUpdate update) {
    if (_disposed || update.skipped || _controller.isPaused) return;
    final result = update.result;
    if (result != null) {
      _controller._set(paused: true);
      widget.onResult(result);
    }
  }

  Future<void> _setTorch(bool on) async {
    final camera = _camera;
    if (camera == null) return _controller._set(torchOn: on);
    try {
      await camera.setFlashMode(on ? FlashMode.torch : FlashMode.off);
      _controller._set(torchOn: on);
    } on CameraException catch (e) {
      debugPrint('Torch unavailable: ${e.description}');
      _controller._set(torchOn: false, torchAvailable: false);
    }
  }

  void _fail(Object e) {
    if (_disposed) return;
    widget.onError?.call(e);
    setState(() => _error = e);
  }

  @override
  void dispose() {
    _disposed = true;
    WidgetsBinding.instance.removeObserver(this);
    _controller.removeListener(_onControllerChanged);
    _controller._view = null;
    _stopCamera();
    _session?.close();
    _ownEngine?.dispose();
    _ownController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final camera = _camera, controller = _controller;
    final Widget body;
    if (_error != null && widget.onError == null) {
      final e = _error;
      body = Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Text(
            e is CameraException ? e.description ?? e.code : '$e',
            textAlign: TextAlign.center,
            style: const TextStyle(color: Colors.white),
          ),
        ),
      );
    } else if (camera == null || !camera.value.isInitialized) {
      body = const Center(child: CircularProgressIndicator());
    } else {
      body = Center(
        child: CameraPreview(
          camera,
          child: CardOverlay(
            overlayColor: widget.overlayColor,
            frameColor: widget.frameColor,
            frameRadius: widget.frameRadius,
            frameStrokeWidth: widget.frameStrokeWidth,
          ),
        ),
      );
    }
    // The buttons stay above the preview while loading and on errors too, so
    // the user can always leave.
    return ColoredBox(
      color: Colors.black,
      child: Stack(
        fit: StackFit.expand,
        children: [
          body,
          SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(8),
              child: Align(
                alignment: Alignment.topCenter,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    if (widget.showBackButton)
                      widget.backButtonBuilder?.call(context) ?? CardScannerBackButton(onPressed: widget.onBack)
                    else
                      const SizedBox.shrink(),
                    if (widget.showTorchButton && controller.torchAvailable)
                      widget.torchButtonBuilder?.call(context, controller) ??
                          CardScannerTorchButton(controller: controller)
                    else
                      const SizedBox.shrink(),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// The default back button of [CardScannerView]; [onPressed] defaults to
/// [Navigator.maybePop].
class CardScannerBackButton extends StatelessWidget {
  const CardScannerBackButton({super.key, this.onPressed});

  /// Called on tap instead of popping the route.
  final VoidCallback? onPressed;

  @override
  Widget build(BuildContext context) => IconButton.filledTonal(
    key: const Key('card_scanner_back'),
    iconSize: 28,
    icon: const BackButtonIcon(),
    onPressed: onPressed ?? () => Navigator.maybePop(context),
  );
}

/// The default torch button of [CardScannerView]; also usable on its own
/// with any [CardScannerController].
class CardScannerTorchButton extends StatelessWidget {
  const CardScannerTorchButton({super.key, required this.controller});

  /// The controller whose torch the button switches.
  final CardScannerController controller;

  @override
  Widget build(BuildContext context) => ListenableBuilder(
    listenable: controller,
    builder: (context, _) => IconButton.filledTonal(
      key: const Key('card_scanner_torch'),
      iconSize: 28,
      icon: Icon(controller.torchOn ? Icons.flashlight_on : Icons.flashlight_off),
      onPressed: controller.torchAvailable ? controller.toggleTorch : null,
    ),
  );
}
