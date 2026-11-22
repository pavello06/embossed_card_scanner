import 'package:embossed_card_scanner/embossed_card_scanner.dart';
import 'package:flutter/material.dart';

void main() => runApp(const ExampleApp());

class ExampleApp extends StatelessWidget {
  const ExampleApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Embossed card scanner',
    theme: ThemeData(colorScheme: ColorScheme.fromSeed(seedColor: Colors.teal)),
    home: const HomePage(),
  );
}

/// A scan button and the scanned number below it.
class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final _number = TextEditingController();

  @override
  void dispose() {
    _number.dispose();
    super.dispose();
  }

  Future<void> _scan() async {
    final result = await Navigator.of(
      context,
    ).push<CardScanResult>(MaterialPageRoute(builder: (_) => const ScanPage()));
    if (result != null) _number.text = result.formatted;
  }

  @override
  Widget build(BuildContext context) => Scaffold(
    appBar: AppBar(title: const Text('Embossed card scanner')),
    body: Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          FilledButton.icon(
            key: const Key('scan'),
            onPressed: _scan,
            icon: const Icon(Icons.credit_card),
            label: const Text('Scan card'),
          ),
          const SizedBox(height: 24),
          TextField(
            key: const Key('number'),
            controller: _number,
            readOnly: true,
            style: const TextStyle(fontFamily: 'monospace', fontSize: 20),
            decoration: const InputDecoration(
              labelText: 'Card number',
              border: OutlineInputBorder(),
            ),
          ),
        ],
      ),
    ),
  );
}

/// Full-screen scanner: back button, torch and card frame are built in.
/// Pops with the result once a number is confirmed.
class ScanPage extends StatelessWidget {
  const ScanPage({super.key});

  @override
  Widget build(BuildContext context) => Scaffold(
    backgroundColor: Colors.black,
    body: CardScannerView(
      onResult: (result) => Navigator.of(context).pop(result),
    ),
  );
}
