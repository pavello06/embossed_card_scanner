# Embossed digit classifier

Trains a CNN that reads embossed (unpainted, raised) card numbers and writes it
to `model/`:

- `model/embossed_digits_cnn.tflite`: TensorFlow Lite model;
- `model/embossed_digits.cnn`: the same weights in the plain `CNN1` binary format (see `cnn.py`);
- `model/embossed_digits.keras` (+ `.temperature`): the training checkpoint, git-ignored.

Training data is fully synthetic:

1. `glyphs.py` renders the ISO 7811 embossing digits from the
   [Gurney-7B](https://github.com/scruss/Gurney-7B) outlines (WTFPL).
2. `synth.py` turns rows of digits into relief, lit from a random direction
   (highlights, shadows, optional paint on top), on random card-like
   backgrounds, then adds warp, blur, noise and JPEG artefacts. The geometry is
   a normalized card at 7 px/mm with a 3.63 mm character pitch.
3. `synth_dense.py` cuts 80 x 160 px crops of longer, grouped rows (plus
   smaller embossed text and glare) and labels every point of the CNN's 2 px
   output grid: digit within 2 px of a glyph centre, background well away from
   glyphs, ignored in between.
4. `cnn.py` defines the fully convolutional network: raw gray pixels in,
   probabilities of 11 classes (digits 0–9 plus "not centred on a digit") on a
   2 px grid out, receptive field 37 x 29 px. It also holds the numpy forward
   pass and the `CNN1` file format.
5. `train_cnn.py` trains it (TensorFlow), calibrates a softmax temperature and
   exports the result.
6. `export_cnn.py` folds batch norm, input scaling and temperature into the
   convolutions, writes the `.cnn` and the `.tflite` and checks that both match
   the trained model.

TensorFlow needs Python <= 3.12:

```sh
./fetch_font.sh
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-cnn.txt
.venv/bin/python train_cnn.py 32000 30           # ~5 min data, ~1 h training on an M1 Pro
.venv/bin/python export_cnn.py model/embossed_digits.keras   # re-export only
```

`train_cnn.py [n_train] [epochs] [init.keras|-] [seeds]`: `init` fine-tunes an
existing checkpoint, `seeds` (comma-separated) adds `n_train` crops per seed.
Generated data is cached as `data_<n>_<seed>.npz`; `python synth_dense.py out.png`
previews a few labelled crops.
