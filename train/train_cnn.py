"""Trains the CNN of cnn.py on synth_dense crops (TensorFlow venv, see README).

    python train_cnn.py [n_train=48000] [epochs=40] [init.keras|-] [seeds=1000]

`seeds` is a comma-separated list; each seed adds n_train crops.

Writes model/embossed_digits.keras (+ .temperature) and exports it
(export_cnn.py) to model/embossed_digits.cnn and model/embossed_digits_cnn.tflite.

Synthetic data is cached as data_<n>_<seed>.npz next to this script.
"""
import os
import sys
import time

import numpy as np

import cnn
import export_cnn
import synth_dense

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = os.path.join(HERE, 'model', 'embossed_digits')

SMOOTH = 0.05
BG_WEIGHT = 0.25  # background outnumbers digit positions ~50:1


def data(n, seed0):
    path = os.path.join(HERE, f'data_{n}_{seed0}.npz')
    if os.path.exists(path):
        z = np.load(path)
        return z['x'], z['y']
    t = time.time()
    x, y = synth_dense.build(n, seed0)
    np.savez(path, x=x, y=y)
    print('generated', x.shape, round(time.time() - t), 's', flush=True)
    return x, y


def loss(y, logits):
    import tensorflow as tf
    y = tf.cast(y, tf.int32)
    valid = y >= 0
    target = tf.one_hot(tf.maximum(y, 0), cnn.CLASSES) * (1 - SMOOTH) + SMOOTH / cnn.CLASSES
    ce = tf.nn.softmax_cross_entropy_with_logits(target, logits)
    w = tf.where(y == cnn.CLASSES - 1, BG_WEIGHT, 1.0) * tf.cast(valid, tf.float32)
    return tf.reduce_sum(ce * w) / tf.maximum(tf.reduce_sum(w), 1.0)


def report(probs, y):
    """Accuracy on digit and background positions, and frequent digit confusions."""
    pred = probs.argmax(-1)
    d, bg = (y >= 0) & (y < 10), y == 10
    conf = np.zeros((11, 11), int)
    np.add.at(conf, (y[d | bg], pred[d | bg]), 1)
    pairs = sorted(((conf[a, b], f'{a}->{b}') for a in range(10) for b in range(11) if a != b), reverse=True)[:6]
    false_digit = (pred[bg] < 10).mean()
    return f'digit acc {(pred[d] == y[d]).mean():.4f} bg acc {1 - false_digit:.4f} top confusions {pairs}'


def temperature(logits, y):
    """Temperature minimizing NLL on validation digit/background positions."""
    m = y >= 0
    z, t = logits[m].astype(np.float64), y[m].astype(int)
    best = (np.inf, 1.0)
    for T in np.arange(0.1, 3.01, 0.05):
        s = z / T
        s = s - s.max(-1, keepdims=True)
        nll = -(s[np.arange(len(t)), t] - np.log(np.exp(s).sum(-1))).mean()
        best = min(best, (nll, float(T)))
    return best[1]


if __name__ == '__main__':
    import tensorflow as tf
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 48000
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    init = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '-' else None
    parts = [data(n, int(seed)) for seed in (sys.argv[4] if len(sys.argv) > 4 else '1000').split(',')]
    x, y = np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    del parts
    xv, yv = data(4000, 1)
    os.makedirs(os.path.dirname(PREFIX), exist_ok=True)
    out = PREFIX + '.keras'
    tf.keras.utils.set_random_seed(0)
    model = cnn.load_keras(init) if init else cnn.build_keras(train=True)
    batch = 64
    steps = epochs * (len(x) // batch)
    opt = tf.keras.optimizers.AdamW(tf.keras.optimizers.schedules.CosineDecay(0.0, steps, warmup_target=1e-3 if init else 3e-3,
                                                                               warmup_steps=steps // 40), weight_decay=1e-4)
    model.compile(opt, loss)
    ds = tf.data.Dataset.from_tensor_slices((x[..., None], y)).shuffle(8192, seed=0).batch(batch).prefetch(4)

    class Report(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if epoch % 5 == 4 or epoch == epochs - 1:
                logits = self.model.predict(xv[..., None], batch_size=256, verbose=0)
                print(f'epoch {epoch + 1} loss {logs["loss"]:.4f}', report(logits, yv), flush=True)
                self.model.save(out)

    t = time.time()
    model.fit(ds, epochs=epochs, verbose=2, callbacks=[Report()])
    logits = model.predict(xv[..., None], batch_size=256, verbose=0)
    T = temperature(logits, yv)
    print('fit', round(time.time() - t), 's; temperature', T, flush=True)
    model.save(out)
    with open(out + '.temperature', 'w') as f:
        f.write(str(T))
    export_cnn.export(model, T, PREFIX)
