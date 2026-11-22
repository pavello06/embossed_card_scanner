"""Fully convolutional digit-map model.

The network slides over a band of the normalized card (raw gray pixels) and
outputs, on a 2 px grid, probabilities of 11 classes: digits 0-9 and
"not centred on a digit". Convolutions are 'valid' with a total stride of 4;
the last layer emits 4 x 11 logits per cell (2 x 2 sub-positions) that
depth_to_space turns into the 2 px grid.

For an input whose top-left pixel is at (ax, ay), output cell (v, u) is
centred at (ax + CENTRE_X + 2u, ay + CENTRE_Y + 2v).
"""
import struct
import numpy as np

# (kernel h, kernel w, stride, out channels); ReLU after all but the last.
LAYERS = [(3, 3, 1, 12), (3, 3, 2, 24), (3, 3, 1, 24), (3, 3, 2, 48),
          (3, 3, 1, 48), (3, 3, 1, 64), (3, 1, 1, 96), (1, 1, 1, 44)]
CLASSES = 11
BLOCK = 2                  # depth_to_space block: output grid step = 4 / BLOCK = 2 px
RF_H, RF_W = 37, 29        # receptive field
CENTRE_Y = RF_H // 2 - 1   # centre of sub-position 0 (sub-position 1 is 2 px further)
CENTRE_X = RF_W // 2 - 1


def out_cells(n, axis):
    """Number of stride-4 cells for an input of n pixels along axis 0 (h) or 1 (w)."""
    for kh, kw, s, _ in LAYERS:
        k = (kh, kw)[axis]
        n = (n - k) // s + 1
    return n


def _depth_to_space():
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable('card')
    class DepthToSpace(tf.keras.layers.Layer):
        def call(self, x):
            return tf.nn.depth_to_space(x, BLOCK)

    return DepthToSpace()


def build_keras(train=True):
    import tensorflow as tf
    L = tf.keras.layers
    x = inp = L.Input((None, None, 1), name='pixels')
    if train:
        x = L.Rescaling(1 / 255, offset=-0.5)(x)
    for i, (kh, kw, s, c) in enumerate(LAYERS):
        last = i == len(LAYERS) - 1
        x = L.Conv2D(c, (kh, kw), strides=s, padding='valid', use_bias=not train or last, name=f'conv{i}')(x)
        if not last:
            if train:
                x = L.BatchNormalization(name=f'bn{i}')(x)
            x = L.ReLU()(x)
    x = _depth_to_space()(x)
    if not train:
        x = L.Softmax()(x)
    return tf.keras.Model(inp, x)


def load_keras(path):
    import tensorflow as tf
    _depth_to_space()  # registers the layer class
    return tf.keras.models.load_model(path, compile=False)


def fold(model, temperature=1.0):
    """Numpy layers [(W[kh,kw,cin,cout], b, stride, relu)] with BN, input scaling and temperature folded in."""
    out = []
    for i, (kh, kw, s, c) in enumerate(LAYERS):
        conv = model.get_layer(f'conv{i}')
        w = conv.get_weights()
        W, b = w[0].astype(np.float64), (w[1].astype(np.float64) if len(w) > 1 else np.zeros(c))
        if i < len(LAYERS) - 1:
            gamma, beta, mean, var = (a.astype(np.float64) for a in model.get_layer(f'bn{i}').get_weights())
            k = gamma / np.sqrt(var + model.get_layer(f'bn{i}').epsilon)
            W, b = W * k, (b - mean) * k + beta
        else:
            W, b = W / temperature, b / temperature
        if i == 0:  # input was x / 255 - 0.5
            b = b - 0.5 * W.sum((0, 1, 2))
            W = W / 255
        out.append((W.astype(np.float32), b.astype(np.float32), s, i < len(LAYERS) - 1))
    return out


def conv(x, W, b, s):
    """Valid conv of x[h, w, cin] (float32) by W[kh, kw, cin, cout]."""
    kh, kw, cin, cout = W.shape
    h, w = (x.shape[0] - kh) // s + 1, (x.shape[1] - kw) // s + 1
    out = np.broadcast_to(b, (h, w, cout)).astype(np.float32)
    for dy in range(kh):
        for dx in range(kw):
            patch = x[dy:dy + s * (h - 1) + 1:s, dx:dx + s * (w - 1) + 1:s]
            out = out + patch @ W[dy, dx]
    return out


def forward(layers, pixels):
    """Probabilities [2 * cells_h, 2 * cells_w, 11] for a uint8 image."""
    x = pixels.astype(np.float32)[..., None]
    for W, b, s, relu in layers:
        x = conv(x, W, b, s)
        if relu:
            x = np.maximum(x, 0)
    h, w, _ = x.shape
    x = x.reshape(h, w, BLOCK, BLOCK, CLASSES).transpose(0, 2, 1, 3, 4).reshape(h * BLOCK, w * BLOCK, CLASSES)
    x = np.exp(x - x.max(-1, keepdims=True))
    return x / x.sum(-1, keepdims=True)


def write(layers, path):
    """Binary CNN1 format (little endian): magic 'CNN1',
    int32 layer count, centre x, centre y, block, classes; then per layer int32
    kh, kw, stride, cin, cout, relu, float32 W[kh][kw][cin][cout], float32 b[cout]."""
    with open(path, 'wb') as f:
        f.write(b'CNN1')
        f.write(struct.pack('<5i', len(layers), CENTRE_X, CENTRE_Y, BLOCK, CLASSES))
        for W, b, s, relu in layers:
            f.write(struct.pack('<6i', *W.shape[:2], s, *W.shape[2:], int(relu)))
            f.write(W.astype('<f4').tobytes(order='C')); f.write(b.astype('<f4').tobytes())


def read(path):
    buf = open(path, 'rb').read()
    assert buf[:4] == b'CNN1', 'not a CNN1 model'
    n, cx, cy, block, classes = struct.unpack_from('<5i', buf, 4)
    assert (cx, cy, block, classes) == (CENTRE_X, CENTRE_Y, BLOCK, CLASSES)
    off, layers = 24, []
    for _ in range(n):
        kh, kw, s, cin, cout, relu = struct.unpack_from('<6i', buf, off); off += 24
        W = np.frombuffer(buf, '<f4', kh * kw * cin * cout, off).reshape(kh, kw, cin, cout); off += W.nbytes
        b = np.frombuffer(buf, '<f4', cout, off); off += b.nbytes
        layers.append((W, b, s, bool(relu)))
    return layers


def macs(h, w):
    cin, total = 1, 0
    for kh, kw, s, c in LAYERS:
        h, w = (h - kh) // s + 1, (w - kw) // s + 1
        total += h * w * kh * kw * cin * c
        cin = c
    return total


if __name__ == '__main__':
    print('MMAC guided band 146x599:', macs(146, 599) / 1e6, ' tracking band 52x599:', macs(52, 599) / 1e6)
