"""Exports a trained CNN (train_cnn.py does this itself; TensorFlow venv):

    python export_cnn.py model/embossed_digits.keras

writes <prefix>.cnn (CNN1 format, see cnn.py) and <prefix>_cnn.tflite next to
the .keras file. Batch norm, input scaling and the calibration temperature
(model.keras.temperature) are folded into the convolutions.
"""
import os
import sys

import numpy as np

import cnn
import synth
import synth_dense


def keras_plain(layers):
    """Inference model (no BN, softmax output) holding the folded weights."""
    model = cnn.build_keras(train=False)
    for i, (W, b, _, _) in enumerate(layers):
        model.get_layer(f'conv{i}').set_weights([W, b])
    return model


def export(trained, T, prefix):
    import tensorflow as tf
    layers = cnn.fold(trained, T)
    cnn.write(layers, prefix + '.cnn')

    # Check folding on a synthetic band: trained logits / T must give the same probabilities.
    synth.rng = np.random.default_rng(0)
    band, _ = synth_dense.sample()
    ref = cnn.forward(cnn.read(prefix + '.cnn'), band)
    logits = trained.predict(band[None, ..., None].astype(np.float32), verbose=0)[0] / T
    p = np.exp(logits - logits.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    print('folded vs trained max |diff| =', float(np.abs(ref - p).max()))

    plain = keras_plain(layers)
    open(prefix + '_cnn.tflite', 'wb').write(tf.lite.TFLiteConverter.from_keras_model(plain).convert())
    interp = tf.lite.Interpreter(model_path=prefix + '_cnn.tflite')
    interp.resize_tensor_input(0, [1, *band.shape, 1]); interp.allocate_tensors()
    interp.set_tensor(interp.get_input_details()[0]['index'], band[None, ..., None].astype(np.float32)); interp.invoke()
    got = interp.get_tensor(interp.get_output_details()[0]['index'])[0]
    print('tflite vs numpy max |diff| =', float(np.abs(got - ref).max()), 'shape', got.shape)
    print('wrote', prefix + '.cnn', prefix + '_cnn.tflite')


if __name__ == '__main__':
    src = sys.argv[1]
    T = float(open(src + '.temperature').read()) if os.path.exists(src + '.temperature') else 1.0
    export(cnn.load_keras(src), T, os.path.splitext(src)[0])
