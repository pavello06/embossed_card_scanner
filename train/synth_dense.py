"""Synthetic card-band crops with dense labels for the CNN (see cnn.py).

Reuses the relief rendering of synth.py, with longer rows grouped like real
numbers (4 digits, gap), an optional smaller embossed row (expiry date, name)
that must read as background, and glare.
"""
import numpy as np

import cnn
import synth

CROP_H, CROP_W = 80, 160
IGNORE = -1


def _glare(img):
    """Broad specular highlight, e.g. a lamp reflected by the card."""
    h, w = img.shape
    gy, gx = np.mgrid[0:h, 0:w]
    cx, cy = synth.rng.uniform(0, w), synth.rng.uniform(-h / 2, 1.5 * h)
    sx, sy = synth.rng.uniform(0.1, 0.6) * w, synth.rng.uniform(0.2, 1.0) * h
    return np.clip(img + synth.rng.uniform(0.15, 0.6) * np.exp(-((gx - cx) / sx) ** 2 - ((gy - cy) / sy) ** 2), 0, 1)


def sample():
    """One crop: (uint8 [CROP_H, CROP_W], int8 labels on the 2 px output grid)."""
    rng = synth.rng
    ss = 2
    scale = rng.uniform(0.85, 1.15)
    n = 12
    if rng.random() < 0.75:  # 4-digit groups separated by an empty pitch
        k0 = int(rng.integers(0, 5))
        digits = [None if (i + k0) % 5 == 4 else int(rng.integers(0, 10)) for i in range(n)]
    else:
        digits = [int(rng.integers(0, 10)) for _ in range(n)]
    mask, centers, cy = synth.render_row(digits, scale, ss)

    # Pad vertically so the row can sit anywhere in (or outside) the crop.
    pad = (CROP_H + 24) * ss
    mask = np.pad(mask, ((pad, pad), (0, 0)))
    cy += pad / ss
    if rng.random() < 0.35:  # smaller embossed text above or below: background
        small = rng.uniform(0.55, 0.75)
        m2, _, cy2 = synth.render_row([int(rng.integers(0, 10)) for _ in range(8)], small, ss)
        dy = int(round((cy + rng.choice([-1, 1]) * rng.uniform(1.0, 1.7) * synth.GLYPH_H * scale - cy2) * ss))
        dx = int(rng.integers(-m2.shape[1] // 2, mask.shape[1] - m2.shape[1] // 2))
        ys, xs = slice(max(dy, 0), min(dy + m2.shape[0], mask.shape[0])), slice(max(dx, 0), min(dx + m2.shape[1], mask.shape[1]))
        mask[ys, xs] = np.maximum(mask[ys, xs], m2[ys.start - dy:ys.stop - dy, xs.start - dx:xs.stop - dx])

    shaded = synth.emboss(mask, ss)
    if rng.random() < 0.3:
        shaded = _glare(shaded)
    shaded, to_warped = synth.warp(shaded)
    img = synth.camera(shaded, ss)

    # Glyph centres in the 1x warped image.
    glyphs = [(d, *(np.array(to_warped(cx * ss, cy * ss)) / ss)) for d, cx in zip(digits, centers) if d is not None]
    pitch, gh = synth.PITCH * scale, synth.GLYPH_H * scale

    # Crop: the row centre lands anywhere from just above to just below it.
    x0 = int(rng.integers(0, img.shape[1] - CROP_W + 1))
    y0 = int(np.clip(round(cy - rng.uniform(-10, CROP_H + 10)), 0, img.shape[0] - CROP_H))
    crop = img[y0:y0 + CROP_H, x0:x0 + CROP_W]

    oh, ow = 2 * cnn.out_cells(CROP_H, 0), 2 * cnn.out_cells(CROP_W, 1)
    Y = y0 + cnn.CENTRE_Y + 2 * np.arange(oh)[:, None]
    X = x0 + cnn.CENTRE_X + 2 * np.arange(ow)[None, :]
    labels = np.full((oh, ow), synth.BACKGROUND, np.int8)
    for d, gx, gy in glyphs:
        dx, dy = np.abs(X - gx), np.abs(Y - gy)
        near = (dx < 0.3 * pitch) & (dy < 0.45 * gh)
        labels[near] = IGNORE
    for d, gx, gy in glyphs:  # positives win over a neighbour's ignore zone
        labels[(np.abs(X - gx) <= 2) & (np.abs(Y - gy) <= 3)] = d
    return crop, labels


def _batch(args):
    seed, count = args
    synth.rng = np.random.default_rng(seed)
    out = [sample() for _ in range(count)]
    return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])


def build(n, seed0, per_job=250):
    from multiprocessing import Pool
    jobs = [(seed0 + i, per_job) for i in range(n // per_job)]
    with Pool() as pool:
        parts = pool.map(_batch, jobs)
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


if __name__ == '__main__':
    import sys
    from PIL import Image
    synth.rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    tiles = []
    for _ in range(8):
        crop, labels = sample()
        vis = np.stack([crop] * 3, -1).copy()
        for v, u in zip(*np.nonzero(labels < synth.BACKGROUND)):
            if labels[v, u] >= 0:
                vis[cnn.CENTRE_Y + 2 * v, cnn.CENTRE_X + 2 * u] = (255, 0, 0)
        tiles.append(np.pad(vis, ((2, 2), (2, 2), (0, 0)), constant_values=255))
    Image.fromarray(np.vstack(tiles)).resize((CROP_W * 3, len(tiles) * (CROP_H + 4) * 3), Image.NEAREST).save(sys.argv[1])
    print('label counts', np.bincount((labels.ravel() + 1).astype(int), minlength=12))
