"""Synthetic embossed card-number relief for the digit CNN (see synth_dense.py).

Geometry is in a normalized card frame of PX_PER_MM pixels per millimetre.
ISO/IEC 7811 embossing: character pitch 3.63 mm, glyph ~2.55 x 4.33 mm.
"""
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import glyphs

PX_PER_MM = 7.0
PITCH = 3.63 * PX_PER_MM          # ~18.15 px
GLYPH_H = 4.33 * PX_PER_MM        # ~21.65 px
BACKGROUND = 10                   # class id for "not centered on a digit"

MOTION_BLUR_P = 0.3               # hand shake
ORNAMENT_P = 0.35                 # line-art over digits

rng = np.random.default_rng()


def _blur(a, sigma):
    if sigma <= 0.05:
        return a
    # Separable gaussian via numpy (PIL can't blur float images).
    r = int(np.ceil(sigma * 3))
    x = np.arange(-r, r + 1)
    k = np.exp(-x ** 2 / (2 * sigma ** 2)); k /= k.sum()
    a = np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, mode='edge'), k, 'valid'), 0, a)
    a = np.apply_along_axis(lambda m: np.convolve(np.pad(m, r, mode='edge'), k, 'valid'), 1, a)
    return a


def _texture(h, w):
    """Random card-like background albedo in 0..1."""
    if rng.random() < ORNAMENT_P:
        return _ornament(h, w)
    kind = rng.choice(['flat', 'gradient', 'noise', 'lines', 'mixed'], p=[.2, .2, .2, .2, .2])
    base = rng.uniform(0.03, 0.95)
    a = np.full((h, w), base)
    if kind in ('gradient', 'mixed'):
        gy, gx = np.mgrid[0:h, 0:w]
        ang = rng.uniform(0, np.pi)
        t = (gx * np.cos(ang) + gy * np.sin(ang)) / max(h, w)
        a = a + rng.uniform(-0.6, 0.6) * (t - t.mean())
    if kind in ('noise', 'mixed'):
        n = _blur(rng.normal(0, 1, (h, w)), rng.uniform(1, 6))
        a = a + n / (n.std() + 1e-6) * rng.uniform(0.03, 0.2)
    if kind in ('lines', 'mixed'):
        img = Image.new('L', (w, h), 0)
        d = ImageDraw.Draw(img)
        for _ in range(rng.integers(2, 25)):
            pts = [(rng.uniform(-10, w + 10), rng.uniform(-10, h + 10)) for _ in range(rng.integers(2, 5))]
            d.line(pts, fill=255, width=int(rng.integers(1, 4)))
        if rng.random() < 0.35:  # large filled artwork shapes crossing the row
            for _ in range(rng.integers(1, 4)):
                cx, cy = rng.uniform(0, w), rng.uniform(0, h)
                pts = [(cx + rng.normal(0, w / 6), cy + rng.normal(0, h / 2)) for _ in range(rng.integers(3, 7))]
                d.polygon(pts, fill=int(rng.integers(120, 256)))
        if rng.random() < 0.3:  # glyph-like clutter (other text on the card)
            for _ in range(rng.integers(1, 6)):
                x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
                d.rectangle([x0, y0, x0 + rng.uniform(2, 8), y0 + rng.uniform(4, 12)], outline=255)
        lines = np.asarray(img).astype(float) / 255
        a = a + lines * rng.uniform(-0.7, 0.7)
    return np.clip(a, 0, 1)


def _ornament(h, w):
    """Line-art illustration (feathers, hatching) printed over part of the
    row, like artwork crossing the embossed digits on some cards."""
    dark = rng.uniform(0.02, 0.3)
    img = Image.new('L', (w, h), int(dark * 255))
    d = ImageDraw.Draw(img)
    light = int(rng.uniform(0.55, 0.95) * 255)
    # A bright shape covering a random part of the row...
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    pts = [(cx + rng.normal(0, w / 4), cy + rng.normal(0, h)) for _ in range(rng.integers(4, 8))]
    d.polygon(pts, fill=light)
    mask = Image.new('1', (w, h), 0)
    ImageDraw.Draw(mask).polygon(pts, fill=1)
    # ...filled with dense dark strokes.
    art = Image.new('L', (w, h), 0)
    ad = ImageDraw.Draw(art)
    density = max(1.0, h * w / 35000)  # stroke count was tuned on a 95 x 360 row canvas
    for _ in range(int(rng.integers(30, 120) * density)):
        x0, y0 = rng.uniform(0, w), rng.uniform(0, h)
        ang, ln = rng.uniform(0, np.pi), rng.uniform(6, 40)
        if rng.random() < 0.5:
            ad.line([(x0, y0), (x0 + ln * np.cos(ang), y0 + ln * np.sin(ang))], fill=255, width=int(rng.integers(2, 5)))
        else:
            r = rng.uniform(4, 16)
            ad.arc([x0 - r, y0 - r, x0 + r, y0 + r], rng.uniform(0, 360), rng.uniform(0, 360), fill=255, width=int(rng.integers(2, 4)))
    img.paste(int(dark * 255), (0, 0), Image.fromarray((np.asarray(art) > 0) & np.asarray(mask)).convert('1'))
    if rng.random() < 0.5:  # long light lines (branches) crossing everything
        d = ImageDraw.Draw(img)
        for _ in range(int(rng.integers(1, 5) * np.sqrt(density))):
            pts = [(rng.uniform(-w / 4, 1.25 * w), rng.uniform(-h / 4, 1.25 * h)) for _ in range(rng.integers(2, 4))]
            d.line(pts, fill=light, width=int(rng.integers(2, 6)))
    return np.asarray(img).astype(float) / 255


def render_row(digits, scale=1.0, ss=2):
    """Height mask of a row of digits laid out on the ISO pitch.

    Returns (mask, centers) at supersampling `ss`; centers are x of each glyph.
    """
    gh = int(round(GLYPH_H * scale * ss))
    pitch = PITCH * scale * ss
    margin = int(pitch * 1.5)
    width = int(margin * 2 + pitch * len(digits))
    height = int(gh * 2.2)
    mask = np.zeros((height, width))
    top = (height - gh) // 2
    centers = []
    for i, d in enumerate(digits):
        if d is None:
            centers.append(None)
            continue
        g = glyphs.render(d, gh)
        cx = margin + pitch * (i + 0.5)
        x0 = int(round(cx - g.shape[1] / 2))
        mask[top:top + gh, x0:x0 + g.shape[1]] = np.maximum(mask[top:top + gh, x0:x0 + g.shape[1]], g)
        centers.append(cx / ss)
    return mask, centers, (top + gh / 2) / ss


def emboss(mask, ss=2):
    """Shade a raised relief of `mask` over a random background (gray 0..1)."""
    h, w = mask.shape
    height = _blur(mask, rng.uniform(0.6, 1.6) * ss)
    albedo = _texture(h, w)
    painted = rng.random() < 0.35
    if painted:
        tip = rng.choice([rng.uniform(0.0, 0.25), rng.uniform(0.6, 1.0)])
        albedo = albedo * (1 - mask) + tip * mask
    # Lighting: azimuth mostly from above, random elevation.
    az = rng.normal(-np.pi / 2, 1.0) if rng.random() < 0.8 else rng.uniform(-np.pi, np.pi)
    el = rng.uniform(0.35, 1.2)
    L = np.array([np.cos(az) * np.cos(el), np.sin(az) * np.cos(el), np.sin(el)])
    gy, gx = np.gradient(height * rng.uniform(1.5, 5.0) * ss)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    diffuse = np.clip(n @ L, 0, 1)
    flat = max(L[2], 0.05)
    shade = 0.35 + 0.65 * diffuse / flat * 0.8
    # Specular: embossed plastic/foil edges catch light.
    view = np.array([0, 0, 1.0])
    hvec = (L + view) / np.linalg.norm(L + view)
    spec = np.clip(n @ hvec, 0, 1) ** rng.uniform(8, 40) * rng.uniform(0.0, 0.9)
    # Cast shadow on the side away from the light.
    sx, sy = int(round(-np.cos(az) * ss * 1.5)), int(round(-np.sin(az) * ss * 1.5))
    shadow = np.roll(np.roll(mask, sy, 0), sx, 1) * (1 - mask)
    img = albedo * shade * (1 - shadow * rng.uniform(0, 0.5)) + spec
    return np.clip(img, 0, 1)


def camera(img, ss=2):
    """Downsample and add capture artefacts; returns uint8 gray at 1x."""
    h, w = img.shape
    im = Image.fromarray((img * 255).astype(np.uint8)).resize((w // ss, h // ss), Image.BILINEAR)
    a = np.asarray(im).astype(float) / 255
    a = _blur(a, rng.uniform(0.0, 1.1))
    if rng.random() < MOTION_BLUR_P:  # hand shake: linear motion blur, mostly horizontal
        a = _motion_blur(a, rng.uniform(1, 4), rng.normal(0, 0.5))
    a = a ** rng.uniform(0.7, 1.4)
    a = a * rng.uniform(0.6, 1.2) + rng.uniform(-0.1, 0.1)
    a = a + rng.normal(0, rng.uniform(0.0, 0.04), a.shape)
    a = np.clip(a, 0, 1)
    im = Image.fromarray((a * 255).astype(np.uint8))
    if rng.random() < 0.6:
        buf = io.BytesIO()
        im.save(buf, 'JPEG', quality=int(rng.integers(30, 90)))
        im = Image.open(buf)
    return np.asarray(im.convert('L')).astype(np.uint8)


def _motion_blur(a, length, angle):
    """Average of `length` px shifted copies along `angle` (radians)."""
    n = max(int(round(length)), 1)
    acc = np.zeros_like(a)
    for i in range(n):
        t = i - (n - 1) / 2
        dx, dy = int(round(t * np.cos(angle))), int(round(t * np.sin(angle)))
        acc += np.roll(np.roll(a, dy, 0), dx, 1)
    return acc / n


def warp(img):
    """Random anisotropic scale + small rotation (imperfect card crop).

    Returns the warped image and a function mapping pre-warp points to it.
    """
    h, w = img.shape
    sx, sy = rng.uniform(0.92, 1.08), rng.uniform(0.92, 1.08)
    t = np.radians(rng.uniform(-3, 3))
    A = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) @ np.diag([sx, sy])
    Ai = np.linalg.inv(A)
    c = np.array([w / 2, h / 2])
    off = c - Ai @ c
    im = Image.fromarray((img * 255).astype(np.uint8)).transform(
        (w, h), Image.AFFINE, (Ai[0, 0], Ai[0, 1], off[0], Ai[1, 0], Ai[1, 1], off[1]),
        resample=Image.BILINEAR, fillcolor=int(img.mean() * 255))
    return np.asarray(im).astype(float) / 255, lambda x, y: tuple(c + A @ (np.array([x, y]) - c))
