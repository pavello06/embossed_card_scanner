"""Parses the Gurney-7B EPS outlines (tiny PostScript subset) into polygons
and rasterizes digit masks with even-odd fill."""
import math, os, re
import numpy as np
from PIL import Image, ImageDraw

NAMES = ['zero','one','two','three','four','five','six','seven','eight','nine']
HERE = os.path.dirname(os.path.abspath(__file__))
EPS_DIR = os.path.join(HERE, 'fonts', 'gurney', 'eps')  # see fetch_font.sh

def _arc_points(cx, cy, r, a1, a2, ccw, n=24):
    if ccw:
        while a2 < a1: a2 += 360
    else:
        while a2 > a1: a2 -= 360
    steps = max(2, int(abs(a2 - a1) / 360 * n * 4))
    return [(cx + r*math.cos(math.radians(a1 + (a2-a1)*t/steps)),
             cy + r*math.sin(math.radians(a1 + (a2-a1)*t/steps))) for t in range(steps+1)]

def _arct(p0, p1, p2, r):
    """PostScript arct: line toward p1, tangent arc of radius r turning to p2."""
    v1 = np.subtract(p0, p1); v2 = np.subtract(p2, p1)
    l1 = np.linalg.norm(v1); l2 = np.linalg.norm(v2)
    if l1 < 1e-9 or l2 < 1e-9 or r < 1e-9: return [tuple(p1)]
    u1 = v1/l1; u2 = v2/l2
    cos = np.clip(np.dot(u1, u2), -1, 1); theta = math.acos(cos)
    if abs(math.sin(theta)) < 1e-9: return [tuple(p1)]
    d = r / math.tan(theta/2)
    t1 = np.add(p1, u1*d); t2 = np.add(p1, u2*d)
    bis = (u1+u2)/np.linalg.norm(u1+u2)
    c = np.add(p1, bis * r/math.sin(theta/2))
    a1 = math.degrees(math.atan2(t1[1]-c[1], t1[0]-c[0]))
    a2 = math.degrees(math.atan2(t2[1]-c[1], t2[0]-c[0]))
    cross = u1[0]*u2[1]-u1[1]*u2[0]
    pts = _arc_points(c[0], c[1], r, a1, a2, ccw=cross < 0)
    return [tuple(t1)] + pts

def parse(name):
    text = open(os.path.join(EPS_DIR, name + '.eps')).read()
    body = text.split('%%EndProlog', 1)[1].split('showpage', 1)[0]
    body = '\n'.join(l for l in body.splitlines() if not l.startswith('%'))
    tokens = re.findall(r'/?[A-Za-z]+|-?\d*\.?\d+', body)
    stack, env, paths, cur = [], {}, [], []
    sx = sy = 1.0  # only the per-glyph flip after the initial page scale matters
    scaled = False
    for tok in tokens:
        if re.fullmatch(r'-?\d*\.?\d+', tok): stack.append(float(tok)); continue
        if tok.startswith('/'): stack.append(tok[1:]); continue
        if tok in env: stack.append(env[tok]); continue
        if tok == 'def': v = stack.pop(); k = stack.pop(); env[k] = v
        elif tok == 'dup': stack.append(stack[-1])
        elif tok in ('add','sub','mul','div'):
            b = stack.pop(); a = stack.pop()
            stack.append({'add':a+b,'sub':a-b,'mul':a*b,'div':a/b}[tok])
        elif tok == 'moveto':
            y = stack.pop(); x = stack.pop()
            if cur: paths.append(cur)
            cur = [(x, y)]
        elif tok == 'lineto':
            y = stack.pop(); x = stack.pop(); cur.append((x, y))
        elif tok == 'arct':
            r = stack.pop(); y2 = stack.pop(); x2 = stack.pop(); y1 = stack.pop(); x1 = stack.pop()
            cur += _arct(cur[-1], (x1, y1), (x2, y2), r)
        elif tok in ('arc','arcn'):
            a2 = stack.pop(); a1 = stack.pop(); r = stack.pop(); cy = stack.pop(); cx = stack.pop()
            cur += _arc_points(cx, cy, r, a1, a2, ccw=(tok == 'arc'))
        elif tok == 'closepath':
            if cur: paths.append(cur); cur = []
        elif tok == 'scale':
            b = stack.pop(); a = stack.pop()
            if scaled: sx *= a; sy *= b
            scaled = True
        elif tok == 'translate': stack.pop(); stack.pop()
        elif tok in ('setlinewidth','setgray'): stack.pop()
        else: stack.clear()  # stroke, showpage, save, etc.
    if cur: paths.append(cur)
    return [[(x*sx, y*sy) for x, y in p] for p in paths]

_cache = {}
def outlines(digit):
    if digit not in _cache: _cache[digit] = parse(NAMES[digit])
    return _cache[digit]

# Glyph box in font units (all digits share it; "8" is slightly shorter).
HALF_W, HALF_H = 1.275, 2.165

def render(digit, height_px, ss=4, stroke_scale=1.0):
    """Binary mask (float 0..1) of a digit `height_px` tall, glyph-box width."""
    h = height_px * ss; scale = h / (2*HALF_H)
    w = int(round(2*HALF_W*scale)); h = int(round(h))
    mask = np.zeros((h, w), bool)
    for path in outlines(digit):
        im = Image.new('1', (w, h), 0)
        ImageDraw.Draw(im).polygon([((x+HALF_W)*scale, (HALF_H-y)*scale) for x, y in path], fill=1)
        mask ^= np.asarray(im, bool)
    img = Image.fromarray((mask*255).astype(np.uint8))
    img = img.resize((max(1, w//ss), height_px), Image.LANCZOS)
    return np.asarray(img).astype(float)/255

if __name__ == '__main__':
    import sys
    tiles = [render(d, 80) for d in range(10)]
    pad = 10
    W = sum(t.shape[1]+pad for t in tiles)
    out = np.zeros((100, W))
    x = 0
    for t in tiles: out[10:90, x:x+t.shape[1]] = t; x += t.shape[1]+pad
    Image.fromarray((255-out*255).astype(np.uint8)).save(sys.argv[1])
