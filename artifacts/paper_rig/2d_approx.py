#!/usr/bin/env python3
"""Fast 2D approximation of paper.blend's Cycles look — v2 after honest self-review.
v2 fixes: content on multiple peeking sheets, per-sheet crumple warp, matte gray
paper (no blow-out), ragged silhouettes, stronger soft shadow, gentler gradient."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import json, math, os

OUT = "/home/riju279/Documents/code/drishyam-1/artifacts/paper_rig"
os.makedirs(OUT, exist_ok=True)
W = H = 2160
RES = getattr(Image, "Resampling", Image)
try:
    f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except Exception:
    f_title = f_small = ImageFont.load_default()

PAPER = 218.0   # matte gray-white, not 255

def make_page(rng, pw, ph):
    img = Image.new("RGB", (pw, ph), (246, 245, 242))
    d = ImageDraw.Draw(img)
    m = max(8, int(pw * 0.07))
    d.text((m, m), "QUARTERLY BUSINESS REPORT", font=f_title, fill=(30, 30, 35))
    d.line([(m, m + 44), (pw - m, m + 44)], fill=(80, 80, 88), width=2)
    qy = m + 62
    d.rectangle([m, qy, pw - m, qy + int(ph * 0.11)], fill=(226, 226, 224))
    by = qy + int(ph * 0.13)
    col_w = (pw - 2 * m - 12) // 2
    rc = np.random.RandomState(rng.randint(0, 2**31))
    for ci in range(2):
        x0 = m + ci * (col_w + 12)
        y = by
        while y < ph - int(ph * 0.30):
            x = x0
            for _ in range(rc.randint(3, 7)):
                w = rc.randint(14, 34)
                if x + w > x0 + col_w: break
                d.rectangle([x, y, x + w, y + 5], fill=(60, 60, 66))
                x += w + 5
            y += 11
    ch_h, ch_w = int(ph * 0.20), int(pw * 0.42)
    cx, cy = pw - m - ch_w, ph - m - ch_h - 14
    d.rectangle([cx, cy, cx + ch_w, cy + ch_h], outline=(90, 90, 100), width=1)
    bars = rc.randint(20, ch_h - 12, 6)
    bw = (ch_w - 20) // 6
    for i, bh in enumerate(bars):
        d.rectangle([cx + 10 + i * bw, cy + ch_h - 8 - bh, cx + 10 + i * bw + bw - 6,
                     cy + ch_h - 8], fill=(80, 110, 175))
    ly = ph - m - 10
    for i in range(3):
        lx = m + i * int(pw * 0.22)
        d.rectangle([lx, ly, lx + 8, ly + 8], fill=(80, 110, 175))
        d.text((lx + 12, ly - 2), "lorem ipsum", font=f_small, fill=(80, 80, 86))
    return np.asarray(img, float)

def crumple(arr, rng, amp):
    """Low-frequency smooth bend field — approximates Cycles' crumpled paper."""
    h, w = arr.shape[:2]
    s = 2
    hs, ws = h // s, w // s
    ys, xs = np.mgrid[0:hs, 0:ws]
    p1, p2, p3 = rng.uniform(0, 6.28, 3)
    dxs = amp * np.sin(xs/ws*2*np.pi*1.3 + p1) * np.sin(ys/hs*2*np.pi*0.9 + p2)
    dys = amp * 0.5 * np.sin(ys/hs*2*np.pi*1.1 + p3) * np.sin(xs/ws*2*np.pi*0.7 + p1)
    dys = dys + amp * 0.3 * np.sin((xs+ys)/(ws+hs)*2*np.pi*2.0 + p2)  # diagonal crease
    xs2 = np.clip(xs + dxs, 0, ws - 1).astype(int)
    ys2 = np.clip(ys + dys, 0, hs - 1).astype(int)
    small = arr[::s, ::s]
    out = small[ys2, xs2]
    # shading from the bend (fake normal): brightness follows local dy gradient
    gy = np.gradient(dys, axis=0)
    shade = 1.0 + np.clip(gy * 8.0, -0.18, 0.18)
    out = out * shade[..., None] if out.ndim == 3 else out * shade
    img = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).resize((w, h), RES.BILINEAR)
    return np.asarray(img, float)

def paste(canvas, tile, alpha, fy, fx):
    Hh, Ww = canvas.shape[:2]
    th, tw = tile.shape[:2]
    y0, x0 = max(0, fy), max(0, fx)
    y1, x1 = min(Hh, fy + th), min(Ww, fx + tw)
    if y1 <= y0 or x1 <= x0: return
    ty0, tx0 = y0 - fy, x0 - fx
    c = canvas[y0:y1, x0:x1]
    t = tile[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)]
    a = alpha[ty0:ty0 + (y1 - y0), tx0:tx0 + (x1 - x0)]
    canvas[y0:y1, x0:x1] = c * (1 - a) + t * a

def render_grid(name, seed):
    rng = np.random.RandomState(seed)
    canvas = np.zeros((H, W, 3), float)
    trav = {"light_left": (0.9, 0.5), "light_right": (-0.9, 0.5),
            "light_top": (0.0, 0.9)}[name]
    L = np.array(trav, float); L /= np.linalg.norm(L)
    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    field = 1.0 - 0.14 * (((xx - W/2) * L[0] + (yy - H/2) * L[1]) / (W/2))
    cell_w, cell_h = W // 2, H // 3
    centers = [(cell_w*0.5, int(H*0.17)), (cell_w*1.5, int(H*0.17)),
               (cell_w*0.5, int(H*0.50)), (cell_w*1.5, int(H*0.50)),
               (cell_w*0.5, int(H*0.83)), (cell_w*1.5, int(H*0.83))]
    for (cx, cy) in centers:
        pw, ph = int(cell_w*0.56), int(cell_w*0.56*297/210)
        tw, th = int(pw*2.0), int(ph*2.0)
        tile = np.zeros((th, tw, 3), float)
        alpha = np.zeros((th, tw, 1), float)
        sh = Image.new("L", (tw, th), 0)
        ImageDraw.Draw(sh).ellipse([tw//2-int(pw*0.70)+int(110*L[0]), th//2-int(ph*0.70)+int(110*L[1]),
                                    tw//2+int(pw*0.70)+int(110*L[0]), th//2+int(ph*0.70)+int(110*L[1])], fill=95)
        shadow = np.asarray(sh.filter(ImageFilter.GaussianBlur(60)), float)[..., None]/255.0
        tile += np.array([13, 13, 15])[None, None] * shadow
        alpha = np.maximum(alpha, shadow*0.5)
        n = rng.randint(14, 20)
        top_content = make_page(rng, pw, ph)
        drifts = np.cumsum(rng.normal(0, 3.2, (n, 2)), axis=0)
        content_depth = rng.randint(4, 7)   # how many top sheets show content
        for si in range(n):
            drift = drifts[si]
            if si >= n - content_depth:
                arr = top_content * (0.72 + 0.06*(n-1-si))
                if si < n - 1:
                    arr = np.asarray(Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).filter(
                        ImageFilter.GaussianBlur(1.2 + 0.4*(n-1-si))), float)
                amp = 6.0 + 2.0*rng.rand()
            else:
                arr = np.full((ph, pw), PAPER * (0.86 + 0.05*rng.rand()))
                amp = 5.0 + 2.5*rng.rand()
            arr = crumple(arr, rng, amp)
            arr = np.asarray(Image.fromarray(np.clip(arr,0,255).astype(np.uint8)).rotate(
                rng.normal(0, 2.2), resample=RES.BICUBIC, expand=False,
                fillcolor=0 if arr.ndim == 2 else (0, 0, 0)), float)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            vis = arr.max(axis=2) > 1
            ox = tw//2 - pw//2 + int(drift[0] - si*L[0]*1.6)
            oy = th//2 - ph//2 + int(drift[1] - si*L[1]*1.6)
            a = alpha[oy:oy+ph, ox:ox+pw]
            tile[oy:oy+ph, ox:ox+pw] = np.where(vis[..., None], arr, tile[oy:oy+ph, ox:ox+pw])
            alpha[oy:oy+ph, ox:ox+pw] = np.maximum(a, vis[..., None])
        fy, fx = int(cy - th/2), int(cx - tw/2)
        y0, x0 = max(0, fy), max(0, fx)
        y1, x1 = min(H, fy + th), min(W, fx + tw)
        tile[y0-fy:y1-fy, x0-fx:x1-fx] *= field[y0:y1, x0:x1][..., None]
        paste(canvas, tile, alpha, fy, fx)
    noise = rng.normal(0, 2.0, (H, W, 1))
    out = np.clip(canvas + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(out).filter(ImageFilter.GaussianBlur(0.4))
    img.save(f"{OUT}/approx_{name}.png")
    return {"azimuth": name, "seed": seed, "light_travel_xy": [round(float(v), 3) for v in L]}

recipes = [render_grid(nm, 100+i) for i, nm in enumerate(["light_left", "light_right", "light_top"])]
with open(f"{OUT}/recipes.json", "w") as f:
    json.dump({"type": "2d_fast_pipeline_recipe_v2", "note": "every render = f(recipe); no pixels stored",
               "rows": recipes}, f, indent=1)

pairs = [("/home/riju279/Downloads/5.png", f"{OUT}/approx_light_left.png"),
         ("/home/riju279/Downloads/6.png", f"{OUT}/approx_light_right.png"),
         ("/home/riju279/Downloads/8.png", f"{OUT}/approx_light_top.png")]
strip = Image.new("RGB", (2170, 3290), (10, 10, 10))
for i, (a, b) in enumerate(pairs):
    strip.paste(Image.open(a).resize((1080, 1080)), (0, i*1095))
    strip.paste(Image.open(b).resize((1080, 1080)), (1090, i*1095))
strip.save(f"{OUT}/comparison_YOURS_left_vs_MINE_right.png")
print("DONE v2")
