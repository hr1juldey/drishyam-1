#!/usr/bin/env python3
"""GPU (PyTorch) batched renderer for the synthetic paper-pile look.
Same physics layers as 2d_approx.py (CPU reference), but every op is a torch
tensor op on the 3060: grid_sample warps (crumple+rotation unified), analytic
gaussian shadow, gradient illumination, batched compositing. Plus torch-native
SSIM/PSNR to NUMBER the approximation against the Cycles templates.

Row format: one paper stack per image, 720x1080 tile (batch element).
Grid format: 6 tiles -> 2160x2160 comparison grid, same layout/seeds as CPU v2.
"""
import torch, math, json, os, time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda"
TD = torch.bfloat16
try:
    import torchvision
    TV = torchvision.__version__
except Exception:
    TV = None

FONT_OK = os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")

def make_page_pil(rng, pw, ph):
    img = Image.new("RGB", (pw, ph), (246, 245, 242))
    if not FONT_OK:
        d = ImageDraw.Draw(img)
        for _ in range(40):
            x, y = rng.randint(10, pw - 40), rng.randint(10, ph - 10)
            d.rectangle([x, y, x + rng.randint(14, 34), y + 5], fill=(60, 60, 66))
        return np.asarray(img, float)
    from PIL import ImageFont
    f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
    f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    d = ImageDraw.Draw(img)
    m = max(8, int(pw * 0.07))
    d.text((m, m), "QUARTERLY BUSINESS REPORT", font=f_title, fill=(30, 30, 35))
    d.line([(m, m + 44), (pw - m, m + 44)], fill=(80, 80, 88), width=2)
    qy = m + 62
    d.rectangle([m, qy, pw - m, qy + int(ph * 0.11)], fill=(226, 226, 224))
    rc = np.random.RandomState(rng.randint(0, 2**31))
    by = qy + int(ph * 0.13)
    col_w = (pw - 2 * m - 12) // 2
    for ci in range(2):
        x0 = m + ci * (col_w + 12); y = by
        while y < ph - int(ph * 0.30):
            x = x0
            for _ in range(rc.randint(3, 7)):
                w = rc.randint(14, 34)
                if x + w > x0 + col_w: break
                d.rectangle([x, y, x + w, y + 5], fill=(60, 60, 66)); x += w + 5
            y += 11
    ch_h, ch_w = int(ph * 0.20), int(pw * 0.42)
    cx, cy = pw - m - ch_w, ph - m - ch_h - 14
    d.rectangle([cx, cy, cx + ch_w, cy + ch_h], outline=(90, 90, 100), width=1)
    bars = rc.randint(20, ch_h - 12, 6); bw = (ch_w - 20) // 6
    for i, bh in enumerate(bars):
        d.rectangle([cx + 10 + i * bw, cy + ch_h - 8 - bh, cx + 10 + i * bw + bw - 6, cy + ch_h - 8],
                    fill=(80, 110, 175))
    ly = ph - m - 10
    for i in range(3):
        lx = m + i * int(pw * 0.22)
        d.rectangle([lx, ly, lx + 8, ly + 8], fill=(80, 110, 175))
    return np.asarray(img, float)

TRAV = {"light_left": (0.9, 0.5), "light_right": (-0.9, 0.5), "light_top": (0.0, 0.9)}

class GPURenderer:
    def __init__(self, th=720, tw=1080, ph=626, pw=884, K=8, device=DEV, atlas_pages=None):
        self.th, self.tw, self.ph, self.pw = th, tw, ph, pw
        self.dev = device
        if atlas_pages is not None:
            from PIL import Image as _I
            pages = [np.asarray(_I.fromarray(np.clip(p, 0, 255).astype(np.uint8)).resize((pw, ph)),
                                np.float32) for p in atlas_pages]
            atlas = np.stack(pages)                       # (K,ph,pw,3)
            K = len(pages)
        else:
            rng = np.random.RandomState(7)
            atlas = np.stack([make_page_pil(rng, pw, ph) for _ in range(K)])
        t = torch.from_numpy(atlas).to(device, TD).permute(0, 3, 1, 2) / 255.0
        alpha = torch.ones(1, 1, ph, pw, device=device, dtype=TD)
        self.content = torch.cat([t, alpha.expand(K, 1, ph, pw)], 1)      # (K,4,ph,pw)
        plain = torch.full((1, 4, ph, pw), 218/255.0, device=device, dtype=TD)
        plain[:, 3] = 1.0
        self.plain = plain
        self.all_src = torch.cat([self.content, plain], 0)   # slot K = plain paper
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, th, device=device, dtype=TD),
                                torch.linspace(-1, 1, tw, device=device, dtype=TD), indexing="ij")
        self.base = torch.stack([xs, ys], -1)                     # (th,tw,2)

    def render_batch(self, B, azimuth, gen, n_lo=14, n_hi=20, content_idx=None,
                     amp_lo=5.0, amp_hi=7.5, drift_sigma=3.2, cdepth_lo=4, cdepth_hi=7,
                     field_strength=0.14, br_lo=0.86, br_hi=0.91, noise_sigma=2.0):
        th, tw, ph, pw, dev = self.th, self.tw, self.ph, self.pw, self.dev
        Lx, Ly = TRAV[azimuth]
        Ln = math.hypot(Lx, Ly); Lx, Ly = Lx / Ln, Ly / Ln
        out = torch.zeros(B, 3, th, tw, device=dev, dtype=TD)
        alpha_acc = torch.zeros(B, 1, th, tw, device=dev, dtype=TD)
        n = torch.randint(n_lo, n_hi, (B,), device=dev)
        S = int(n.max())
        drift = torch.randn(B, S, 2, device=dev, dtype=TD) * drift_sigma
        drift = drift.cumsum(1)
        rot = (torch.randn(B, S, device=dev, dtype=TD) * 2.2) * math.pi / 180
        amp = amp_lo + (amp_hi - amp_lo) * torch.rand(B, S, device=dev, dtype=TD)
        br = br_lo + (br_hi - br_lo) * torch.rand(B, S, device=dev, dtype=TD)
        cdepth = torch.randint(cdepth_lo, cdepth_hi, (B,), device=dev).clamp_(1, S)
        base = self.base.unsqueeze(0)                              # (1,th,tw,2)
        # shadow: analytic gaussian distance field, offset along light travel
        sx = (base[..., 0] * tw / 2 - 110 * Lx)
        sy = (base[..., 1] * th / 2 - 110 * Ly)
        d2 = (sx / (pw * 0.70)) ** 2 + (sy / (ph * 0.70)) ** 2
        shadow = torch.exp(-d2 / 1.2) * 0.5                        # (1,th,tw)
        glow = torch.tensor([13, 13, 15], device=dev, dtype=TD).view(1, 3, 1, 1) / 255
        out += glow * shadow.unsqueeze(1)
        alpha_acc = torch.maximum(alpha_acc, shadow.unsqueeze(1) * 0.5)
        for s in range(S):
            th_ = rot[:, s]                                        # (B,)
            dr = drift[:, s]                                       # (B,2)
            ca, sa = torch.cos(th_), torch.sin(th_)
            x = base[..., 0] * tw / 2; y = base[..., 1] * th / 2   # px in tile
            xr = x - (dr[:, 0] - s * Lx * 1.6).view(B, 1, 1)
            yr = y - (dr[:, 1] - s * Ly * 1.6).view(B, 1, 1)
            xs_r = xr * ca.view(B, 1, 1) + yr * sa.view(B, 1, 1)
            ys_r = -xr * sa.view(B, 1, 1) + yr * ca.view(B, 1, 1)
            A = amp[:, s].view(B, 1, 1)
            p1 = torch.rand(B, 1, 1, device=dev, dtype=TD) * 6.283
            p2 = torch.rand(B, 1, 1, device=dev, dtype=TD) * 6.283
            dx = A * torch.sin(xs_r / pw * 2 * math.pi * 1.3 + p1) * torch.sin(ys_r / ph * 2 * math.pi * 0.9 + p2)
            dy = A * 0.5 * torch.sin(ys_r / ph * 2 * math.pi * 1.1 + p2) * torch.sin(xs_r / pw * 2 * math.pi * 0.7 + p1)
            dy = dy + A * 0.3 * torch.sin((xs_r + ys_r) / (pw + ph) * 2 * math.pi * 2.0 + p2)
            gx = dy * 8.0                                          # fake-normal shading
            shade = 1 + torch.clamp(torch.gradient(gx, dim=1)[0], -0.18, 0.18)
            u = (xs_r + dx) / pw + 0.5
            v = (ys_r + dy) / ph + 0.5
            grid = torch.stack([u * 2 - 1, v * 2 - 1], -1)   # NO clamp: out-of-page -> zeros (padding_mode)
            is_content = (s >= S - cdepth)                         # (B,) bool
            if content_idx is None:
                idx = torch.randint(0, self.content.shape[0], (B,), device=dev)
            else:
                idx = torch.tensor(content_idx, device=dev)
            flat_idx = torch.where(is_content, idx, torch.full_like(idx, self.all_src.shape[0] - 1))
            src = self.all_src[flat_idx]                           # (B,4,ph,pw) one gather
            warped = torch.nn.functional.grid_sample(
                src, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            brk = br[:, s].view(B, 1, 1, 1)
            rgb = warped[:, :3] * shade.unsqueeze(1) * brk * 255
            a = warped[:, 3:4]
            content_mask = ((s >= S - cdepth) & (s < S - 1)).float().view(B, 1, 1, 1)
            k = 5
            rgb_blur = torch.nn.functional.avg_pool2d(rgb, k, stride=1, padding=k // 2)
            rgb = torch.where(content_mask.bool(), rgb_blur, rgb)
            out = out * (1 - a) + rgb * a
            alpha_acc = 1 - (1 - alpha_acc) * (1 - a)
        # illumination gradient
        gx_ = torch.linspace(-1, 1, tw, device=dev, dtype=TD)
        gy_ = torch.linspace(-1, 1, th, device=dev, dtype=TD)
        field = 1.0 - field_strength * ((gx_.view(1, 1, -1) * Lx + gy_.view(-1, 1) * Ly))
        out = out * field.view(1, 1, th, tw)
        out = out + torch.randn(out.shape, device=dev, dtype=TD) * noise_sigma
        return out, alpha_acc

def _gauss_window(k=11, dev="cuda", dtype=torch.float32):
    x = torch.arange(k, device=dev, dtype=dtype) - (k - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    return g.unsqueeze(0).unsqueeze(0) * g.unsqueeze(0).unsqueeze(0).t().reshape(1, 1, 1, k).squeeze(-1) if False else (g[:, None] @ g[None, :]).view(1, 1, k, k)

def ssim_torch(a, b, k=11):
    """Proper windowed SSIM (gaussian 11x11, full-map mean). a,b (H,W) in [0,1]."""
    a = a.float(); b = b.float()
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    w = _gauss_window(k, a.device).expand(1, 1, k, k)
    mu_a = torch.nn.functional.conv2d(a[None, None], w, padding=k // 2)[0, 0]
    mu_b = torch.nn.functional.conv2d(b[None, None], w, padding=k // 2)[0, 0]
    mu_aa, mu_bb, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    s_a = torch.nn.functional.conv2d((a * a)[None, None], w, padding=k // 2)[0, 0] - mu_aa
    s_b = torch.nn.functional.conv2d((b * b)[None, None], w, padding=k // 2)[0, 0] - mu_bb
    s_ab = torch.nn.functional.conv2d((a * b)[None, None], w, padding=k // 2)[0, 0] - mu_ab
    ssim_map = ((2 * mu_ab + C1) * (2 * s_ab + C2)) / ((mu_aa + mu_bb + C1) * (s_a + s_b + C2))
    return float(ssim_map.mean())

def psnr_torch(a, b):
    a = a.float().mean(dim=-3) if a.dim() > 2 else a
    b = b.float().mean(dim=-3) if b.dim() > 2 else b
    mse = ((a - b) ** 2).mean()
    return float(10 * torch.log10(1.0 / mse))

def to_img(t):
    return np.clip(t.detach().float().cpu().numpy().transpose(1, 2, 0), 0, 255).astype(np.uint8)

if __name__ == "__main__":
    import random
    class G:
        def seed_drawn(self, a, b): return random.randint(a, b)
    g = G()

    # --- benchmark: ROW config (896x896 tile, portrait page — the factory row format) ---
    Rrow = GPURenderer(th=896, tw=896, ph=719, pw=509)
    for B in (8, 32, 64, 128):
        try:
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(3):
                out, _a = Rrow.render_batch(B, "light_left", g)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 3
            print(f"row896 batch {B:3d}: {dt*1000:7.1f} ms/batch -> {B/dt:8.1f} img/s "
                  f"-> 200k rows in {200000/(B/dt)/3600:.2f} h")
            del out, _a
            torch.cuda.empty_cache()
        except torch.OutOfMemoryError:
            print(f"row896 batch {B:3d}: OOM")
            torch.cuda.empty_cache()
            break

    # --- verification: tile proportions matched to CPU v2, same layout/seeds ---
    Rv = GPURenderer(th=1620, tw=1208, ph=853, pw=604)
    centers = [(540, int(2160*0.17)), (1620, int(2160*0.17)),
               (540, int(2160*0.50)), (1620, int(2160*0.50)),
               (540, int(2160*0.83)), (1620, int(2160*0.83))]
    for az in ("light_left",):
        batch, balpha = Rv.render_batch(6, az, g)
        canvas = Image.new("RGB", (2160, 2160), (3, 3, 4))
        for i, (cx, cy) in enumerate(centers):
            tile = Image.fromarray(to_img(batch[i]))
            mask = Image.fromarray((balpha[i, 0].float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8), "L")
            canvas.paste(tile, (int(cx - 1208/2), int(cy - 1620/2)), mask)
        canvas.save(f"{OUT}/gpu_{az}.png")
        cpu = Image.open(f"{OUT}/approx_{az}.png").convert("RGB")
        tpl = Image.open("/home/riju279/Downloads/5.png").convert("RGB").resize((2160, 2160))
        tt = torch.from_numpy(np.asarray(canvas, np.float32) / 255).permute(2, 0, 1)[0]
        tc = torch.from_numpy(np.asarray(cpu, np.float32) / 255).permute(2, 0, 1)[0]
        tp = torch.from_numpy(np.asarray(tpl, np.float32) / 255).permute(2, 0, 1)[0]
        m = {"ssim_gpu_vs_cpu": round(ssim_torch(tt, tc), 4),
             "psnr_gpu_vs_cpu": round(psnr_torch(tt, tc), 2),
             "ssim_gpu_vs_cycles_template": round(ssim_torch(tt, tp), 4),
             "ssim_cpu_vs_cycles_template": round(ssim_torch(tc, tp), 4)}
        print("METRICS", az, json.dumps(m))
        with open(f"{OUT}/gpu_metrics.json", "w") as f:
            json.dump(m, f, indent=1)
    print("torchvision:", TV)
