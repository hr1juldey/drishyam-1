#!/usr/bin/env python3
"""Readable training row: real-bill content -> GPU paper pipeline.
Generates one savings-statement page + one card e-statement page (content_generator),
pushes them through GPURenderer as the atlas (photo-look: stack, light, shadow, noise),
and saves side-by-side [generated flat page | photographed-paper row] + row recipes.
"""
import os, sys, json
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from content_generator import gen_statement, gen_card   # noqa: E402
import importlib                                        # noqa: E402
import torch                                            # noqa: E402
_gpu = importlib.import_module("2d_gpu")
_gpu.TD = torch.float32   # fp32 render: bf16 grid quantization destroys small text (proven by probe)
GPURenderer, to_img = _gpu.GPURenderer, _gpu.to_img

rng = np.random.RandomState(21)
pages, gts = [], {}
for fam, fn in (("savings", gen_statement), ("card", gen_card)):
    page, gt = fn(rng)
    pages.append(page)
    gts[fam] = gt

R = GPURenderer(th=1080, tw=1080, ph=1000, pw=707, atlas_pages=pages)  # doc-photo framing ~93%
class G:
    def seed_drawn(self, a, b): return int(rng.randint(a, b))
g = G()
rows = {}
idx = {"savings": 0, "card": 1}
DOC = dict(n_lo=2, n_hi=3, amp_lo=0.3, amp_hi=0.8, drift_sigma=0.6,
           cdepth_lo=1, cdepth_hi=2, field_strength=0.07, br_lo=0.94, br_hi=0.97,
           noise_sigma=1.0)
for i, (fam, az) in enumerate((("savings", "light_left"), ("card", "light_right"))):
    out, _a = R.render_batch(2, az, g, content_idx=[idx[fam], idx[fam]], **DOC)
    row = to_img(out[0])
    Image.fromarray(row).save(f"{HERE}/paper_row_{fam}.png")
    rows[fam] = row

REAL = {"savings": "/home/riju279/.hermes/cache/documents/native-inputs/16d3056cac04c8192f22ad8d812c769227868138b6de38640fce87f6002ef412/149452fb-9f48-4531-a681-f7e5ca3e579d.png",
        "card": "/home/riju279/.hermes/cache/documents/native-inputs/29afdb69e7ac6c90f29fc63c2aa53841dd08e4e0f19a49416fdf2f46f4de5f39/f2d02cdf-54c8-42d0-9046-b83afd13b131.png"}
for fam in ("savings", "card"):
    rw = Image.open(REAL[fam]).convert("RGB")
    mw = Image.fromarray(np.clip(pages[idx[fam]], 0, 255).astype(np.uint8))
    W = 880
    rh = int(rw.height * W / rw.width); mh = int(mw.height * W / mw.width)
    H = max(rh, mh)
    cmp_img = Image.new("RGB", (W * 2 + 16, H), (25, 25, 28))
    cmp_img.paste(rw.resize((W, rh)), (0, 0))
    cmp_img.paste(mw.resize((W, mh)), (W + 16, 0))
    cmp_img.save(f"{HERE}/equal_scale_{fam}.png")

# side-by-side: [flat generated page | photographed row]
strips = {}
for fam in ("savings", "card"):
    flat = Image.fromarray(np.clip(pages[idx[fam]], 0, 255).astype(np.uint8)).resize((763, 1080))
    strip = Image.new("RGB", (763 + 20 + 1080, 1080), (10, 10, 10))
    strip.paste(flat, (0, 0))
    strip.paste(Image.fromarray(rows[fam]), (783, 0))
    strip.save(f"{HERE}/readable_row_{fam}.png")
    strips[fam] = f"{HERE}/readable_row_{fam}.png"

# row recipe: content ground truth + degradation annotation (what the factory stores)
recipes = {fam: {"content": gts[fam],
                 "paper": {"family": "loose_a4_stack", "azimuth": az, "seed_row": int(rng.randint(0, 2**31)),
                           "crumple": "medium", "shadow": "soft_orbit"},
                 "row_image": f"paper_row_{fam}.png"}
           for fam, az in (("savings", "light_left"), ("card", "light_right"))}
with open(f"{HERE}/row_recipes.json", "w") as f:
    json.dump(recipes, f, indent=1)

for fam in ("savings", "card"):
    g_ = gts[fam]
    keys = ["account_number", "n_transactions"] if fam == "savings" else \
           ["customer_name", "card_masked", "total_amount_due", "min_amount_due",
            "due_date", "n_transactions", "invoice_no"]
    print(fam.upper(), json.dumps({k: g_[k] for k in keys}))
print("STRIPS", json.dumps(strips))
