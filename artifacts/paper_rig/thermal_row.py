#!/usr/bin/env python3
"""Thermal receipt through the GPU photo pipeline: clean + degraded rows."""
import os, sys, json
import numpy as np
from PIL import Image
import importlib, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from thermal_generator import gen_thermal, apply_degradation
_gpu = importlib.import_module("2d_gpu")
_gpu.TD = torch.float32                       # fp32: text-carrying texture
GPURenderer, to_img = _gpu.GPURenderer, _gpu.to_img

rng = np.random.RandomState(9)
seed = 777
page, gt = gen_thermal(rng, seed)
gt["degradation_plan"] = {"total_amt": "smudge_heavy", "date": "fade",
                          "item_1_name": "smudge_light", "gstin": "thermal_band"}
deg, applied = apply_degradation(page, gt, np.random.RandomState(seed + 1))
gt["legibility_applied"] = applied
json.dump(gt, open(f"{HERE}/thermal_gt.json", "w"), indent=1)

R = GPURenderer(th=1080, tw=1080, ph=1000, pw=334, atlas_pages=[deg])   # narrow tall receipt
class G:
    def seed_drawn(self, a, b): return int(rng.randint(a, b))
g = G()
DOC = dict(n_lo=2, n_hi=3, amp_lo=0.3, amp_hi=0.8, drift_sigma=0.6,
           cdepth_lo=1, cdepth_hi=2, field_strength=0.07, br_lo=0.94, br_hi=0.97,
           noise_sigma=1.0)
out, _a = R.render_batch(1, "light_right", g, content_idx=[0], **DOC)
row = to_img(out[0])
Image.fromarray(row).save(f"{HERE}/paper_row_thermal.png")
strip = Image.new("RGB", (354 + 1080, 1080), (10, 10, 10))
strip.paste(Image.fromarray(np.clip(deg, 0, 255).astype(np.uint8)).resize((334, 1000)), (0, 0))
strip.paste(Image.fromarray(row), (354, 0))
strip.save(f"{HERE}/thermal_row_strip.png")
print("APPLIED", json.dumps(applied))
print("thermal row saved; total in gt:", gt["total"], "legibility:", json.dumps(applied))
