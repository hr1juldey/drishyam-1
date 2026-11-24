"""seed -> fully packed training example: rendered 896px pixels + input_ids with
256 image placeholders + labels masked to the short answer. Deterministic in
(family, seed). The batched producer path is producer.py; this module is the
single-row reference used by tests and eval."""
import os, sys, json, importlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                                # noqa: E402
_gpu = importlib.import_module("2d_gpu")
_gpu.TD = torch.float32                                     # text textures: fp32 only
from jit_pipeline import gen_page, GEOM, DOC, AZ            # noqa: E402
from row_labels import build_qa                             # noqa: E402
import importlib as _il                                     # noqa: E402
GPURenderer = _il.import_module("2d_gpu").GPURenderer

_SNAP = os.path.expanduser("~/.cache/huggingface/hub/models--google--t5gemma-2-270m-270m/snapshots/7c38f16641f455ef0685b18431faf1b17722d5a1")
_PROC = None

def get_processor():
    global _PROC
    if _PROC is None:
        from transformers import AutoProcessor
        _PROC = AutoProcessor.from_pretrained(_SNAP)
    return _PROC

def model_image_token_id():
    cfg = json.load(open(os.path.join(_SNAP, "config.json")))
    v = cfg.get("image_token_index")
    if v is None:
        raise RuntimeError("image_token_index missing from model config.json — do NOT guess it")
    return v

def _rend(fam, cache={}):
    if fam not in cache:
        g = GEOM[fam]
        cache[fam] = GPURenderer(th=896, tw=896, atlas_pages=[
            np.full((g["ph"], g["pw"], 3), 250.0)], **g)
    return cache[fam]

class _G:
    def __init__(self, seed): self.s = seed
    def seed_drawn(self, a, b): return int(np.random.RandomState(self.s).randint(a, b))

def render_896(family, seed):
    """row -> (uint8 HWC 896x896 photo-look image, gt dict)"""
    page, gt = gen_page(family, seed)
    rend = _rend(family)
    rend.set_atlas([page])
    out, _a = rend.render_batch(1, AZ[seed % 3], _G(seed), content_idx=[0], seed=seed, **DOC)
    img = out[0].permute(1, 2, 0).float().cpu().numpy()
    return np.clip(img, 0, 255).astype(np.uint8), gt

def build_example(family, seed, proc):
    img, gt = render_896(family, seed)
    qa = build_qa(family, gt, seed)[0]
    enc = proc(images=img, text=f"<start_of_image>{qa['q']}", return_tensors="pt")
    input_ids = enc["input_ids"][0].tolist()
    ans_ids = proc.tokenizer(qa["a"] + proc.tokenizer.eos_token,
                             add_special_tokens=False)["input_ids"]
    labels = [-100] * len(input_ids) + ans_ids
    return {"pixel_values": enc["pixel_values"][0].numpy().astype(np.float32),
            "input_ids": input_ids, "labels": labels,
            "answer_len": len(ans_ids), "image_token_id": model_image_token_id(),
            "qa": qa}
