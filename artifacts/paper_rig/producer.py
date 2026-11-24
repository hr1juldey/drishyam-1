"""Producer thread: manifest rows -> packed micro-batches on GPU (side stream).
CPU: PIL page + degradation + tokenizer packing; GPU: doc-photo render + 896px
downsample into bf16 pixel batch. One family per slot (manifest is family-blocked).
Determinism: gen_page + render_batch are seeded per row/seed (provenance law)."""
import os, sys, time, threading, importlib
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_gpu = importlib.import_module("2d_gpu")
_gpu.TD = torch.float32
GPURenderer = _gpu.GPURenderer
from jit_pipeline import gen_page, GEOM, DOC, AZ                  # noqa: E402
from row_labels import build_qa                                   # noqa: E402
from pack_batch import get_processor, _SNAP                       # noqa: E402

class JITProducer(threading.Thread):
    def __init__(self, rows, micro=8):
        super().__init__(daemon=True)
        self.rows, self.micro = rows, micro
        self.renders, self.i = {}, 0
        self.stop = threading.Event()
        self.queue, self.cv = [], threading.Condition()
        self.gen_ms, self.pack_ms, self.render_ms = [], [], []
        from transformers import AutoProcessor
        self.proc = AutoProcessor.from_pretrained(_SNAP)
        self.pad_id = self.proc.tokenizer.pad_token_id or 0

    def _renderer(self, fam):
        if fam not in self.renders:
            g = GEOM[fam]
            self.renders[fam] = GPURenderer(th=896, tw=896, atlas_pages=[
                np.full((g["ph"], g["pw"], 3), 250.0)], **g)
        return self.renders[fam]

    def get(self, timeout=None):
        with self.cv:
            if not self.cv.wait_for(lambda: len(self.queue) > 0, timeout):
                raise TimeoutError("producer starved")
            return self.queue.pop(0)

    def run(self):
        img_tok = json_image_token()
        stream = torch.cuda.Stream()
        while not self.stop.is_set():
            i = self.i
            if i + self.micro > len(self.rows):
                break
            chunk = self.rows[i:i + self.micro]
            fam = chunk[0]["family"]
            t0 = time.time()
            pages, gts = [], []
            for r in chunk:
                pg, gt = gen_page(fam, r["seed"])
                pages.append(pg); gts.append(gt)
            t_gen = (time.time() - t0) * 1000

            t0 = time.time()
            qas = [build_qa(fam, gt, r["seed"])[0] for r, gt in zip(chunk, gts)]
            texts = [f"<start_of_image>{qa['q']}" for qa in qas]
            enc = self.proc(images=[[p.astype(np.uint8)] for p in pages], text=texts,
                            return_tensors="pt", padding=True)
            ans = [self.proc.tokenizer(qa["a"] + self.proc.tokenizer.eos_token,
                                       add_special_tokens=False)["input_ids"] for qa in qas]
            L_ans = max(len(a) for a in ans) + 2               # start + answer + eos
            labels = torch.full((len(chunk), L_ans), -100, dtype=torch.long)
            dec_in = torch.full((len(chunk), L_ans), self.pad_id, dtype=torch.long)
            dmask = torch.zeros((len(chunk), L_ans), dtype=torch.long)
            for j, a in enumerate(ans):
                seq = torch.tensor(a + [self.proc.tokenizer.eos_token_id])
                labels[j, :len(seq)] = seq
                dec_in[j, 0] = self.pad_id                     # decoder start token
                dec_in[j, 1:1 + len(seq)] = seq
                dmask[j, :1 + len(seq)] = 1
            t_pack = (time.time() - t0) * 1000

            t0 = time.time()
            rend = self._renderer(fam)
            with torch.cuda.stream(stream):
                rend.set_atlas(pages)
                out, _a = rend.render_batch(len(chunk), AZ[chunk[0]["seed"] % 3], None,
                                            content_idx=[0] * len(chunk), **DOC)
                px = F.interpolate(out, size=(896, 896), mode="bilinear",
                                   antialias=True).to(torch.bfloat16)
            torch.cuda.synchronize()
            t_render = (time.time() - t0) * 1000

            slot = {"family": fam, "pixel_values": px.to("cuda"),
                    "input_ids": enc["input_ids"].to("cuda"),
                    "attention_mask": enc["attention_mask"].to("cuda"),
                    "dec_in": dec_in.to("cuda"), "dmask": dmask.to("cuda"),
                    "labels": labels.to("cuda"),
                    "t_gen": t_gen, "t_pack": t_pack, "t_render": t_render,
                    "row_ids": [r["row_id"] for r in chunk]}
            with self.cv:
                self.queue.append(slot)
                self.cv.notify()
            self.gen_ms.append(t_gen); self.pack_ms.append(t_pack); self.render_ms.append(t_render)
            self.i += self.micro

def json_image_token():
    import json
    cfg = json.load(open(os.path.join(_SNAP, "config.json")))
    return cfg["image_token_index"]
