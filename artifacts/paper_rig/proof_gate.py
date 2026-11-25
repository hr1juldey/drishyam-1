"""Pixel-path proof gate: render FRESH unseen-seed pages through the FULL
pipeline (pixels -> SigLIP -> projector -> trained decoder) and compare to
ground truth. This is the gate the embedding-space eval cannot replace.
Usage: uv run python artifacts/paper_rig/proof_gate.py --ckpt runs/img-e2/final
Writes runs/proof/PROOF.md + the page PNGs."""
import os, sys, json, argparse
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
_gpu = __import__("importlib").import_module("2d_gpu")
_gpu.TD = __import__("torch").float32

PROOF_SEEDS = [7777001, 7777002, 7777003, 7777004, 7777005, 7777006, 7777007]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "runs", "proof"))
    ap.add_argument("--per-page", type=int, default=3, help="questions per page")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    from pack_batch import render_896, get_processor
    from row_labels import build_qa
    from PIL import Image
    import torch
    from transformers import T5Gemma2ForConditionalGeneration

    fams = ["thermal", "savings", "card", "electricity", "telecom", "salary", "table"]
    proc = get_processor()
    model = T5Gemma2ForConditionalGeneration.from_pretrained(
        a.ckpt, dtype=torch.bfloat16).cuda().eval()

    def answer(pil_img, q):
        enc = proc(images=pil_img, text=f"<start_of_image>{q}", return_tensors="pt")
        px = enc["pixel_values"].to(device="cuda", dtype=torch.bfloat16)
        ids = enc["input_ids"].cuda()
        mask = enc["attention_mask"].cuda()
        with torch.no_grad():
            out = model.generate(pixel_values=px, input_ids=ids, attention_mask=mask,
                                 max_new_tokens=16, num_beams=1, do_sample=False)
        return proc.tokenizer.decode(out[0][ids.shape[1]:],
                                     skip_special_tokens=True).strip()

    def norm(s):
        return " ".join(str(s).lower().split())

    results = []
    n_exact = n_tot = 0
    per_fam = {}
    for seed in PROOF_SEEDS:
        fam = fams[seed % len(fams)]
        img_u8, gt = render_896(fam, seed)
        png = os.path.join(a.out, f"proof_{fam}_{seed}.png")
        Image.fromarray(img_u8).save(png)
        qas = build_qa(fam, gt, seed)[: a.per_page]
        for qa in qas:
            pred = answer(Image.fromarray(img_u8), qa["q"])
            ok = norm(pred) == norm(qa["a"])
            n_tot += 1
            n_exact += ok
            per_fam.setdefault(fam, []).append(ok)
            results.append({"seed": seed, "family": fam, "q": qa["q"],
                            "gold": qa["a"], "pred": pred, "exact": ok})
            print(f"[{fam} {seed}] Q {qa['q']!r} GOLD {qa['a']!r} PRED {pred!r} "
                  f"{'OK' if ok else 'MISS'}", flush=True)

    md = ["# Pixel-path proof gate", "",
          f"ckpt `{a.ckpt}` · {len(PROOF_SEEDS)} fresh unseen seeds · full pixel pipeline", "",
          "| family | seed | question | gold | predicted | exact |", "|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['family']} | {r['seed']} | {r['q']} | {r['gold']} | {r['pred']} "
                  f"| {'✅' if r['exact'] else '❌'} |")
    md += ["", f"**exact match: {n_exact}/{n_tot} = {100.0*n_exact/max(1,n_tot):.1f}%**",
           "", "Per-family: " + json.dumps(
               {f: f"{sum(v)}/{len(v)}" for f, v in sorted(per_fam.items())})]
    open(os.path.join(a.out, "PROOF.md"), "w").write("\n".join(md) + "\n")
    json.dump(results, open(os.path.join(a.out, "proof.json"), "w"), indent=2)
    print(f"PROOF: {n_exact}/{n_tot} exact -> {a.out}/PROOF.md", flush=True)


if __name__ == "__main__":
    main()
