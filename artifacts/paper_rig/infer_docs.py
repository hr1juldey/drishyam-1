"""Use the model: python infer_docs.py --image form.png --q "What is the total?"
Full pipeline: image -> SigLIP -> projector -> (optionally trained) decoder.
Default checkpoint = the overnight runbook's runs/img-e2/final; override with --ckpt."""
import os, sys, argparse
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--q", type=str, required=True)
    ap.add_argument("--ckpt", type=str, default=None,
                    help="checkpoint dir (default: latest runs/*/final with a saved model)")
    a = ap.parse_args()

    ckpt = a.ckpt
    if not ckpt:
        runs = sorted(os.listdir(os.path.join(HERE, "..", "runs")),
                      reverse=True) if os.path.isdir(os.path.join(HERE, "..", "runs")) else []
        for r in runs:
            cand = os.path.join(HERE, "..", "runs", r, "final")
            if os.path.exists(os.path.join(cand, "config.json")):
                ckpt = cand
                break
    if not ckpt:
        from pack_batch import _SNAP
        ckpt = _SNAP
    print(f"ckpt: {ckpt}")

    from transformers import T5Gemma2ForConditionalGeneration, AutoProcessor
    proc = AutoProcessor.from_pretrained(ckpt)
    model = T5Gemma2ForConditionalGeneration.from_pretrained(
        ckpt, dtype=torch.bfloat16).cuda().eval()

    from PIL import Image
    img = Image.open(a.image).convert("RGB")
    enc = proc(images=img, text=f"<start_of_image>{a.q}", return_tensors="pt")
    px = enc["pixel_values"].to(device="cuda", dtype=torch.bfloat16)
    ids = enc["input_ids"].cuda()
    mask = enc["attention_mask"].cuda()
    with torch.no_grad():
        out = model.generate(pixel_values=px, input_ids=ids, attention_mask=mask,
                             max_new_tokens=16, num_beams=1, do_sample=False)
    new = out[0][ids.shape[1]:]
    print("Q:", a.q)
    print("A:", proc.tokenizer.decode(new, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
