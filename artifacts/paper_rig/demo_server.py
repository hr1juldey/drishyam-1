"""Demo backend for the 270M document reader. Runs on the home 3060.
Endpoints:
  POST /read  {image: dataURL, q: question} -> {answer, confidence?, latency_ms}
  GET  /health -> {model: ckpt, families: [...], ready: bool}
Camera capture happens in the browser; this server only reads images.
Launch: uv run python artifacts/paper_rig/demo_server.py --port 8300
Tunnel: ngrok http 8300  (frontend on Vercel points here)."""
import os, sys, base64, argparse, time, io
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")

STATE = {"model": None, "proc": None, "ckpt": None, "families": [
    "retail_receipt", "bank_savings_statement", "bank_card_estatement",
    "electricity_bill", "telecom_invoice", "salary_slip", "table_sheet"]}


def load_model(ckpt):
    from transformers import T5Gemma2ForConditionalGeneration, AutoProcessor
    print(f"loading {ckpt} ...", flush=True)
    proc = AutoProcessor.from_pretrained(ckpt)
    model = T5Gemma2ForConditionalGeneration.from_pretrained(
        ckpt, dtype=torch.bfloat16).cuda().eval()
    STATE["model"] = model
    STATE["proc"] = proc
    STATE["ckpt"] = ckpt
    print("ready", flush=True)


def best_frame(frames):
    """frames: list of np grayscale images -> sharpest by Laplacian variance."""
    scores = []
    for f in frames:
        fx = np.diff(f.astype(np.float32), axis=1)[:, :-1] ** 2
        fy = np.diff(f.astype(np.float32), axis=0)[:-1, :] ** 2
        scores.append(float(fx.mean() + fy.mean()))
    return int(np.argmax(scores)), scores


def read_image(img_uint8, question, max_new=16):
    model, proc = STATE["model"], STATE["proc"]
    pil = Image.fromarray(img_uint8)
    enc = proc(images=pil, text=f"<start_of_image>{question}", return_tensors="pt")
    px = enc["pixel_values"].to(device="cuda", dtype=torch.bfloat16)
    ids = enc["input_ids"].cuda()
    mask = enc["attention_mask"].cuda()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(pixel_values=px, input_ids=ids, attention_mask=mask,
                             max_new_tokens=max_new, num_beams=1, do_sample=False)
    new = out[0][ids.shape[1]:]
    answer = proc.tokenizer.decode(new, skip_special_tokens=True).strip()
    return answer, round((time.time() - t0) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8300)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    a = ap.parse_args()
    ckpt = a.ckpt
    if not ckpt:
        cand = os.path.join(HERE, "..", "..", "runs", "img-e2", "final")
        ckpt = cand if os.path.exists(os.path.join(cand, "config.json")) else None
    if not ckpt:
        from pack_batch import _SNAP
        ckpt = _SNAP  # pretrained fallback; demo still works, untrained answers

    from flask import Flask, request, jsonify, send_from_directory
    app = Flask(__name__)

    @app.get("/")
    def index():
        return send_from_directory(os.path.join(HERE, "demo"), "index.html")

    @app.post("/read")
    def read_ep():
        data = request.get_json(force=True)
        q = data.get("q") or "What is the total amount?"
        raw = base64.b64decode(data["image"].split(",")[-1])
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
        arr = np.asarray(pil)
        if data.get("frames"):  # multiple candidate frames: pick sharpest
            frames = []
            for fr in data["frames"]:
                f = Image.open(io.BytesIO(base64.b64decode(fr.split(",")[-1]))).convert("L")
                frames.append(np.asarray(f.resize((256, 256))))
            idx, _ = best_frame(frames)
            pil = Image.open(io.BytesIO(base64.b64decode(data["frames"][idx].split(",")[-1]))).convert("RGB")
        img = np.asarray(pil)
        answer, ms = read_image(img, q)
        return jsonify({"answer": answer, "latency_ms": ms, "ckpt": os.path.basename(STATE["ckpt"])})

    @app.get("/health")
    def health():
        return jsonify({"ready": STATE["model"] is not None, "ckpt": STATE["ckpt"],
                        "families": STATE["families"]})

    load_model(ckpt)
    app.run(host=a.host, port=a.port, threaded=False)


if __name__ == "__main__":
    main()
