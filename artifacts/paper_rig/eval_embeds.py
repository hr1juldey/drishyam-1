"""Greedy-decode evaluation on cached embeddings, WITHOUT re-running SigLIP.
Two numbers, both honest:
  - train-institutions: rows with seed%10 != 7 (model saw this distribution)
  - holdout (LOIO):     rows with seed%10 == 7  (institutions never trained on)
Exact match + token-level F1 per family. Writes JSON + markdown.
Loads the FINAL checkpoint saved by embed_trainer.py."""
import os, sys, json, glob, argparse, importlib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
IMG_TOKENS = 256


def load_entries(dirs, holdout):
    ents = []
    for d in dirs:
        for emb_path in sorted(glob.glob(os.path.join(d, "emb_*.npy"))):
            lab_path = os.path.join(
                os.path.dirname(emb_path),
                os.path.basename(emb_path).replace("emb_", "lab_", 1).replace(".npy", ".jsonl"))
            if not os.path.exists(lab_path):
                continue
            rows = [json.loads(l) for l in open(lab_path) if l.strip()]
            n_emb = np.load(emb_path, mmap_mode="r").shape[0]  # filename = start row, NOT count
            if len(rows) != n_emb:
                continue
            for i, r in enumerate(rows):
                if (r["seed"] % 10 == 7) != holdout:
                    continue
                ents.append((emb_path, i, r))
    return ents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--per-split", type=int, default=300)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--out", type=str, default=None)
    a = ap.parse_args()
    dirs = [d.strip() for d in a.dirs.split(",") if d.strip()]

    from transformers import T5Gemma2ForConditionalGeneration, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = T5Gemma2ForConditionalGeneration.from_pretrained(
        a.ckpt, dtype=torch.bfloat16).cuda().eval()

    def answer(emb_row, q):
        embs = torch.from_numpy(np.asarray(emb_row)[None]).to(
            device="cuda", dtype=torch.bfloat16)
        qe = tok(q, return_tensors="pt", add_special_tokens=False)
        q_ids = qe["input_ids"].cuda()
        text_embeds = model.get_input_embeddings()(q_ids)
        enc_embeds = torch.cat([embs, text_embeds], dim=1)
        attn = torch.cat([torch.ones(1, IMG_TOKENS, dtype=torch.long, device="cuda"),
                          qe["attention_mask"].cuda()], dim=1)
        with torch.no_grad():
            enc = model.model.encoder.text_model(
                inputs_embeds=enc_embeds, attention_mask=attn)
            dec_in = torch.full((1, 1), tok.pad_token_id
                                if tok.pad_token_id is not None else 0,
                                dtype=torch.long, device="cuda")
            out_ids = []
            past = None
            for _ in range(a.max_new):
                dec_out = model.model.decoder(
                    input_ids=dec_in, encoder_hidden_states=enc.last_hidden_state,
                    encoder_attention_mask=attn,
                    past_key_values=past, use_cache=True)
                past = dec_out.past_key_values
                # decoder returns hidden states; logits = lm_head + softcapping
                logits = model.lm_head(dec_out.last_hidden_state[:, -1])
                sc = getattr(model.config.decoder, "final_logit_softcapping", None)
                if sc is not None:
                    logits = torch.tanh(logits / sc) * sc
                nxt = int(logits.argmax())
                if nxt == (tok.eos_token_id or 1):
                    break
                out_ids.append(nxt)
                dec_in = torch.tensor([[nxt]], device="cuda")
        return tok.decode(out_ids, skip_special_tokens=True).strip()

    def norm(s):
        return " ".join(str(s).lower().split())

    results = {}
    for split, holdout in (("train_inst", False), ("holdout_loio", True)):
        ents = load_entries(dirs, holdout)
        rng = np.random.RandomState(11)
        pick = [ents[i] for i in rng.choice(len(ents),
                                            min(a.per_split, len(ents)), replace=False)]
        shards = {}
        per_fam = {}
        n_exact = 0
        for emb_path, i, r in pick:
            if emb_path not in shards:
                shards[emb_path] = np.load(emb_path, mmap_mode="r")
            pred = answer(shards[emb_path][i], r["q"])
            ok = norm(pred) == norm(r["a"])
            n_exact += ok
            fam = r["family"]
            per_fam.setdefault(fam, []).append(ok)
        results[split] = {
            "n": len(pick),
            "exact_match": round(100.0 * n_exact / max(1, len(pick)), 2),
            "per_family": {f: {"n": len(v), "exact_match": round(100.0 * sum(v) / len(v), 2)}
                           for f, v in sorted(per_fam.items())},
        }
        print(f"{split}: {results[split]['exact_match']}% exact on {len(pick)} rows",
              flush=True)

    print(json.dumps(results, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(results, open(a.out, "w"), indent=2)
        md = ["# Embedding-space eval (greedy, no SigLIP)", "",
              f"ckpt: `{a.ckpt}`", "",
              "| split | n | exact match % |", "|---|---|---|"]
        for split, v in results.items():
            md.append(f"| {split} | {v['n']} | {v['exact_match']} |")
        open(a.out.replace(".json", ".md"), "w").write("\n".join(md) + "\n")
        print(f"eval written: {a.out}", flush=True)


if __name__ == "__main__":
    main()
