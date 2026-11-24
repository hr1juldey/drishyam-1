"""Diagnostic: raw greedy predictions vs gold on 5 embedding rows.
Answers the question: is the eval decode path broken, or is the model
genuinely not exact-matching yet?"""
import os, sys, json
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
IMG = 256


def main():
    ckpt = os.path.join(HERE, "..", "..", "runs", "img-e2", "final")
    from transformers import T5Gemma2ForConditionalGeneration, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = T5Gemma2ForConditionalGeneration.from_pretrained(
        ckpt, dtype=torch.bfloat16).cuda().eval()
    emb = np.load(os.path.join(HERE, "embeddings", "emb_000000.npy"), mmap_mode="r")
    rows = [json.loads(l) for l in open(os.path.join(HERE, "embeddings", "lab_000000.jsonl"))][:5]

    def answer(emb_row, q, max_new=16):
        e = torch.from_numpy(np.asarray(emb_row)[None]).to("cuda", dtype=torch.bfloat16)
        qe = tok(q, return_tensors="pt", add_special_tokens=False)
        q_ids = qe["input_ids"].cuda()
        te = model.get_input_embeddings()(q_ids)
        ee = torch.cat([e, te], dim=1)
        attn = torch.cat([torch.ones(1, IMG, dtype=torch.long, device="cuda"),
                          qe["attention_mask"].cuda()], dim=1)
        with torch.no_grad():
            enc = model.model.encoder.text_model(inputs_embeds=ee, attention_mask=attn)
            dec = torch.full((1, 1), tok.bos_token_id if tok.bos_token_id is not None else 2,
                             dtype=torch.long, device="cuda")
            past = None
            out_ids = []
            for _ in range(max_new):
                d = model.model.decoder(input_ids=dec, encoder_hidden_states=enc.last_hidden_state,
                                        encoder_attention_mask=attn, past_key_values=past,
                                        use_cache=True)
                past = d.past_key_values
                lg = model.lm_head(d.last_hidden_state[:, -1])
                sc = getattr(model.config.decoder, "final_logit_softcapping", None)
                if sc is not None:
                    lg = torch.tanh(lg / sc) * sc
                n = int(lg.argmax())
                if n == (tok.eos_token_id or 1):
                    break
                out_ids.append(n)
                dec = torch.tensor([[n]], device="cuda")
        return tok.decode(out_ids, skip_special_tokens=True).strip()

    print(f"pad={tok.pad_token_id} eos={tok.eos_token_id} "
          f"decoder_start={getattr(model.config, 'decoder_start_token_id', '?')} "
          f"softcap={getattr(model.config.decoder, 'final_logit_softcapping', '?')}", flush=True)
    for i, r in enumerate(rows):
        print(f"[{i}] Q: {r['q']!r}", flush=True)
        print(f"    GOLD: {r['a']!r}", flush=True)
        print(f"    PRED: {answer(emb[i], r['q'])!r}", flush=True)


if __name__ == "__main__":
    main()
