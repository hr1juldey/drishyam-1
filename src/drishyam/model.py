"""G1-M1: DecoupledDecision — state-only encoder cache + parallel slot decoder + slot head."""

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from drishyam.tokenize import add_slot_token, question_text  # noqa: F401 — re-export (tests/eval import from here)


def build_backbone(model_id: str, torch_dtype=torch.bfloat16):
    try:
        from transformers import AutoModelForSeq2SeqLM

        return AutoModelForSeq2SeqLM.from_pretrained(model_id, dtype=torch_dtype)
    except (ValueError, KeyError, OSError):
        from transformers import T5Gemma2ForConditionalGeneration

        return T5Gemma2ForConditionalGeneration.from_pretrained(model_id, dtype=torch_dtype)


def attach_lora(bb):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
        ],
        # peft>=0.14; if TypeError: drop the arg (vision adapters would get no gradient)
        exclude_modules=r".*vision.*",
    )
    return get_peft_model(bb, cfg)


class SlotHead(nn.Module):
    """LayerNorm -> 640 -> 320 -> 1: one scalar logit per option (spec §2)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, 320), nn.GELU(), nn.Linear(320, 1)
        )

    def forward(self, h):
        return self.net(h).squeeze(-1)  # [..., n_slots]


class DecoupledDecision(nn.Module):
    def __init__(self, backbone, slot_id: int):
        super().__init__()
        self.bb, self.slot_id = backbone, slot_id
        cfg = backbone.config
        d = None
        for src in (getattr(cfg, "text_config", None), cfg,
                    getattr(cfg, "decoder", None), getattr(cfg, "encoder", None)):
            if src is None:
                continue
            for attr in ("d_model", "hidden_size"):
                v = src.get(attr) if isinstance(src, dict) else getattr(src, attr, None)
                if v:
                    d = int(v)
                    break
            if d:
                break
        if not d:
            raise RuntimeError("cannot locate model dim in config (d_model/hidden_size)")
        self.head = SlotHead(d)
        import inspect

        self._can_limit_logits = "logits_to_keep" in inspect.signature(backbone.forward).parameters

    def encode_state(self, state_ids, state_mask):
        enc = self.bb.get_encoder()(input_ids=state_ids, attention_mask=state_mask)
        return enc.last_hidden_state  # [B, L_state, d] — THE cache

    def forward(self, state_ids, state_mask, dec_ids, dec_mask, slot_pos_list,
                row_case=None, blank_state=False):
        h_enc = self.encode_state(state_ids, state_mask)
        if blank_state:
            h_enc = torch.zeros_like(h_enc)  # anti-dead-weight ablation (spec §4.3)
        n_dec = dec_ids.size(0)
        # transformers 5 mask builders (create_bidirectional_mask) vmap-index the padding
        # mask by decoder batch — stride-0 expand() views die on device-side asserts, so
        # masks/hidden must be MATERIALIZED to the decoder batch, never shipped as views.
        if row_case is not None:
            h_enc = h_enc[row_case]
            state_mask = state_mask[row_case]
        else:
            h_enc = h_enc.repeat(n_dec, 1, 1)
            state_mask = state_mask.repeat(n_dec, 1)
        # output_hidden_states=True: Seq2SeqLMOutput has no decoder_last_hidden_state field,
        # so the last decoder layer comes from decoder_hidden_states[-1] (pre-final-LN — the
        # head's own LayerNorm absorbs the difference, and both train/eval use the same path).
        # logits_to_keep=0 skips the vocab-wide lm_head matmul (~GBs saved) when supported.
        kwargs: dict = {"output_hidden_states": True, "use_cache": False}
        if self._can_limit_logits:
            kwargs["logits_to_keep"] = 0
        # transformers 5.x requires encoder_outputs as a ModelOutput (tuple form no longer accepted)
        out = self.bb(
            decoder_input_ids=dec_ids,
            decoder_attention_mask=dec_mask,
            encoder_outputs=BaseModelOutput(last_hidden_state=h_enc),
            attention_mask=state_mask,
            **kwargs,
        )
        h = out.decoder_hidden_states[-1].float()  # head stays fp32 for calibration-grade numerics
        return [self.head(h[i, p]) for i, p in enumerate(slot_pos_list)]  # per-question [n_slots_i]

    def encode_state_embeds(self, page_embs, page_mask):
        """Gate-2: cached page tokens [B, 256, d] -> h_enc, world knowledge untouched.

        Encoder path: get_encoder()(inputs_embeds=...) works on plain T5 (t5-small CPU
        tests); T5Gemma2's verified path (eval_metrics.py / proof gate) is
        model.model.encoder.text_model(inputs_embeds=...) — try/except resolves per model."""
        embs = page_embs.to(next(self.bb.parameters()).dtype)
        try:
            enc = self.bb.get_encoder()(inputs_embeds=embs, attention_mask=page_mask)
        except (TypeError, AttributeError):
            enc = self.bb.model.encoder.text_model(inputs_embeds=embs,
                                                   attention_mask=page_mask)
        return enc.last_hidden_state

    def forward_embeds(self, page_embs, page_mask, dec_ids, dec_mask, slot_pos_list,
                       row_case=None, blank_state=False):
        """forward() with the state side swapped to cached page embeddings. The decoder
        side (slot collation, fp32 head, logits_to_keep, blank ablation, row_case
        materialization) is byte-identical to forward()."""
        h_enc = self.encode_state_embeds(page_embs, page_mask)
        if blank_state:
            h_enc = torch.zeros_like(h_enc)
        n_dec = dec_ids.size(0)
        if row_case is not None:
            h_enc, page_mask = h_enc[row_case], page_mask[row_case]
        else:
            h_enc = h_enc.repeat(n_dec, 1, 1)
            page_mask = page_mask.repeat(n_dec, 1)
        kwargs: dict = {"output_hidden_states": True, "use_cache": False}
        if self._can_limit_logits:
            kwargs["logits_to_keep"] = 0
        out = self.bb(
            decoder_input_ids=dec_ids,
            decoder_attention_mask=dec_mask,
            encoder_outputs=BaseModelOutput(last_hidden_state=h_enc),
            attention_mask=page_mask,
            **kwargs,
        )
        h = out.decoder_hidden_states[-1].float()
        return [self.head(h[i, p]) for i, p in enumerate(slot_pos_list)]


def trainable_groups(model, wd: float = 0.01):
    """LoRA + slot head train fully; weight decay skips norms and biases."""
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 or "norm" in n.lower() else decay).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
