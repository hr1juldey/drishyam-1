"""Tokenization + padded fixed-shape collation (the cross-question invariance mechanism)."""

import torch

SLOT = "<slot>"


def add_slot_token(tok, model) -> int:
    """Add the dedicated <slot> token; resize tied embeddings BEFORE LoRA attach (DECISIONS §3)."""
    tok.add_tokens([SLOT], special_tokens=True)
    slot_id = tok.convert_tokens_to_ids(SLOT)
    model.resize_token_embeddings(len(tok))
    inp, out = model.get_input_embeddings(), model.get_output_embeddings()
    if out is not None and out.weight.data_ptr() != inp.weight.data_ptr():
        out.weight = inp.weight  # re-tie if resize broke the shared tensor
    return slot_id


def question_text(q) -> str:
    opts = " <slot> ".join(q.options) + " <slot>"  # slot AFTER its option text (spec §2)
    return f"Question: {q.text} | Type: {q.primitive.value} | Options: {opts}"


def collate_questions(tok, questions, fixed_len: int, device="cpu"):
    """Every decoder row padded to the SAME fixed_len, so 'Q1 alone' and 'Q1 in batch'
    launch identical kernels -> exact invariance by construction (DECISIONS §4)."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    slot_id = tok.convert_tokens_to_ids(SLOT)
    raws = [tok(question_text(q), add_special_tokens=True)["input_ids"] for q in questions]
    dec_ids = torch.tensor([r + [pad] * (fixed_len - len(r)) for r in raws], device=device)
    dec_mask = torch.tensor(
        [[1] * len(r) + [0] * (fixed_len - len(r)) for r in raws], device=device
    )
    slot_pos = [(torch.tensor(r, device=device) == slot_id).nonzero().flatten() for r in raws]
    n_slots = [len(p) for p in slot_pos]
    return dec_ids, dec_mask, slot_pos, n_slots
