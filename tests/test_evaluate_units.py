import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from drishyam.evaluate import flip_rate, max_prob_delta
from drishyam.model import DecoupledDecision
from drishyam.schemas import Primitive, Question
from drishyam.tokenize import add_slot_token, collate_questions


def test_flip_rate_hand_computed():
    a = [{"probs": np.array([0.6, 0.4])}, {"probs": np.array([0.3, 0.7])}]
    b = [{"probs": np.array([0.4, 0.6])}, {"probs": np.array([0.2, 0.8])}]
    assert flip_rate(a, b) == 0.5  # first flipped, second same


def test_max_prob_delta_ragged():
    a = [{"probs": np.array([0.5, 0.5])}, {"probs": np.array([0.2, 0.3, 0.5])}]
    b = [{"probs": np.array([0.5001, 0.4999])}, {"probs": np.array([0.2, 0.3, 0.5])}]
    assert abs(max_prob_delta(a, b) - 0.0001) < 1e-12


def test_blank_state_flag_shapes():
    """CPU wiring: blank_state must not change shapes and must change logits (t5-small)."""
    tok = AutoTokenizer.from_pretrained("t5-small")
    base = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    slot = add_slot_token(tok, base)
    dec = DecoupledDecision(base, slot)
    qs = [Question(qid="q", primitive=Primitive.NOUL, text="ok?", options=["No", "Yes"])]
    di, dm, pos, _ = collate_questions(tok, qs, 32, device="cpu")
    s = torch.tensor([[1, 2, 3]])
    sm = torch.ones_like(s)
    out_real = dec(s, sm, di, dm, pos)
    out_blank = dec(s, sm, di, dm, pos, blank_state=True)
    assert len(out_real) == len(out_blank) == 1
    assert out_real[0].shape == out_blank[0].shape == (2,)
    assert not torch.allclose(out_real[0], out_blank[0], atol=1e-6)
