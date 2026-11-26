"""text-row trainer mechanics on CPU (t5-small): fixed-shape state batching,
fail-fast question length, one real optimizer pass + grad-free val pass."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drishyam.model import DecoupledDecision, add_slot_token  # noqa: E402
from drishyam.schemas import Primitive, Question  # noqa: E402
from drishyam.train_text_row import (  # noqa: E402
    fixed_len_for_cases,
    run_rows,
    state_batch_ids,
)

STATE_LEN = 24
FIXED_LEN = 48


def tiny_setup():
    tok = AutoTokenizer.from_pretrained("t5-small")
    bb = T5ForConditionalGeneration.from_pretrained("t5-small")
    slot_id = add_slot_token(tok, bb)
    return tok, DecoupledDecision(bb, slot_id)


def tiny_cases():
    def q(qid, prim, text, options, idx):
        return (
            Question(qid=qid, primitive=Primitive(prim), text=text, options=options),
            [1.0 if i == idx else 0.0 for i in range(len(options))],
        )

    cases = []
    for i, state in enumerate(
        [
            "Invoice total is Rs 1,234.50, due 12 May.",
            "The package shipped on 3 September via DHL.",
            "Review says the screen arrived cracked.",
        ]
    ):
        qs, golds = zip(
            q(f"{i}-n", "noul", "Is it paid?", ["No", "Yes"], i % 2),
            q(
                f"{i}-c",
                "choice",
                "Which value is printed?",
                ["Rs 1.00", "Rs 2.00", "Rs 3.00"],
                i % 3,
            ),
            strict=False,
        )
        cases.append(
            {
                "case_id": f"bev:test:train:{i}",
                "state": state,
                "domain": "test",
                "questions": list(qs),
                "gold": list(golds),
            }
        )
    return cases


def test_state_batch_ids_fixed_shape_and_truncation():
    tok, _ = tiny_setup()
    short, long_state = "Order #1.", "word " * 200
    ids, mask = state_batch_ids(tok, [short, long_state], STATE_LEN)
    assert ids.shape == (2, STATE_LEN) and mask.shape == (2, STATE_LEN)
    n_short = len(tok(short, add_special_tokens=True)["input_ids"])
    assert mask[0].tolist() == [1] * n_short + [0] * (STATE_LEN - n_short)
    assert mask[1].tolist() == [1] * STATE_LEN  # truncated, no padding
    ref = torch.tensor(
        tok(long_state, add_special_tokens=True, truncation=True, max_length=STATE_LEN)[
            "input_ids"
        ]
    )
    assert torch.equal(ids[1], ref)


def test_fixed_len_for_cases_cap_is_fail_fast():
    tok, _ = tiny_setup()
    cases = tiny_cases()
    fixed_len = fixed_len_for_cases(tok, cases, cap=192)
    assert 8 <= fixed_len <= 192
    with pytest.raises(ValueError, match="question-len-cap"):
        fixed_len_for_cases(tok, cases, cap=8)


def test_run_rows_train_and_val_cpu():
    torch.manual_seed(0)
    tok, model = tiny_setup()
    cases = tiny_cases()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss1, (p1, g1) = run_rows(
        model, tok, cases, "cpu", STATE_LEN, FIXED_LEN, opt=opt, rows_per_batch=2
    )
    loss2, _ = run_rows(model, tok, cases, "cpu", STATE_LEN, FIXED_LEN, opt=opt, rows_per_batch=2)
    assert np.isfinite(loss1) and np.isfinite(loss2) and loss1 > 0
    assert len(p1) == 6 and len(g1) == 6  # 3 cases x 2 questions, shuffled
    for p, g in zip(p1, g1, strict=False):
        assert p.shape == g.shape and abs(float(g.sum()) - 1.0) < 1e-9
        assert np.isfinite(p).all() and abs(float(p.sum()) - 1.0) < 1e-5
    vloss, (vp, vg) = run_rows(model, tok, cases, "cpu", STATE_LEN, FIXED_LEN)
    assert np.isfinite(vloss) and len(vp) == len(vg) == 6
    # a single-case pass still yields every question row of that case
    solo = run_rows(model, tok, [cases[0]], "cpu", STATE_LEN, FIXED_LEN, rows_per_batch=1)[1][0]
    assert len(solo) == 2


def test_run_rows_batched_matches_solo():
    """Question logits must not depend on which states share the batch."""
    torch.manual_seed(0)
    tok, model = tiny_setup()
    cases = tiny_cases()
    model.eval()
    fixed_len = fixed_len_for_cases(tok, cases, cap=192)
    with torch.inference_mode():
        solo = run_rows(model, tok, cases[:1], "cpu", STATE_LEN, fixed_len, rows_per_batch=2)[1][0]
        batched = run_rows(model, tok, cases, "cpu", STATE_LEN, fixed_len, rows_per_batch=6)[1][0]
    torch.testing.assert_close(batched[0], solo[0], rtol=0, atol=1e-5)
