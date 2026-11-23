import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from drishyam.model import DecoupledDecision, add_slot_token, question_text
from drishyam.schemas import Primitive, Question

FIXED_LEN = 48


def tiny_setup():
    tok = AutoTokenizer.from_pretrained("t5-small")
    bb = T5ForConditionalGeneration.from_pretrained("t5-small")
    slot_id = add_slot_token(tok, bb)
    return tok, bb, slot_id


def encode_q(tok, q: Question):
    raw = tok(question_text(q), add_special_tokens=True)["input_ids"]
    ids = raw + [tok.pad_token_id] * (FIXED_LEN - len(raw))
    mask = [1] * len(raw) + [0] * (FIXED_LEN - len(raw))
    return torch.tensor([ids]), torch.tensor([mask]), raw


def rows_of(m, tok, s_ids, s_mask, qs):
    enc = [tok(question_text(q), add_special_tokens=True)["input_ids"] for q in qs]
    dec_ids = torch.tensor([e + [tok.pad_token_id] * (FIXED_LEN - len(e)) for e in enc])
    dec_mask = torch.tensor([[1] * len(e) + [0] * (FIXED_LEN - len(e)) for e in enc])
    slot_pos = [(torch.tensor(e) == m.slot_id).nonzero().flatten() for e in enc]
    with torch.no_grad():
        return m(s_ids, s_mask, dec_ids, dec_mask, slot_pos)


def test_slot_shapes_and_invariance():
    torch.manual_seed(0)
    tok, bb, slot_id = tiny_setup()
    m = DecoupledDecision(bb, slot_id).eval()
    s_ids = torch.randint(3, 100, (1, 16))
    s_mask = torch.ones_like(s_ids)
    choice = Question(qid="q1", primitive=Primitive.CHOICE, text="Which action?",
                      options=["pay now", "hold", "reject"])
    noul = Question(qid="q2", primitive=Primitive.NOUL, text="Is it duplicate?",
                    options=["No", "Yes"])
    alone = rows_of(m, tok, s_ids, s_mask, [choice])
    batch = rows_of(m, tok, s_ids, s_mask, [choice, noul])
    reordered = rows_of(m, tok, s_ids, s_mask, [noul, choice])
    assert alone[0].shape == (3,)
    assert len(batch) == 2 and batch[0].shape == (3,) and batch[1].shape == (2,)
    assert torch.isfinite(alone[0]).all()
    # (a) identical kernel shapes, different row position: BITWISE equal (no cross-row leakage)
    assert torch.equal(reordered[1], batch[0]), "Q1 depends on its row position — cross-row leakage"
    # (b) batch-size change (1 vs 2 rows) = fp reduction-order physics: bounded noise only
    torch.testing.assert_close(batch[0], alone[0], rtol=0, atol=1e-6)
    # (c) and the gate's actual metric: delta vanishes at 4 decimal places (DECISIONS §4)
    assert (batch[0].round(decimals=6) == alone[0].round(decimals=6)).all(), \
        "delta above gate sensitivity"


def test_question_text_format():
    q = Question(qid="q", primitive=Primitive.CHOICE, text="Which action?",
                 options=["pay now", "hold", "reject"])
    assert question_text(q) == (
        "Question: Which action? | Type: choice | Options: pay now <slot> hold <slot> reject <slot>"
    )
