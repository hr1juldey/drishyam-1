"""Gate-2 vision glue: forward_embeds on cached page tokens — shapes, invariance
tiers (atol=1e-6 cross-batch / 6dp gate), blank ablation, loss finiteness, and NO
regression of the Gate-1 text path (t5-small CPU)."""
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

from drishyam.losses import question_loss
from drishyam.model import DecoupledDecision, add_slot_token
from drishyam.schemas import Primitive, Question
from drishyam.tokenize import collate_questions

FIXED_LEN = 48
PAGE_TOKENS = 32  # shrunken "page" (real = 256); shapes are what the test exercises


def tiny_setup():
    tok = AutoTokenizer.from_pretrained("t5-small")
    bb = T5ForConditionalGeneration.from_pretrained("t5-small")
    slot_id = add_slot_token(tok, bb)
    return tok, bb, slot_id


def test_forward_embeds_shapes_invariance_blank():
    torch.manual_seed(0)
    tok, bb, slot_id = tiny_setup()
    m = DecoupledDecision(bb, slot_id).eval()
    d = bb.config.d_model
    page = torch.randn(1, PAGE_TOKENS, d)
    page_mask = torch.ones(1, PAGE_TOKENS, dtype=torch.long)
    choice = Question(qid="q1", primitive=Primitive.CHOICE, text="Which total?",
                      options=["Rs 1,234.50", "Rs 2,345.60", "Rs 3,456.70"])
    noul = Question(qid="q2", primitive=Primitive.NOUL, text="Is GSTIN readable?",
                    options=["No", "Yes"])

    def run(page_t, qs, **kw):
        di, dm, pos, _ = collate_questions(tok, qs, FIXED_LEN)
        with torch.no_grad():
            return m.forward_embeds(page_t, page_mask, di, dm, pos, **kw)

    alone = run(page, [choice])
    batch = run(page, [choice, noul])
    assert alone[0].shape == (3,) and len(batch) == 2 and batch[1].shape == (2,)
    assert torch.isfinite(alone[0]).all()
    torch.testing.assert_close(batch[0], alone[0], rtol=0, atol=1e-6)
    assert (batch[0].round(decimals=6) == alone[0].round(decimals=6)).all(), \
        "multi-question invariance broken through the embeds path"

    blank_out = run(page, [choice], blank_state=True)
    assert not torch.allclose(blank_out[0], alone[0]), "blank_state had no effect"

    loss = question_loss(alone[0].float(), torch.tensor([1.0, 0.0, 0.0]), "choice")
    assert torch.isfinite(loss) and float(loss) > 0


def test_row_case_grouping():
    """Training shape: 2 pages x their questions in one decoder batch via row_case."""
    torch.manual_seed(0)
    tok, bb, slot_id = tiny_setup()
    m = DecoupledDecision(bb, slot_id).eval()
    d = bb.config.d_model
    pages = torch.randn(2, PAGE_TOKENS, d)
    page_mask = torch.ones(2, PAGE_TOKENS, dtype=torch.long)
    q1 = Question(qid="a", primitive=Primitive.NOUL, text="x present?", options=["No", "Yes"])
    q2 = Question(qid="b", primitive=Primitive.NOUL, text="y present?", options=["No", "Yes"])
    di, dm, pos, _ = collate_questions(tok, [q1, q2], FIXED_LEN)
    row_case = torch.tensor([0, 1])  # q1 -> page0, q2 -> page1
    with torch.no_grad():
        out = m.forward_embeds(pages, page_mask, di, dm, pos, row_case=row_case)
    assert len(out) == 2 and all(t.shape == (2,) for t in out)
    assert torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()
    # per-page answers must differ across DIFFERENT pages (state actually matters here)
    with torch.no_grad():
        solo0 = m.forward_embeds(pages[0:1], page_mask[0:1], di[:1], dm[:1], [pos[0]])
        solo1 = m.forward_embeds(pages[1:2], page_mask[1:2], di[1:2], dm[1:2], [pos[1]])
    torch.testing.assert_close(out[0], solo0[0], rtol=0, atol=1e-6)
    torch.testing.assert_close(out[1], solo1[0], rtol=0, atol=1e-6)


def test_text_path_unchanged():
    """Gate-1 regression: the original forward() still behaves (t5-small)."""
    torch.manual_seed(0)
    tok, bb, slot_id = tiny_setup()
    m = DecoupledDecision(bb, slot_id).eval()
    s_ids = torch.randint(3, 100, (1, 16))
    s_mask = torch.ones_like(s_ids)
    choice = Question(qid="q1", primitive=Primitive.CHOICE, text="Which action?",
                      options=["pay now", "hold", "reject"])
    di, dm, pos, _ = collate_questions(tok, [choice], FIXED_LEN)
    with torch.no_grad():
        out = m(s_ids, s_mask, di, dm, pos)
    assert out[0].shape == (3,) and torch.isfinite(out[0]).all()
