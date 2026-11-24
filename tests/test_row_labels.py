import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "artifacts", "paper_rig"))
from row_labels import build_qa

def _thermal_gt(total_destroyed=False):
    import thermal_generator as tg
    rng = np.random.RandomState(5)
    page, gt = tg.gen_thermal(rng, 12345)
    if total_destroyed:
        gt.setdefault("legibility_applied", {})["total_amt"] = "smudge_heavy"
    return gt

def test_thermal_total_value_when_clean():
    qa = build_qa("thermal", _thermal_gt(False), seed=6)  # 6%6==0 -> total_amt
    assert any(q["a"].startswith("Rs ") for q in qa)

def test_thermal_total_unreadable_when_smudged_heavy():
    gt = _thermal_gt(True)
    qa = build_qa("thermal", gt, seed=6)  # -> total_amt
    tot = [q for q in qa if q["field"] == "total_amt"]
    assert tot and tot[0]["a"] == "unreadable"

def test_one_question_per_call_and_rotation_changes_field():
    gt = _thermal_gt(False)
    a = build_qa("thermal", gt, seed=1)
    b = build_qa("thermal", gt, seed=2)
    assert len(a) == 1 and len(b) == 1
    assert a[0]["field"] != b[0]["field"]

def test_savings_and_card_answer_from_scalar_gt():
    from content_generator import gen_statement, gen_card
    rng = np.random.RandomState(3)
    page, gt = gen_statement(rng)
    qa = build_qa("savings", gt, seed=4)  # 4%4==0 -> account_number
    assert qa[0]["a"] == gt["account_number"]
    rng = np.random.RandomState(4)
    page, gt = gen_card(rng)
    qa = build_qa("card", gt, seed=6)  # 6%6==0 -> total_due
    assert qa[0]["a"] == f"Rs {gt['total_amount_due']:.2f}"
