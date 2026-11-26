import os
import sys

import numpy as np

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
    from content_generator import gen_card, gen_statement
    rng = np.random.RandomState(3)
    page, gt = gen_statement(rng, seed=1)
    qa = build_qa("savings", gt, seed=4)  # 4%4==0 -> account_number
    assert qa[0]["field"] == "account_number"
    assert qa[0]["a"] == gt["account_number"]
    qb = build_qa("savings", gt, seed=3)  # 3%4==3 -> bank (now variable, not hardcoded)
    assert qb[0]["a"] == gt["bank"]
    rng = np.random.RandomState(4)
    page, gt = gen_card(rng, seed=2)
    qa = build_qa("card", gt, seed=0)  # 0%7==0 -> total_due
    assert qa[0]["field"] == "total_due"
    assert qa[0]["a"] == f"Rs {gt['total_amount_due']:.2f}"

def test_new_families_answer_from_gt():
    from fin_docs import gen_electricity, gen_salary, gen_table, gen_telecom
    for fam, fn in (("electricity", gen_electricity), ("telecom", gen_telecom),
                    ("salary", gen_salary), ("table", gen_table)):
        rng = np.random.RandomState(9)
        page, gt = fn(rng, seed=5)
        qa = build_qa(fam, gt, seed=0)
        assert len(qa) == 1 and qa[0]["a"] is not None and str(qa[0]["a"]) != ""
        assert qa[0]["field"] != ""

def test_loio_holdout_bucket():
    from institutions import holdout
    seeds = [1000001 + i * 7919 for i in range(200)]
    held = [s for s in seeds if holdout(s)]
    assert 0 < len(held) < len(seeds) // 3    # a real minority bucket, not everything


# ---------------- K5-v4: train-family omission support ----------------
# gt keys the train-family extractors ask about (identity keys excluded)
ASKED_GT_KEYS = {
    "electricity": {"printed_total", "units_consumed", "due_date", "consumer_no", "provider"},
    "telecom": {"printed_total", "due_date", "invoice_no", "mobile", "provider"},
    "savings": {"account_number", "from_date", "n_transactions", "bank"},
    "card": {"total_amount_due", "min_amount_due", "due_date", "card_masked",
             "customer_name", "invoice_no", "issuer"},
}
IDENTITY_KEYS = {"family", "provider", "provider_index", "bank", "bank_index",
                 "issuer", "issuer_index", "consumer_no"}
# the one amount per family that must survive every omission
KEPT_AMOUNTS = {"electricity": "energy_charges", "telecom": "plan_amount",
                "card": "credit_limit", "savings": "n_transactions"}
GENS = {"electricity": "gen_electricity", "telecom": "gen_telecom",
        "savings": "gen_statement", "card": "gen_card"}


def _first_asked_twin(family):
    """First registry entry whose omitted field is asked by the family extractor."""
    import fin_docs
    seeds = sorted(int(s) for s in fin_docs._OMIT.get(family, {}))
    for s in seeds:
        omit = fin_docs.omitted_fields(family, s)
        if omit and (sorted(omit)[0] in ASKED_GT_KEYS[family]):
            return s, sorted(omit)[0]
    raise AssertionError(f"no extractor-asked twin registered for {family}")


def _gen(family):
    import fin_docs
    if family in ("electricity", "telecom"):
        return getattr(fin_docs, f"gen_{family}")
    from content_generator import gen_card, gen_statement
    return {"savings": gen_statement, "card": gen_card}[family]


def test_k5_omittable_sets_contract():
    import fin_docs
    assert set(fin_docs.OMITTABLE_FIELDS) == {"electricity", "telecom", "savings", "card"}
    for fam, omittable in fin_docs.OMITTABLE_FIELDS.items():
        assert not (set(omittable) & IDENTITY_KEYS), fam
        assert KEPT_AMOUNTS[fam] not in omittable, fam          # one amount always survives
        # every extractor-asked non-identity gt key must be omittable (twin No rows)
        for asked in ASKED_GT_KEYS[fam] - IDENTITY_KEYS - {KEPT_AMOUNTS[fam]}:
            if asked in {"provider", "consumer_no", "bank", "issuer", "n_transactions"}:
                continue
            assert asked in omittable, (fam, asked)
    # registry only omits declared-omittable fields (salary predates the v4 sets)
    for fam, seeds in fin_docs._OMIT.items():
        if fam not in fin_docs.OMITTABLE_FIELDS:
            continue
        for fields in seeds.values():
            assert set(fields) <= set(fin_docs.OMITTABLE_FIELDS[fam]), fam


def test_k5_train_twin_drops_key_from_gt_and_pairs():
    """Registered omission on a train family: key gone from gt AND from extractor pairs."""
    import fin_docs
    from row_labels import TABLES
    for fam in GENS:
        seed, field = _first_asked_twin(fam)
        assert fin_docs.omitted_fields(fam, seed) == frozenset({field})
        _, gt = _gen(fam)(np.random.RandomState(seed), seed)
        assert field not in gt, (fam, seed, field, sorted(gt))
        pairs = TABLES[fam](gt)
        assert field not in [f for f, _, _ in pairs], (fam, seed, field)
        assert len(pairs) >= 2, fam                       # page still answerable
        # identity fields + the kept amount survive on the twin page
        must_keep = (IDENTITY_KEYS & ASKED_GT_KEYS[fam]) | {"family", KEPT_AMOUNTS[fam]}
        for k in must_keep:
            assert k in gt, (fam, seed, k, sorted(gt))


def test_k5_extractors_none_tolerant_on_pruned_gt():
    """Pruned gt dicts (K5 twins) must not KeyError; missing fields just drop out."""
    from row_labels import _card_pairs, _electricity_pairs, _savings_pairs, _telecom_pairs
    pruned = [
        (_electricity_pairs, {"family": "electricity_bill", "provider": "TNEB",
                              "provider_index": 0, "consumer_no": "1234567890",
                              "due_date": "15/01/2024", "energy_charges": 100.0},
         ["due_date", "consumer_no", "provider"]),          # printed_total/units pruned
        (_telecom_pairs, {"family": "telecom_invoice", "provider": "JIO", "provider_index": 1,
                          "due_date": "15/01/2024", "subtotal": 299.0},
         ["due_date", "provider"]),                         # printed_total/invoice/mobile pruned
        (_savings_pairs, {"family": "bank_savings_statement", "bank": "HDFC BANK",
                          "bank_index": 2, "n_transactions": 11},
         ["n_transactions", "bank"]),                       # account/from_date pruned
        (_card_pairs, {"family": "bank_card_estatement", "issuer": "ICICI BANK",
                       "issuer_index": 3},
         ["issuer"]),                                       # everything else pruned
    ]
    for fn, gt, want_fields in pruned:
        pairs = fn(gt)                                      # must not raise
        assert [f for f, _, _ in pairs] == want_fields, (fn.__name__, pairs)


def test_k5_complete_gt_pair_order_unchanged():
    """Non-twin pages: extractor returns the full canonical pair list, in order."""
    import fin_docs
    from row_labels import TABLES
    canonical = {
        "electricity": ["total_amount", "units_consumed", "due_date", "consumer_no", "provider"],
        "telecom": ["total_amount", "due_date", "invoice_no", "mobile", "provider"],
        "savings": ["account_number", "from_date", "n_transactions", "bank"],
        "card": ["total_due", "min_due", "due_date", "card_masked", "customer_name",
                 "invoice_no", "issuer"],
    }
    for fam in GENS:
        seed = {"electricity": 1_005_011, "telecom": 1_007_013,
                "savings": 1_009_017, "card": 1_011_019}[fam]
        assert not fin_docs.omitted_fields(fam, seed)
        _, gt = _gen(fam)(np.random.RandomState(seed), seed)
        assert [f for f, _, _ in TABLES[fam](gt)] == canonical[fam], fam
