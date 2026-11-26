"""SROIE verification-row construction: gold quoting, perturbation safety, balance.

Pure-CPU laws of the protocol (docs/BENCHMARK_LEDGER.md SROIE row):
  L1 TRUE claims quote the receipt's gold value verbatim.
  L2 FALSE claims are drawn from OTHER receipts' same-field gold and are never
     equal to this receipt's gold (exact OR whitespace-normalized compare).
  L3 Exactly one TRUE + one FALSE per (receipt, field) -> 50/50 balanced.
  L4 Determinism: seed 2026 fully determines the row set (rows digest equal).
  L5 FALSE claims come from the same FIELD type (totals drawn from totals).
  L6 Question template carries field name + claimed value + No/Yes options.
"""
import json

import numpy as np

from drishyam.eval_sroie import (
    FIELDS,
    SEED,
    build_verification_rows,
    false_claim_pool,
    question_template,
    row_case,
    sha256_rows,
)

RECEIPTS = [
    {"key": f"R{i:03d}", "gold": {
        "total": tv, "date": dv, "company": cv,
        "address": f"{i} JALAN TEST",
    }}
    for i, (tv, dv, cv) in enumerate([
        ("10.00", "01/01/2019", "ALPHA SDN BHD"),
        ("22.50", "02/01/2019", "BETA TRADING"),
        ("7.60", "03/01/2019", "ALPHA SDN BHD"),   # duplicate company on purpose
        ("193.00", "15/01/2019", "GAMMA MART"),
        ("436.20", "09/02/2018", "DELTA ENTERPRISE"),
        ("12.00", " 27/03/2018 ", "EPSILON SENDIRIAN BERHAD"),  # padded gold
    ])
]


def test_l1_true_claims_quote_gold_verbatim():
    rows = build_verification_rows(RECEIPTS, SEED)
    for r in rows:
        if r["claim_type"] == "true":
            assert r["claimed"] == r["gold"], r
    # padded gold: outer whitespace stripped, inner verbatim
    padded = [r for r in rows if r["receipt_key"] == "R005" and r["claim_type"] == "true"]
    assert {r["claimed"] for r in padded} == {"12.00", "27/03/2018", "EPSILON SENDIRIAN BERHAD"}


def test_l2_false_claims_never_gold():
    rows = build_verification_rows(RECEIPTS, SEED)
    for r in rows:
        if r["claim_type"] == "false":
            assert r["claimed"] != r["gold"], r
            assert r["claimed"].strip() != r["gold"].strip(), r


def test_l2b_false_pool_excludes_gold_and_self():
    pool = false_claim_pool(RECEIPTS, "company", "ALPHA SDN BHD")
    assert "ALPHA SDN BHD" not in pool
    assert pool == ["BETA TRADING", "DELTA ENTERPRISE", "EPSILON SENDIRIAN BERHAD",
                    "GAMMA MART"]


def test_l3_balance_5050_per_field_and_overall():
    rows = build_verification_rows(RECEIPTS, SEED)
    assert len(rows) == len(RECEIPTS) * len(FIELDS) * 2
    yes = sum(r["answer"] == "Yes" for r in rows)
    no = sum(r["answer"] == "No" for r in rows)
    assert yes == no == len(RECEIPTS) * len(FIELDS)
    for field in FIELDS:
        sel = [r for r in rows if r["field"] == field]
        assert sum(r["answer"] == "Yes" for r in sel) == sum(r["answer"] == "No" for r in sel)


def test_l4_determinism_and_row_ids():
    a = build_verification_rows(RECEIPTS, SEED)
    b = build_verification_rows(RECEIPTS, SEED)
    assert a == b
    assert sha256_rows(a) == sha256_rows(b)
    assert len({r["row_id"] for r in a}) == len(a)  # unique row ids


def test_l5_false_value_from_same_field_type():
    rows = build_verification_rows(RECEIPTS, SEED)
    by_field = {f: {r["gold"][f].strip() for r in RECEIPTS} for f in FIELDS}
    for r in rows:
        if r["claim_type"] == "false":
            assert r["claimed"] in by_field[r["field"]], r


def test_l6_question_template_and_schema_case():
    q = row_case({"row_id": "T:total:true", "field": "total", "claimed": "193.00"})
    assert '"193.00"' in q.text and "total" in q.text
    assert "No" in q.text and "Yes" in q.text
    assert q.options == ["No", "Yes"] and q.primitive.value == "noul"
    assert question_template("date", "01/01/2019") == (
        'Does the date on this receipt read "01/01/2019"? Answer No or Yes.')


def test_l7_different_receipt_same_company_gold_cross_draw():
    """Two receipts sharing a company gold must draw FALSE values != their own gold;
    the shared value is a legal FALSE for the other one only via the pool rule."""
    rows = build_verification_rows(RECEIPTS, SEED)
    for r in rows:
        if r["field"] == "company" and r["claim_type"] == "false":
            assert r["claimed"] != r["gold"]


def test_metric_block_shapes():
    """metric_block sanity on a toy balanced set (auc/ece finite, acc exact)."""
    from drishyam.eval_sroie import metric_block

    rows = [
        {"claim_type": "true", "answer": "Yes"},
        {"claim_type": "false", "answer": "No"},
    ]
    probs = [np.array([0.2, 0.8]), np.array([0.9, 0.1])]
    blk = metric_block(rows, probs)
    assert blk["n"] == 2 and blk["n_true"] == 1 and blk["n_false"] == 1
    assert blk["acc"] == 1.0 and blk["acc_true"] == 1.0 and blk["acc_false"] == 1.0
    assert np.isfinite(blk["auc_p_yes"]) and np.isfinite(blk["ece15"])
    # auc of perfect separation = 1.0
    assert blk["auc_p_yes"] == 1.0
    _ = json.dumps(blk)  # json-serializable for the table
