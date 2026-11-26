"""bev_data: bev-decision row-contract parsing, validation gates, val carve."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drishyam.bev_data import (  # noqa: E402
    carve_val,
    cases_from_parquet,
    parse_row,
    question_from_dict,
)


def _q_choice(label="Rs 20.00"):
    return {
        "type": "choice",
        "instructions": "Which total is printed?",
        "criteria": {"Rs 10.00": "low", "Rs 20.00": "high"},
        "label": label,
    }


def _q_noul(label=True):
    return {"type": "noul", "instructions": "Is it paid?", "label": label}


def _q_score(label=1):
    return {
        "type": "score",
        "instructions": "Rate legibility",
        "criteria": ["L1 bad", "L2 ok", "L3 good"],
        "label": label,
    }


def _row(state, qs, domain="Sentiment, emotion, and moderation"):
    types = list({q["type"] for q in qs.values()})
    return {
        "state": state,
        "questions_json": qs,
        "domain": domain,
        "question_types": types,
        "question_count": len(qs),
    }


def _write_parquet(rows, tmp_path, name="bev_tiny.parquet"):
    tbl = pa.table(
        {
            "state": pa.array([r["state"] for r in rows], type=pa.large_string()),
            "questions_json": pa.array(
                [json.dumps(r["questions_json"]) for r in rows], type=pa.large_string()
            ),
            "domain": pa.array([r["domain"] for r in rows], type=pa.string()),
            "question_types": pa.array(
                [r["question_types"] for r in rows], type=pa.list_(pa.string())
            ),
            "question_count": pa.array([r["question_count"] for r in rows], type=pa.int16()),
        }
    )
    path = tmp_path / name
    import pyarrow.parquet as pq

    pq.write_table(tbl, path)
    return path


def test_question_mappings():
    q, gold = question_from_dict("c1", _q_choice())
    assert q.options == ["Rs 10.00", "Rs 20.00"]  # criteria keys in stored order
    assert q.primitive.value == "choice" and gold == [0.0, 1.0]
    q, gold = question_from_dict("n1", _q_noul(True))
    assert q.options == ["No", "Yes"] and gold == [0.0, 1.0]
    q, gold = question_from_dict("n2", _q_noul(False))
    assert gold == [1.0, 0.0]
    q, gold = question_from_dict("s1", _q_score(2))
    assert q.options == ["L1 bad", "L2 ok", "L3 good"] and gold == [0.0, 0.0, 1.0]


def test_question_validation_raises():
    bad = [
        ("unknown-type", {"type": "rank", "instructions": "x"}),
        ("choice-label-missing", {**_q_choice(), "label": "Rs 30.00"}),
        ("choice-one-option", {**_q_choice(), "criteria": {"only": "one"}}),
        ("noul-label-str", {**_q_noul(), "label": "true"}),
        ("score-label-oob", _q_score(3)),
        ("score-label-bool", {**_q_score(), "label": True}),
        ("score-criteria-dict", {**_q_score(), "criteria": {"a": "b"}}),
        ("score-empty-level", {**_q_score(), "criteria": ["a", ""]}),
        ("no-instructions", {"type": "noul", "label": True}),
    ]
    for name, q in bad:
        with pytest.raises(ValueError, match=name.split("-")[0]):
            question_from_dict(name, q)


def test_cases_from_parquet_end_to_end(tmp_path):
    rows = [
        _row("State A text.", {"c": _q_choice()}),
        _row("State B text.", {"n": _q_noul(False)}),
        _row("State C text.", {"s": _q_score(0)}, domain="Retail"),
        _row("State D text.", {"c": _q_choice("Rs 10.00"), "n": _q_noul(True)}),
        _row("   ", {"c": _q_choice()}),  # empty state -> skipped, reported
    ]
    path = _write_parquet(rows, tmp_path)
    cases, skipped = cases_from_parquet(path)
    assert skipped == 1 and len(cases) == 4
    assert [c["case_id"] for c in cases] == [f"bev:default:train:{i}" for i in range(4)]
    assert cases[2]["domain"] == "Retail"
    for c in cases:
        assert len(c["questions"]) == len(c["gold"]) >= 1
        for q, g in zip(c["questions"], c["gold"], strict=False):
            assert abs(sum(g) - 1.0) < 1e-9 and len(g) == len(q.options) >= 2
    multi = cases[3]
    assert [q.primitive.value for q in multi["questions"]] == ["choice", "noul"]
    assert multi["gold"][0] == [1.0, 0.0] and multi["gold"][1] == [0.0, 1.0]


def test_limit_stops_early(tmp_path):
    rows = [_row(f"State {i}.", {"n": _q_noul(i % 2 == 0)}) for i in range(5)]
    cases, skipped = cases_from_parquet(_write_parquet(rows, tmp_path), limit=2)
    assert len(cases) == 2 and skipped == 0


def test_parse_row_contract_gates():
    good = _row("s", {"c": _q_choice()})
    good["questions_json"] = json.dumps(good["questions_json"])
    bad_count = {**good, "question_count": 2}
    with pytest.raises(ValueError, match="question_count"):
        parse_row(7, bad_count, "default", "train")
    bad_types = {**good, "question_types": ["noul"]}
    with pytest.raises(ValueError, match="question_types"):
        parse_row(7, bad_types, "default", "train")
    bad_q = _row("s", {"c": {**_q_choice(), "label": "nope"}})
    bad_q["questions_json"] = json.dumps(bad_q["questions_json"])
    with pytest.raises(ValueError, match="row 7"):
        parse_row(7, bad_q, "default", "train")


def test_parse_determinism(tmp_path):
    rows = [
        _row("State A.", {"c": _q_choice(), "n": _q_noul()}),
        _row("State B.", {"s": _q_score()}),
    ]
    path = _write_parquet(rows, tmp_path)
    one = cases_from_parquet(path)[0]
    two = cases_from_parquet(path)[0]
    assert json.dumps(one, default=str) == json.dumps(two, default=str)


def test_carve_val_deterministic_disjoint():
    cases = [{"case_id": f"c{i}", "state": f"State {i}.", "questions": [], "gold": []}
             for i in range(10)]
    train, val = carve_val(cases, 3, seed=42)
    assert len(train) == 7 and len(val) == 3
    ids_t = {c["case_id"] for c in train}
    ids_v = {c["case_id"] for c in val}
    assert not ids_t & ids_v and len(ids_t | ids_v) == 10
    train2, val2 = carve_val(cases, 3, seed=42)
    assert [c["case_id"] for c in train] == [c["case_id"] for c in train2]
    assert [c["case_id"] for c in val] == [c["case_id"] for c in val2]
    train3, _ = carve_val(cases, 3, seed=7)
    assert [c["case_id"] for c in train] != [c["case_id"] for c in train3]
    with pytest.raises(ValueError):
        carve_val(cases, 0)
    with pytest.raises(ValueError):
        carve_val(cases, 10)
