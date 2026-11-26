"""JevBench public-231: task->case mapping (verified row contracts), tier loading,
accuracy summarization, and a t5-small CPU predict pass."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drishyam.eval_jevbench import (  # noqa: E402
    load_public,
    predict_rows,
    summarize,
    task_to_case,
)
from drishyam.model import DecoupledDecision, add_slot_token  # noqa: E402

STATE_LEN = 32
FIXED_LEN = 64


def _choice(tid, labels, expected, tier="easy"):
    return {
        "id": tid,
        "family": "intent",
        "split": "public",
        "state": "Msg body.",
        "question": {
            "type": "choice",
            "instructions": "Which intent?",
            "criteria": {k: f"{k} desc" for k in labels},
        },
        "labels": labels,
        "expected": expected,
        "tier": tier,
    }


def _noul(tid, expected, tier="easy"):
    return {
        "id": tid,
        "family": "fact",
        "split": "public",
        "state": "Facts here.",
        "question": {
            "type": "noul",
            "instructions": "Is it so?",
            "criteria": {"false": "not so", "true": "so"},
        },
        "labels": ["no", "yes"],
        "expected": expected,
        "tier": tier,
    }


def _score(tid, levels, expected, tier="hard"):
    return {
        "id": tid,
        "family": "ordinal",
        "split": "public",
        "state": "Incident.",
        "question": {"type": "score", "instructions": "Rate impact.", "criteria": levels},
        "labels": [str(i) for i in range(len(levels))],
        "expected": expected,
        "tier": tier,
    }


def _write_public(tmp_path, files):
    for fname, tasks in files.items():
        (tmp_path / fname).write_text("\n".join(json.dumps(t) for t in tasks) + "\n")


def test_task_to_case_mappings():
    c = task_to_case(_choice("t1", ["track_order", "cancel_order"], "cancel_order"), "easy")
    assert c["questions"][0].options == ["track_order", "cancel_order"]
    assert c["gold"] == [[0.0, 1.0]] and c["tier"] == "easy"
    n = task_to_case(_noul("t2", "yes"), "easy")
    assert n["questions"][0].options == ["No", "Yes"]
    assert n["gold"] == [[0.0, 1.0]]
    assert task_to_case(_noul("t3", "no"), "easy")["gold"] == [[1.0, 0.0]]
    s = task_to_case(_score("t4", ["L0", "L1", "L2", "L3"], 2), "hard")
    assert s["questions"][0].options == ["L0", "L1", "L2", "L3"]  # anchored levels
    assert s["gold"] == [[0.0, 0.0, 1.0, 0.0]]
    s_str = task_to_case(_score("t5", ["L0", "L1"], "1"), "hard")
    assert s_str["gold"] == [[0.0, 1.0]]  # string expected falls back to labels.index


def test_task_to_case_gates():
    with pytest.raises(ValueError, match="not in labels"):
        task_to_case(_choice("t1", ["a", "b"], "c"), "easy")
    bad_labels = _noul("t2", "yes")
    bad_labels["labels"] = ["true", "false"]
    with pytest.raises(ValueError, match="noul labels"):
        task_to_case(bad_labels, "easy")
    bad_exp = _noul("t3", "maybe")
    with pytest.raises(ValueError, match="noul expected"):
        task_to_case(bad_exp, "easy")
    with pytest.raises(ValueError, match="unknown question type"):
        task_to_case(
            {**_choice("t4", ["a", "b"], "a"), "question": {"type": "rank", "instructions": "?"}},
            "easy",
        )
    with pytest.raises(ValueError, match="score gold"):
        task_to_case(_score("t5", ["L0", "L1"], 5), "hard")


def test_load_public(tmp_path):
    _write_public(
        tmp_path,
        {
            "easy.jsonl": [
                _choice("easy-intent-00", ["a", "b"], "a"),
                _noul("easy-fact-00", "yes"),
            ],
            "original.jsonl": [_noul("original-policy-01-0", "no", tier="standard")],
            "hard.jsonl": [
                _score("hard-ordinal-01", ["L0", "L1", "L2"], 1),
                _choice("hard-x-00", ["a", "b", "c", "d", "e"], "e"),
            ],
        },
    )
    cases = load_public(tmp_path)
    assert len(cases) == 5
    assert {c["tier"] for c in cases} == {"easy", "standard", "hard"}
    assert sorted(c["case_id"] for c in cases if c["tier"] == "hard") == [
        "hard-ordinal-01",
        "hard-x-00",
    ]
    (tmp_path / "original.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="download the public JSONLs"):
        load_public(tmp_path)


def test_summarize():
    def row(cid, tier, prim, probs, gold):
        return {
            "case_id": cid,
            "tier": tier,
            "prim": prim,
            "probs": np.asarray(probs),
            "gold": np.asarray(gold),
            "options": ["a", "b"],
            "expected_raw": "a",
        }

    rows = [
        row("e1", "easy", "choice", [0.9, 0.1], [1.0, 0.0]),  # correct
        row("e2", "easy", "noul", [0.2, 0.8], [1.0, 0.0]),  # wrong
        row("s1", "standard", "noul", [0.7, 0.3], [1.0, 0.0]),  # correct
        row("h1", "hard", "score", [0.1, 0.9], [0.0, 1.0]),  # correct
        row("h2", "hard", "choice", [0.6, 0.4], [1.0, 0.0]),  # correct
    ]
    s = summarize(rows)
    assert s["n_tasks"] == 5
    assert s["acc"] == pytest.approx(4 / 5)
    assert s["acc_easy"] == pytest.approx(0.5)
    assert s["acc_standard"] == 1.0 and s["n_standard"] == 1
    assert s["acc_hard"] == 1.0 and s["n_hard"] == 2
    assert s["acc_choice"] == 1.0 and s["n_choice"] == 2
    assert s["acc_noul"] == pytest.approx(0.5) and s["n_noul"] == 2
    assert s["acc_score"] == 1.0 and s["n_score"] == 1
    assert s["brier"] > 0
    assert np.isnan(summarize([])["acc"])


def test_predict_rows_cpu():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("t5-small")
    bb = T5ForConditionalGeneration.from_pretrained("t5-small")
    slot_id = add_slot_token(tok, bb)
    model = DecoupledDecision(bb, slot_id).eval()
    cases = [
        task_to_case(_choice("t1", ["track", "hold", "pay"], "pay"), "easy"),
        task_to_case(_noul("t2", "no"), "standard"),
    ]
    rows = predict_rows(model, tok, cases, STATE_LEN, FIXED_LEN, device="cpu")
    assert len(rows) == 2
    assert rows[0]["probs"].shape == (3,) and rows[1]["probs"].shape == (2,)
    assert rows[0]["prim"] == "choice" and rows[1]["tier"] == "standard"
    for r in rows:
        assert np.isfinite(r["probs"]).all()
        assert abs(float(r["probs"].sum()) - 1.0) < 1e-5
        assert r["correct"] in (True, False)
