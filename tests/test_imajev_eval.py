"""CPU tests: ImajevBench record -> our typed row schema mapping + tau rule.

Their official scoring module is loaded verbatim (dependency-free) so every
prediction object we emit is validated against THEIR candidate keys, their
prob-vector validator and THEIR typed correctness — not against our own
reimplementation. No GPU, no checkpoint, no dataset dependency.
"""

import pytest

from drishyam.eval_imajev import (
    build_row_text,
    load_scoring,
    predictions_at,
    record_to_case,
    state_text,
    sweep,
    to_prediction,
    typed_correct,
)

scorer = load_scoring()


def _rec(id="x1", ftype="boolean", gold=True, field=None, track="visual", images=1):
    base_field = {"id": "decision", "question": "Is it so?", "type": ftype}
    if ftype == "choice":
        base_field["options"] = [{"value": v, "description": None} for v in ("2.30", "5.40")]
    elif ftype == "ordinal":
        base_field["levels"] = [
            {"value": 1, "description": "low"}, {"value": 2, "description": "mid"},
            {"value": 3, "description": "high"},
        ]
    if field is not None:
        base_field = field
    return {
        "id": id,
        "group_id": "g1",
        "track": track,
        "family": "rule_exception",
        "split": "dev",
        "images": [
            {"path": f"assets/{i}.jpg", "sha256": str(i)} for i in range(images)
        ],
        "request": {"fields": [base_field], "state": None},
        "gold": gold,
        "provenance": {"image_sources": [{"scene_kind": "menu"}]} if images else [],
    }


# --- boolean -> noul -------------------------------------------------------

def test_boolean_maps_to_noul_no_yes():
    case = record_to_case(_rec(gold=True))
    assert case["prim"] == "noul"
    assert case["question"].options == ["No", "Yes"]
    assert case["gold_idx"] == 1  # gold True -> Yes
    assert record_to_case(_rec(gold=False))["gold_idx"] == 0


def test_boolean_option_keys_are_their_candidate_keys():
    case = record_to_case(_rec(gold=True))
    keys = scorer.candidate_keys(_rec())[:2]  # their order: true first
    assert sorted(case["option_keys"]) == sorted(keys) == ["false", "true"]
    assert case["option_keys"] == ["false", "true"]  # No->false, Yes->true


def test_boolean_prediction_typed_correct_against_their_scorer():
    rec = _rec(gold=True)
    case = record_to_case(rec)
    probs = [0.1, 0.9]  # Yes
    pred = to_prediction(rec, case, probs, tau=0.0)
    assert pred["status"] == "answered" and pred["value"] is True  # type bool
    assert typed_correct([rec], {rec["id"]: pred}, scorer) == [True]
    wrong = to_prediction(rec, case, [0.9, 0.1], tau=0.0)
    assert wrong["value"] is False and typed_correct([rec], {rec["id"]: wrong}, scorer) == [False]


# --- choice -> choice ------------------------------------------------------

def test_choice_maps_their_values_in_order():
    case = record_to_case(_rec(ftype="choice", gold="5.40"))
    assert case["prim"] == "choice"
    assert case["question"].options == ["2.30", "5.40"]
    assert case["gold_idx"] == 1
    assert case["option_keys"] == scorer.candidate_keys(_rec(ftype="choice"))[:2]


def test_choice_prediction_value_is_str_and_scores():
    rec = _rec(ftype="choice", gold="2.30")
    case = record_to_case(rec)
    pred = to_prediction(rec, case, [0.7, 0.3], tau=0.0)
    assert pred["value"] == "2.30"  # str, their _typed_equal needs str
    assert typed_correct([rec], {rec["id"]: pred}, scorer) == [True]


# --- ordinal -> score ------------------------------------------------------

def test_ordinal_maps_level_descriptions_with_values():
    case = record_to_case(_rec(ftype="ordinal", gold=2))
    assert case["prim"] == "score"
    assert case["question"].options == ["low", "mid", "high"]
    assert case["option_keys"] == ["1", "2", "3"]
    assert case["gold_idx"] == 1


def test_ordinal_prediction_value_is_int():
    rec = _rec(ftype="ordinal", gold=3)
    case = record_to_case(rec)
    pred = to_prediction(rec, case, [0.1, 0.2, 0.7], tau=0.0)
    assert pred["value"] == 3 and type(pred["value"]) is int
    assert typed_correct([rec], {rec["id"]: pred}, scorer) == [True]


# --- Unknown (gold null) + abstention rule ---------------------------------

def test_null_gold_is_abstention_target():
    rec = _rec(gold=None)
    case = record_to_case(rec)
    assert case["gold_idx"] is None
    pred = to_prediction(rec, case, [0.1, 0.9], tau=0.95)  # max 0.9 < 0.95 -> abstain
    assert pred["status"] == "abstained" and pred["value"] is None
    assert typed_correct([rec], {rec["id"]: pred}, scorer) == [True]  # their rule


def test_abstention_threshold_boundary():
    rec = _rec(gold=None)
    case = record_to_case(rec)
    probs = [0.5, 0.5]
    assert to_prediction(rec, case, probs, tau=0.5)["status"] == "answered"  # not < tau
    assert to_prediction(rec, case, probs, tau=0.500001)["status"] == "abstained"


def test_answered_on_unknown_gold_is_wrong():
    rec = _rec(gold=None)
    case = record_to_case(rec)
    pred = to_prediction(rec, case, [0.2, 0.8], tau=0.0)
    assert typed_correct([rec], {rec["id"]: pred}, scorer) == [False]


# --- prob vectors pass THEIR validator --------------------------------------

@pytest.mark.parametrize(
    "ftype,gold,probs,tau",
    [
        ("boolean", True, [0.3, 0.7], 0.0),
        ("boolean", None, [0.3, 0.7], 0.99),
        ("choice", "5.40", [0.6, 0.4], 0.0),
        ("ordinal", 2, [0.2, 0.5, 0.3], 0.0),
    ],
)
def test_prob_vectors_valid_per_their_validator(ftype, gold, probs, tau):
    rec = _rec(ftype=ftype, gold=gold)
    case = record_to_case(rec)
    pred = to_prediction(rec, case, probs, tau)
    state, vec = scorer._valid_vector(rec, pred)
    assert state == "valid", (state, pred)
    assert set(vec) == set(scorer.candidate_keys(rec))
    assert abs(sum(vec.values()) - 1.0) < 1e-9


def test_missing_prediction_scores_incorrect_not_crash():
    rec = _rec(gold=True)
    assert typed_correct([rec], {}, scorer) == [False]
    report = scorer.score([rec], {})
    assert report["capability"]["all_records"]["accuracy"] == 0.0


# --- tau sweep picks the abstention-aware optimum ---------------------------

def test_tau_sweep_finds_abstention_only_split():
    # gold null where the model is least confident -> fitting tau must abstain
    # exactly there and nowhere else (their typed metric rewards it).
    recs = [
        _rec(id="a", gold=True),   # confident Yes
        _rec(id="b", gold=False),  # confident No
        _rec(id="c", gold=None),   # uncertain -> should be abstained
    ]
    cases = [record_to_case(r) for r in recs]
    probs = [[0.1, 0.9], [0.9, 0.1], [0.55, 0.45]]
    preds0 = predictions_at(recs, cases, probs, 0.0)
    assert typed_correct(recs, preds0, scorer) == [True, True, False]
    curve = sweep(recs, cases, probs, scorer)
    best = max(curve, key=lambda e: (e["acc"], -e["tau"]))
    tau, acc = best["tau"], best["acc"]
    assert acc == 1.0 and tau > 0.45
    preds = predictions_at(recs, cases, probs, tau)
    assert [p["status"] for p in preds.values()] == ["answered", "answered", "abstained"]
    assert typed_correct(recs, preds, scorer) == [True, True, True]


# --- state text budget (question+options always fit) ------------------------

class _TinyTok:
    """Deterministic fake tokenizer: ~1 token per char, single <slot> token."""

    slot_id = 999

    def convert_tokens_to_ids(self, tok):
        return self.slot_id

    def __call__(self, text, add_special_tokens=True):
        ids = [self.slot_id if text.startswith("<slot>", i) else (ord(c) % 1000)
               for i, c in enumerate(text)]
        return {"input_ids": ids}

    @property
    def pad_token_id(self):
        return 0


def test_state_truncates_never_the_question_slots():
    tok = _TinyTok()
    case = record_to_case(_rec(ftype="choice", gold="5.40"))
    case["state"] = "x" * 5000
    text, flags = build_row_text(case, tok, fixed_len=100)
    assert flags["question"] is False and flags["state"] is True
    assert "State: " in text and text.count("<slot>") == 2  # both options keep slots
    assert len(tok(text)["input_ids"]) <= 100


def test_no_state_keeps_plain_suffix():
    tok = _TinyTok()
    case = record_to_case(_rec())
    text, flags = build_row_text(case, tok, fixed_len=100)
    assert flags == {"question": False, "state": False}
    assert text.startswith("Question: Is it so?")


def test_state_text_sorted_keys_json():
    rec = _rec()
    rec["request"]["state"] = {"b": 1, "a": {"z": 2, "y": [3]}}
    assert state_text(rec) == '{"a": {"y": [3], "z": 2}, "b": 1}'
