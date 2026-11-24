import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "artifacts", "paper_rig"))

def _proc():
    from pack_batch import get_processor
    return get_processor()

def test_placeholders_are_256():
    from pack_batch import build_example
    ex = build_example("thermal", 12345, _proc())
    mm_tok = sum(1 for t in ex["input_ids"] if t == ex["image_token_id"])
    assert mm_tok == 256, f"got {mm_tok} image placeholders"

def test_labels_mask_prompt_and_keep_answer():
    from pack_batch import build_example
    ex = build_example("thermal", 12345, _proc())
    lab = ex["labels"]
    assert lab.count(-100) > len(lab) * 0.5
    assert all(t != -100 for t in lab[-ex["answer_len"]:])
    assert lab[-1] != -100

def test_deterministic_same_seed():
    from pack_batch import build_example
    a = build_example("thermal", 12345, _proc())
    b = build_example("thermal", 12345, _proc())
    assert a["input_ids"] == b["input_ids"]
    assert (a["pixel_values"] == b["pixel_values"]).all()

def test_answer_fits_budget():
    from pack_batch import build_example
    for fam, seed in (("thermal", 1), ("savings", 2), ("card", 3)):
        ex = build_example(fam, seed, _proc())
        assert ex["answer_len"] <= 28, f"{fam}: answer_len {ex['answer_len']} > 28"
