"""Gate-2 label factory tests: coverage, one-hot contract, LOFO, determinism, gt parity."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drishyam.vision_data import (  # noqa: E402
    EMB_DIRS,
    UNSEEN_FAMILY,
    LabelFactory,
    load_pages,
    page_gt,
)


def test_pages_banked_and_parity():
    pages = load_pages()
    assert len(pages) > 80_000, f"only {len(pages)} pages found in {EMB_DIRS}"
    for (fam, seed), (rel, idx) in list(pages.items())[:5]:
        emb = np.load(ROOT / rel, mmap_mode="r")
        lab = (ROOT / rel).parent / (ROOT / rel).name.replace("emb_", "lab_", 1).replace(
            ".npy", ".jsonl")
        rows = [json.loads(l) for l in open(lab) if l.strip()]
        assert emb.shape[0] == len(rows)


def test_page_gt_matches_jit_pipeline():
    sys.path.insert(0, str(ROOT / "artifacts" / "paper_rig"))
    from jit_pipeline import gen_page

    for fam, seed in [("thermal", 1_000_003), ("electricity", 1_005_011),
                      ("salary", 1_002_007)]:
        _, gt_ref = gen_page(fam, seed)
        gt = page_gt(fam, seed)
        assert gt.get("legibility_applied") == gt_ref.get("legibility_applied"), fam
        for k, v in gt_ref.items():
            if k in ("legibility_applied", "degradation_plan", "row_id"):
                continue
            assert json.dumps(gt.get(k), sort_keys=True, default=str) == \
                json.dumps(v, sort_keys=True, default=str), (fam, seed, k)


def test_rows_contract_and_lofo():
    pages = load_pages()
    small = {k: v for i, (k, v) in enumerate(pages.items())
             if i % 997 == 0}  # ~90 pages, fast
    pools = {}
    insts = []
    factory = LabelFactory(small, pools, insts, seed=42)
    emitted = []
    for (fam, seed), loc in small.items():
        if fam == UNSEEN_FAMILY:
            continue
        gt = page_gt(fam, seed)
        qs, golds = factory.questions_for(fam, seed, gt, "train")
        if qs:
            emitted.append({"family": fam, "seed": seed, "questions": qs, "gold": golds})
    assert emitted, "no questions emitted on the probe sample"
    for r in emitted:
        for q, g in zip(r["questions"], r["gold"], strict=False):
            assert abs(sum(g) - 1.0) < 1e-6
            assert len(g) == len(q["options"]) >= 2
            assert q["primitive"] in ("noul", "choice", "score")
        assert all(q["qid"].split(":")[0] == r["family"] for q in r["questions"])


def test_factory_determinism():
    pages = load_pages()
    probe = [(k, v) for i, (k, v) in enumerate(pages.items()) if i % 499 == 0][:12]
    outs = []
    for _ in range(2):
        factory = LabelFactory(dict(probe), {}, [], seed=7)
        one = []
        for (fam, seed), loc in probe:
            gt = page_gt(fam, seed)
            qs, golds = factory.questions_for(fam, seed, gt, "train")
            one.append(json.dumps([qs, golds], sort_keys=True, default=str))
        outs.append(one)
    assert outs[0] == outs[1], "label factory is not deterministic for fixed seed"


def _yes_idx(options):
    for i, o in enumerate(options):
        if o.lower().strip() in ("yes", "true", "readable", "legible", "correct"):
            return i
    raise AssertionError(f"no yes-option in {options}")


def test_k5_twin_emits_presence_no():
    """Registered omission: page_gt drops the field AND the factory asks it with gold No."""
    import fin_docs
    assert fin_docs.omitted_fields("salary", 500123) == frozenset({"period"})
    gt = page_gt("salary", 500123)
    assert "period" not in gt, f"registry omission not applied to gt: {sorted(gt)}"
    factory = LabelFactory({}, {}, [], seed=1)
    qs, golds = factory.questions_for("salary", 500123, gt, "unseen")
    hits = [(q, g) for q, g in zip(qs, golds, strict=False)
            if q["qid"].endswith("noul:pres:period")]
    assert hits, f"no presence question for omitted field: {[q['qid'] for q in qs]}"
    q, g = hits[0]
    assert g[_yes_idx(q["options"])] == 0.0, "omitted field must have gold No"


def test_k5_non_twin_parity_and_no_ghost_questions():
    """Non-registered seeds: gt unchanged, no omitted-field questions emitted."""
    import fin_docs
    assert fin_docs.omitted_fields("salary", 1_002_007) == frozenset()
    gt = page_gt("salary", 1_002_007)
    assert "period" in gt and "medical" in gt
    factory = LabelFactory({}, {}, [], seed=1)
    qs, _ = factory.questions_for("salary", 1_002_007, gt, "unseen")
    pres = [q["qid"] for q in qs if ":noul:pres:" in q["qid"]]
    assert len(pres) == 3, f"expected exactly 3 presence questions, got {pres}"


def test_k5_thermal_unreadable_presence_no():
    """Thermal variance: unreadable fields get a presence question with gold No."""
    found = False
    for seed in range(1_000_000, 1_001_000):
        gt = page_gt("thermal", seed)
        leg = gt.get("legibility_applied", {})
        from row_labels import _leg
        unread = [f for f in leg if _leg(gt, f) == "unreadable"]
        if not unread:
            continue
        factory = LabelFactory({}, {}, [], seed=3)
        qs, golds = factory.questions_for("thermal", seed, gt, "train")
        for q, g in zip(qs, golds, strict=False):
            if q["qid"].endswith(f"noul:pres:{unread[0]}"):
                assert g[_yes_idx(q["options"])] == 0.0
                found = True
        if found:
            break
    assert found, "no thermal page with unreadable field produced a presence-No in probe range"


# ---------------- K5-v4: train families + presence balancing ----------------
def test_k5_train_family_gt_parity_fixed_seeds():
    """Non-registered train seeds render gt byte-identical to the pre-change capture."""
    fixture = ROOT / "tests" / "fixtures" / "gate2_k5_parity_gts.json"
    with open(fixture) as fh:
        fix = json.load(fh)
    assert set(fix) == {"electricity", "telecom", "savings", "card"}
    for fam, rec in fix.items():
        gt = page_gt(fam, rec["seed"])
        assert json.dumps(gt, sort_keys=True, default=str) == \
            json.dumps(rec["gt"], sort_keys=True, default=str), fam


def test_k5_train_twin_emits_presence_no_and_template_matches_yes():
    """Train-family twin: presence-No emitted; twin No template == base Yes template."""
    from test_row_labels import _first_asked_twin

    from drishyam.vision_data import FIELD_LABEL
    for fam in ("electricity", "telecom", "savings", "card"):
        seed, field = _first_asked_twin(fam)
        gt = page_gt(fam, seed)
        assert field not in gt
        factory = LabelFactory({}, {}, [], seed=1)
        qs, golds = factory.questions_for(fam, seed, gt, "train")
        hits = [(q, g) for q, g in zip(qs, golds, strict=False)
                if q["qid"] == f"{fam}:{seed}:noul:pres:{field}"]
        assert hits, f"no presence question for {fam} twin {field}: {[q['qid'] for q in qs]}"
        q, g = hits[0]
        assert g[_yes_idx(q["options"])] == 0.0
        # same template the base Yes row for the matching extractor field uses
        twin_label = FIELD_LABEL.get(field, field)
        assert q["text"] == f"Is the {twin_label} present and readable on this page?"
        for alias, label in (("printed_total", "total_amount"), ("min_amount_due", "min_due")):
            if field == alias:
                assert twin_label == FIELD_LABEL[label], (fam, field)


def test_k5_train_non_twin_no_ghost_questions():
    """Non-registered train seeds: exactly the 3 base presence questions (2 Yes + 1
    cross-family No — the other-family field name is a designed presence-No)."""
    for fam, seed in (("electricity", 1_005_011), ("telecom", 1_007_013),
                      ("savings", 1_009_017), ("card", 1_011_019)):
        gt = page_gt(fam, seed)
        factory = LabelFactory({}, {}, [], seed=1)
        qs, golds = factory.questions_for(fam, seed, gt, "train")
        pres = [(q, g) for q, g in zip(qs, golds, strict=False) if ":noul:pres:" in q["qid"]]
        assert len(pres) == 3, (fam, [q["qid"] for q, _ in pres])
        from row_labels import TABLES
        own_fields = {f for f, _, _ in TABLES[fam](gt)}
        nos = [(q, g) for q, g in pres if g[_yes_idx(q["options"])] != 1.0]
        yes = [(q, g) for q, g in pres if q not in [nq for nq, _ in nos]]
        assert len(nos) == 1 and nos[0][0]["qid"].rsplit(":", 1)[1] not in own_fields, \
            (fam, [(q["qid"], g) for q, g in pres])
        assert len(yes) == 2, (fam, [(q["qid"], g) for q, g in pres])


def _pres_case(family, qid, text, yes, options=("No", "Yes")):
    gold = [0.0, 0.0]
    gold[options.index("Yes" if yes else "No")] = 1.0
    return {"family": family,
            "questions": [{"qid": qid, "primitive": "noul", "text": text,
                           "options": list(options)}],
            "gold": [gold]}


def _count_presence(rows):
    yes = no = 0
    for r in rows:
        for q, g in zip(r["questions"], r["gold"], strict=False):
            if q["primitive"] == "noul" and ":noul:pres:" in q["qid"]:
                if g[_yes_idx(q["options"])] == 1.0:
                    yes += 1
                else:
                    no += 1
    return yes, no


def test_balance_train_presence_caps_yes_rows():
    """Cap: Yes <= 2x No per (family, template) group; excess Yes dropped by qid order."""
    from drishyam.vision_data import balance_train_presence
    tmpl = "Is the total amount payable present and readable on this page?"
    rows = []
    # electricity group: 9 Yes + 2 No -> cap 4 Yes, drop 5
    for i in range(9):
        rows.append(_pres_case("electricity", f"electricity:{i}:noul:pres:total_amount",
                               tmpl, yes=True))
    # shuffled option order must be detected too
    rows.append(_pres_case("electricity", "electricity:90:noul:pres:total_amount",
                           tmpl, yes=True, options=("Yes", "No")))
    for i in (91, 92):
        rows.append(_pres_case("electricity", f"electricity:{i}:noul:pres:printed_total",
                               tmpl, yes=False))
    # same template, different family -> own group, no No rows -> untouched
    for i in range(5):
        rows.append(_pres_case("card", f"card:{i}:noul:pres:total_due", tmpl, yes=True))
    # a non-presence decision on a capped page must survive
    rows.append({"family": "electricity",
                 "questions": [{"qid": "electricity:0:choice:val:total_amount",
                                "primitive": "choice", "text": "Which value ...",
                                "options": ["a", "b", "c"]}],
                 "gold": [[1.0, 0.0, 0.0]]})

    import copy
    snapshot = copy.deepcopy(rows)
    dropped = balance_train_presence(rows)
    assert dropped == 6                       # 10 Yes, 2 No -> cap 4 -> drop 6
    yes, no = _count_presence(rows)
    assert (yes, no) == (9, 2)                # 4 capped electricity + 5 untouched card
    kept_qids = [q["qid"] for r in rows for q in r["questions"]]
    # determinism: lowest qids kept
    assert "electricity:0:noul:pres:total_amount" in kept_qids
    assert "electricity:3:noul:pres:total_amount" in kept_qids
    assert "electricity:4:noul:pres:total_amount" not in kept_qids
    assert "electricity:90:noul:pres:total_amount" not in kept_qids
    assert "card:0:noul:pres:total_due" in kept_qids           # untouched group
    assert "electricity:0:choice:val:total_amount" in kept_qids  # non-presence kept
    # every surviving case keeps gold/options aligned
    for r in rows:
        assert len(r["questions"]) == len(r["gold"])
    # determinism: same input -> same outcome
    rows2 = copy.deepcopy(snapshot)
    assert balance_train_presence(rows2) == 6
    assert json.dumps(rows, sort_keys=True) == json.dumps(rows2, sort_keys=True)


def test_balance_train_presence_noop_cases():
    """Groups without No rows, or already within cap, are left untouched."""
    from drishyam.vision_data import balance_train_presence
    tmpl = "Is the due date present and readable on this page?"
    rows = [_pres_case("savings", f"savings:{i}:noul:pres:from_date", tmpl, yes=True)
            for i in range(6)]
    assert balance_train_presence(rows) == 0
    assert _count_presence(rows) == (6, 0)
    rows = ([_pres_case("telecom", f"telecom:{i}:noul:pres:due_date", tmpl, yes=True)
             for i in range(2)]
            + [_pres_case("telecom", "telecom:9:noul:pres:due_date", tmpl, yes=False)])
    assert balance_train_presence(rows) == 0          # 2 Yes == 2 x 1 No -> at cap
    assert _count_presence(rows) == (2, 1)


def test_balance_flag_wiring():
    """--balance-presence exists, defaults OFF, and applies to the train split only."""
    import inspect

    from drishyam.vision_data import main
    src = inspect.getsource(main)
    assert 'ap.add_argument("--balance-presence", action="store_true"' in src
    assert 'if a.balance_presence:' in src
    assert 'balance_train_presence(out_rows["train"])' in src
    assert 'balance_train_presence(out_rows["val"])' not in src

