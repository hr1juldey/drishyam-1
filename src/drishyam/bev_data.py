"""bev-decision (avbiswas/bev-decision) -> drishyam typed-question cases (TEXT track).

Row contract (runs/hub_ref/jfbench_research.md §2, re-verified by full audit on
2026-11-26 against revision 11ed9b5b879e7f9524e7c89c1037d498c54206f0):
  5 columns: state (large_string), questions_json (large_string), domain (string),
  question_types (list[str]), question_count (int16).
  questions_json = {name: {type, instructions, criteria, label}}, where
    choice: criteria = {answer_key: description}; label = one key
            -> options = criteria keys in stored order, gold at the label's index
    noul:   label = bool (no criteria)
            -> options = ["No", "Yes"] (repo convention), gold = 1 if label else 0
    score:  criteria = ordered list of anchored levels; label = zero-based index
            -> options = criteria levels, one-hot gold at label
  Audit of default train+test (150,000 rows / 296,582 questions): zero contract
  violations; min 2 choice options, min 3 score levels, 6 empty states in train
  (skipped + reported). No val split exists (dataset card, §5) — the trainer carves
  one out of train; the test split stays sealed for eval.

Offline law (HF_HUB_OFFLINE=1): the `datasets` library cannot resolve a hub repo
from cache, so parquets are resolved through huggingface_hub's snapshot dir and
read with pyarrow directly. Snapshot must already be in the HF cache; if missing
we download it once (online) and pin the revision from then on.
"""

import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError
from pydantic import ValidationError

from drishyam.schemas import Primitive, Question

REPO_ID = "avbiswas/bev-decision"
PIN = "11ed9b5b879e7f9524e7c89c1037d498c54206f0"  # main, per research file §1
CONFIG_DIRS = {
    "default": "data",
    "hard_50k": "hard_50k",
    "numeric_temporal": "numeric_temporal",
    "skills": "skills",
    "counterfactual_15k": "counterfactual_15k",
}
NOUL_OPTIONS = ["No", "Yes"]
TYPES = {p.value for p in Primitive}


def resolve_snapshot_dir(revision: str = PIN) -> Path:
    """Cached snapshot first (offline law); download once if the pin is absent."""
    try:
        return Path(
            snapshot_download(
                REPO_ID, repo_type="dataset", revision=revision, local_files_only=True
            )
        )
    except LocalEntryNotFoundError:
        return Path(snapshot_download(REPO_ID, repo_type="dataset", revision=revision))


def question_from_dict(qid: str, q: dict) -> tuple[Question, list[float]]:
    """bev-decision question object -> (drishyam Question, one-hot gold)."""
    if not isinstance(q, dict) or q.get("type") not in TYPES:
        raise ValueError(f"{qid}: unknown question type {q.get('type')!r}")
    prim = Primitive(q["type"])
    text = q.get("instructions")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{qid}: missing instructions")
    if prim is Primitive.NOUL:
        label = q.get("label")
        if not isinstance(label, bool):
            raise ValueError(f"{qid}: noul label must be bool, got {label!r}")
        gold = [0.0] * len(NOUL_OPTIONS)
        gold[1 if label else 0] = 1.0
        return Question(qid=qid, primitive=prim, text=text, options=list(NOUL_OPTIONS)), gold
    criteria = q.get("criteria")
    if prim is Primitive.CHOICE:
        if not isinstance(criteria, dict) or len(criteria) < 2:
            raise ValueError(f"{qid}: choice criteria must be a dict with >=2 keys")
        options = list(criteria.keys())
        label = q.get("label")
        if label not in criteria:
            raise ValueError(f"{qid}: choice label {label!r} not in criteria keys")
        gold = [1.0 if k == label else 0.0 for k in options]
        return Question(qid=qid, primitive=prim, text=text, options=options), gold
    # score: criteria = ordered anchored levels, label = zero-based index
    if not isinstance(criteria, list) or len(criteria) < 2:
        raise ValueError(f"{qid}: score criteria must be a list with >=2 levels")
    if not all(isinstance(c, str) and c.strip() for c in criteria):
        raise ValueError(f"{qid}: score levels must be non-empty strings")
    label = q.get("label")
    if not isinstance(label, int) or isinstance(label, bool) or not 0 <= label < len(criteria):
        raise ValueError(f"{qid}: score label {label!r} out of range")
    gold = [1.0 if i == label else 0.0 for i in range(len(criteria))]
    return Question(qid=qid, primitive=prim, text=text, options=list(criteria)), gold


def parse_row(idx: int, row: dict, config: str, split: str) -> dict | None:
    """One parquet row -> trainer case dict; None for a stateless row (skip)."""
    state = row.get("state")
    if not isinstance(state, str) or not state.strip():
        return None
    raw = json.loads(row["questions_json"])
    if not isinstance(raw, dict) or not raw:
        return None
    if row.get("question_count") != len(raw):
        raise ValueError(
            f"row {idx}: question_count {row.get('question_count')} != {len(raw)} questions"
        )
    try:
        pairs = [question_from_dict(qid, q) for qid, q in raw.items()]
    except (ValueError, ValidationError) as e:
        raise ValueError(f"row {idx}: {e}") from e
    types_seen = {q.primitive.value for q, _ in pairs}
    types_declared = set(row.get("question_types") or [])
    if types_declared and types_declared != types_seen:
        raise ValueError(
            f"row {idx}: question_types {sorted(types_declared)} != {sorted(types_seen)}"
        )
    return {
        "case_id": f"bev:{config}:{split}:{idx}",
        "state": state,
        "domain": row.get("domain", ""),
        "questions": [q for q, _ in pairs],
        "gold": [g for _, g in pairs],
    }


def cases_from_parquet(
    path: str | Path, config: str = "default", split: str = "train", limit: int = 0
) -> tuple[list[dict], int]:
    """Parquet -> (cases, skipped_stateless). Pure-local entry point (tests/smokes)."""
    t = pq.read_table(path)
    cols = {
        c: t.column(c).to_pylist()
        for c in ("state", "questions_json", "domain", "question_types", "question_count")
    }
    cases, skipped = [], 0
    for i in range(t.num_rows):
        if limit and len(cases) >= limit:
            break
        row = {c: cols[c][i] for c in cols}
        case = parse_row(i, row, config, split)
        if case is None:
            skipped += 1
            continue
        cases.append(case)
    return cases, skipped


def load_cases(
    config: str = "default", split: str = "train", limit: int = 0, revision: str = PIN
) -> tuple[list[dict], int]:
    """Resolve the pinned snapshot (offline-first) and parse one parquet split."""
    if config not in CONFIG_DIRS:
        raise ValueError(f"config {config!r} not in {sorted(CONFIG_DIRS)}")
    if split not in ("train", "test"):
        raise ValueError(
            f"split {split!r} not in ['train', 'test'] (bev-decision has no val split)"
        )
    path = resolve_snapshot_dir(revision) / CONFIG_DIRS[config] / f"{split}.parquet"
    return cases_from_parquet(path, config=config, split=split, limit=limit)


def carve_val(cases: list[dict], n_val: int, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Deterministic val carve out of a split that has none (seeded permutation)."""
    if not 0 < n_val < len(cases):
        raise ValueError(f"n_val={n_val} out of range for {len(cases)} cases")
    order = list(range(len(cases)))
    import random

    random.Random(seed).shuffle(order)
    val_idx = set(order[:n_val])
    train = [c for i, c in enumerate(cases) if i not in val_idx]
    val = [c for i, c in enumerate(cases) if i in val_idx]
    return train, val
