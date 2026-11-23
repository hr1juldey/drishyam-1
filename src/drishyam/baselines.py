"""G1-B1: trivial baselines — per-question majority class + TF-IDF char-ngram calibrated LogReg."""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from drishyam.dataset import CaseDataset
from drishyam.train import ece15

ART = Path("artifacts")


def jitter_states_by_case() -> dict[tuple[str, str], str]:
    """(workflow, case_id) -> jittered_state from the deterministic D3 suite artifact."""
    path = ART / "jitter_suite.jsonl"
    if not path.exists():
        raise SystemExit("missing artifacts/jitter_suite.jsonl — run drishyam.stress first")
    out = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        out[(r["workflow"], r["case_id"])] = r["jittered_state"]
    return out


def majority_dist(golds: list[list[float]]) -> np.ndarray:
    """Empirical mean of gold distributions = the per-question majority predictor."""
    return np.mean(np.asarray(golds, dtype=float), axis=0)


def ovr_auc(probs: list, golds: list) -> float:
    """Mean one-vs-rest ROC AUC of p(option) vs (gold argmax == option); folds without both
    classes are skipped. Implemented without sklearn.metrics to avoid small-fold edge cases."""
    import numpy as _np

    aucs = []
    k = probs[0].shape[0]
    for c in range(k):
        y = _np.array([1.0 if g.argmax() == c else 0.0 for g in golds])
        if y.min() == y.max():
            continue
        s = _np.array([p[c] for p in probs])
        order = _np.argsort(s)
        ranks = _np.empty(len(s))
        ranks[order] = _np.arange(1, len(s) + 1)
        # average ranks for ties
        s_sorted = s[order]
        i = 0
        while i < len(s_sorted):
            j = i
            while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
                j += 1
            if j > i:
                ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2
            i = j + 1
        n_pos = y.sum()
        n_neg = len(y) - n_pos
        aucs.append(float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)))
    return float(np.mean(aucs)) if aucs else float("nan")


def metrics_rows(rows: list[dict]) -> dict:
    """rows: {prim, probs, gold} ragged — hand-computed fixtures in tests/test_baselines.py."""
    acc = float(np.mean([float(r["probs"].argmax() == r["gold"].argmax()) for r in rows]))
    brier = float(np.mean([float(((r["probs"] - r["gold"]) ** 2).sum()) for r in rows]))
    score_rows = [r for r in rows if r["prim"] == "score"]
    score_mae = (float(np.mean([abs(float(r["probs"].argmax()) - float(r["gold"].argmax()))
                                for r in score_rows]))
                 if score_rows else float("nan"))
    # AUC pools questions by option count (ragged: noul=2, choice=3-4, score=4 cannot share a pool)
    by_len: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        by_len[len(r["probs"])].append(r)
    aucs = [ovr_auc([r["probs"] for r in g], [r["gold"] for r in g])
            for g in by_len.values() if len(g) >= 8]
    return {
        "acc": acc, "brier": brier, "score_mae": score_mae,
        "ece15": ece15([r["probs"] for r in rows], [r["gold"] for r in rows]),
        "ovr_auc": float(np.nanmean(aucs)) if aucs else float("nan"),
        "n": len(rows),
    }


def question_key(case: dict, i: int) -> str:
    return f"{case['workflow']}::{case['questions'][i].qid}"


def fit_baselines(train_cases: list[dict]) -> dict:
    """Per (workflow::qid): majority dist + char-TFIDF LogReg on the modal gold option."""
    by_q: dict[str, list[dict]] = defaultdict(list)
    for case in train_cases:
        for i, _q in enumerate(case["questions"]):
            by_q[question_key(case, i)].append({"case": case, "i": i})
    models = {}
    for key, items in by_q.items():
        golds = [np.asarray(c["case"]["gold"][c["i"]], dtype=float) for c in items]
        prim = items[0]["case"]["questions"][items[0]["i"]].primitive.value
        dist = majority_dist([list(g) for g in golds])
        y = [int(g.argmax()) for g in golds]
        texts = [c["case"]["state"] for c in items]
        if len(set(y)) > 1:
            pipe = make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                min_df=2, sublinear_tf=True),
                LogisticRegression(max_iter=1000, C=1.0),
            )
            pipe.fit(texts, y)
        else:
            pipe = None  # constant class — LogReg undefined; majority covers it
        models[key] = {"majority": dist, "logreg": pipe, "prim": prim, "classes": sorted(set(y))}
    return models


def predict_rows(models: dict, cases: list[dict],
                 states_override: list[list[str]] | None = None) -> list[dict]:
    rows = []
    for ci, case in enumerate(cases):
        for i, q in enumerate(case["questions"]):
            key = question_key(case, i)
            mdl = models[key]
            state = case["state"]
            if states_override is not None:
                state = states_override[ci]
            if mdl["logreg"] is not None:
                p = mdl["logreg"].predict_proba([state])[0]
                # align sklearn class order to option order
                aligned = np.zeros(len(mdl["majority"]))
                for cls, pr in zip(mdl["logreg"].classes_, p, strict=False):
                    aligned[cls] = pr
                probs = aligned
            else:
                probs = mdl["majority"].copy()
            rows.append({
                "case_id": case["case_id"], "qid": q.qid, "prim": q.primitive.value,
                "probs": probs, "gold": np.asarray(case["gold"][i], dtype=float),
            })
    return rows


def fit_priors(train_cases: list[dict]) -> dict:
    """Seen-only PRIMITIVE-LEVEL priors for zero-shot unseen-workflow eval: option NAMES do
    not align across workflows for choice (uniform there); noul (false/true) and score
    (levels 0..K-1) are index-aligned, so pooled dists + pooled LogReg are honest."""
    pools: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    for case in train_cases:
        for i, q in enumerate(case["questions"]):
            k = len(q.options)
            pools[f"{q.primitive.value}::K{k}"].append((case["state"], case["gold"][i]))
    priors: dict[str, dict] = {}
    for key, items in pools.items():
        golds = [np.asarray(g, dtype=float) for _, g in items]
        dist = majority_dist([list(g) for g in golds])
        y = [int(g.argmax()) for g in golds]
        pipe = None
        if len(set(y)) > 1:
            pipe = make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                min_df=2, sublinear_tf=True),
                LogisticRegression(max_iter=1000, C=1.0),
            )
            pipe.fit([t for t, _ in items], y)
        priors[key] = {"dist": dist, "logreg": pipe, "classes": sorted(set(y))}
    return priors


def predict_unseen_rows(priors: dict, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        for i, q in enumerate(case["questions"]):
            k = len(q.options)
            pr = priors.get(f"{q.primitive.value}::K{k}")
            if pr is None:
                probs = np.full(k, 1.0 / k)  # no seen pool for this (prim,K) — honest uniform
            elif pr["logreg"] is not None:
                p = pr["logreg"].predict_proba([case["state"]])[0]
                aligned = np.zeros(len(pr["dist"]))
                for cls, pv in zip(pr["logreg"].classes_, p, strict=False):
                    aligned[cls] = pv
                probs = aligned if aligned.sum() > 0 else pr["dist"].copy()
                probs = probs / probs.sum()
            else:
                probs = pr["dist"].copy()
            rows.append({
                "case_id": case["case_id"], "qid": q.qid,
                "prim": q.primitive.value,
                "probs": probs, "gold": np.asarray(case["gold"][i], dtype=float),
            })
    return rows


def main() -> None:
    from drishyam.splits import SEEN, UNSEEN

    spec = json.loads(Path("splits/split_v1.json").read_text())
    train = []
    for wf in SEEN:
        ds = CaseDataset(wf, "train", spec["train"][wf])
        train += [ds[i] for i in range(len(ds))]
    models = fit_baselines(train)

    test = []
    for wf in SEEN:
        ds = CaseDataset(wf, "test", list(range(100)))
        test += [ds[i] for i in range(len(ds))]
    unseen = []
    ds = CaseDataset(UNSEEN, "test", list(range(100)))
    unseen += [ds[i] for i in range(len(ds))]
    val = []
    for wf in SEEN:
        ds = CaseDataset(wf, "train", spec["val"][wf])
        val += [ds[i] for i in range(len(ds))]

    report = {
        "val": metrics_rows(predict_rows(models, val)),
        "test": metrics_rows(predict_rows(models, test)),
        "unseen_test": metrics_rows(predict_unseen_rows(fit_priors(train), unseen)),
    }

    # case/whitespace jitter: predict clean vs jittered on the same 300 test cases
    jmap = jitter_states_by_case()
    flips = {}
    for wf in SEEN:
        ds = CaseDataset(wf, "test", list(range(100)))
        cs = [ds[i] for i in range(len(ds))]
        rc = predict_rows(models, cs)
        cs_j = [dict(c, state=jmap[(wf, c["case_id"])]) for c in cs]
        rj = predict_rows(models, cs_j)
        flips[wf] = float(np.mean([a["probs"].argmax() != b["probs"].argmax()
                                   for a, b in zip(rc, rj, strict=False)]))
    report["jitter_flip_rate"] = float(np.mean(list(flips.values())))
    report["jitter_per_workflow"] = flips

    ART.mkdir(exist_ok=True)
    (ART / "baseline_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
