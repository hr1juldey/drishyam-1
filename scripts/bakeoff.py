"""G2-TF1 bake-off + night drafter: teacher signals via local ollama, OFFLINE batch only.

Bake-off: both drafters relabel the same 50 gold train cases (3 samples, temp 0.7);
reports mean soft-KL to gold, argmax agreement, and cases/hour -> drafter decision.
Drafter: winning config drafts NEW seen-3 states (quarantined under drafts/).
"""

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[2]
OLLAMA = "http://127.0.0.1:11434/api/chat"


def call(model: str, prompt: str, temperature: float, seed: int, num_predict: int = 320):
    r = requests.post(OLLAMA, json={
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": {"temperature": temperature, "num_predict": num_predict,
                    "seed": seed, "num_thread": 8},
    }, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


def extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def label_prompt(state: str, questions: dict) -> str:
    return (
        "You are labeling operational decision cases. Output ONLY JSON, no explanation.\n"
        "For EACH question output a probability distribution over its option keys,\n"
        "summing to 1.\n\n"
        f"STATE:\n{state}\n\nQUESTIONS:\n{json.dumps(questions, ensure_ascii=False)}\n\n"
        'OUTPUT: {"answers": {"<qid>": {"<option_key>": p, ...}}}'
    )


def _num(x) -> float:
    """Null-safe coercion: teacher JSON may contain explicit nulls."""
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0.0


def mean_kl(draft: dict, gold: dict) -> float:
    """KL(gold || draft) averaged over questions with matching keys."""
    kls = []
    for qid, g in gold.items():
        keys = list(g.keys())
        gv = np.array([_num(gold[qid][k]) for k in keys]) + 1e-9
        if qid not in draft:
            continue
        dv = np.array([_num(draft[qid].get(k)) for k in keys]) + 1e-9
        gv, dv = gv / gv.sum(), dv / dv.sum()
        kls.append(float((gv * np.log(gv / dv)).sum()))
    return float(np.mean(kls)) if kls else float("nan")


def sample_cases(n: int, seed: int) -> list[dict]:
    from datasets import load_dataset

    from drishyam.data_inspect import REPO
    from drishyam.splits import SEEN

    rng = random.Random(seed)
    picked = []
    per = n // 3 + 1
    for wf in SEEN:
        ds = load_dataset(REPO, wf, split="train")
        idxs = rng.sample(range(len(ds)), min(per, len(ds)))
        for i in idxs:
            r = ds[i]
            picked.append({
                "workflow": wf, "state": r["state"],
                "questions": json.loads(r["questions"]),
                "gold": {q: g["probabilities"] for q, g in json.loads(r["gold"]).items()},
            })
    return picked[:n]


def bakeoff(models: list[str], n_cases: int, samples: int, seed: int) -> dict:
    cases = sample_cases(n_cases, seed)
    out = {"n_cases": len(cases), "samples": samples, "models": {}}
    for model in models:
        t0 = time.time()
        kls, agree, ok = [], [], 0
        for c in cases:
            prompt = label_prompt(c["state"], c["questions"])
            sample_ps = []
            for s in range(samples):
                txt = call(model, prompt, 0.7, seed * 100 + s)
                j = extract_json(txt)
                if j and "answers" in j:
                    sample_ps.append(j["answers"])
            if not sample_ps:
                continue
            ok += 1
            keys = {q: list(g.keys()) for q, g in c["gold"].items()}
            acc_p = {q: {k: float(np.mean([_num(p.get(q, {}).get(k)) for p in sample_ps]))
                         for k in ks} for q, ks in keys.items()}
            kls.append(mean_kl(acc_p, c["gold"]))
            agree.append(all(
                max(acc_p[q], key=acc_p[q].get) == max(g, key=g.get)
                for q, g in c["gold"].items() if q in acc_p))
        wall = time.time() - t0
        out["models"][model] = {
            "ok_cases": ok, "mean_soft_kl": float(np.nanmean(kls)),
            "argmax_agree_frac": float(np.mean(agree)) if agree else float("nan"),
            "cases_per_hour": ok / (wall / 3600), "wall_min": wall / 60,
        }
        print(model, out["models"][model], flush=True)
    return out


def drafter(model: str, budget_epoch: int, max_cases: int, seed: int) -> None:
    from datasets import load_dataset

    from drishyam.data_inspect import REPO
    from drishyam.splits import SEEN

    out = ROOT / "drafts" / "seen3_drafts.jsonl"
    out.parent.mkdir(exist_ok=True)
    templates = {wf: json.loads(load_dataset(REPO, wf, split="train")[0]["questions"])
                 for wf in SEEN}
    done = fails = 0
    while time.time() < budget_epoch and done < max_cases and fails < 3:
        wf = SEEN[done % 3]
        prompt = (
            f"Write ONE realistic, specific {wf.replace('_', ' ')} scenario description "
            "(120-180 words, concrete details, numbers, dates). Then answer the fixed "
            "question template for it.\n"
            f"QUESTION TEMPLATE:\n{json.dumps(templates[wf], ensure_ascii=False)}\n\n"
            'OUTPUT ONLY JSON: {"state": "...", "answers": {"<qid>": {"<key>": p, ...}}}'
        )
        try:
            j = extract_json(call(model, prompt, 0.8, seed + done))
        except requests.RequestException:
            j = None
        if not j or "state" not in j or "answers" not in j:
            fails += 1
            continue
        fails = 0
        done += 1
        with out.open("a") as f:
            f.write(json.dumps({"workflow": wf, "state": j["state"],
                                "questions": templates[wf], "answers": j["answers"],
                                "drafted_by": model, "samples": 1}, ensure_ascii=False) + "\n")
    print(f"drafted {done} cases ({fails} consecutive fails at stop)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bakeoff", "draft"], required=True)
    ap.add_argument("--model", default="gemma4:31b")
    ap.add_argument("--cases", type=int, default=50)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--budget-hours", type=float, default=4.0)
    ap.add_argument("--max-cases", type=int, default=400)
    ap.add_argument("--out", default="artifacts/bakeoff.json")
    a = ap.parse_args()
    if a.mode == "bakeoff":
        rep = bakeoff(a.model.split("+"), a.cases, a.samples, a.seed)
        (ROOT / a.out).write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))
    else:
        drafter(a.model, time.time() + a.budget_hours * 3600, a.max_cases, a.seed)


if __name__ == "__main__":
    main()
