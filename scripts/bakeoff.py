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

ROOT = Path(__file__).resolve().parents[1]
OLLAMA = "http://127.0.0.1:11434/api/chat"


def call(model: str, prompt: str, temperature: float, seed: int, num_predict: int = 320,
         num_gpu: int | None = None):
    opts = {"temperature": temperature, "num_predict": num_predict,
            "seed": seed, "num_thread": 8}
    if num_gpu is not None:
        opts["num_gpu"] = num_gpu  # 0 = CPU-only by law on training nights
    r = requests.post(OLLAMA, json={
        "model": model, "stream": False, "think": False,
        "messages": [{"role": "user", "content": prompt}],
        "options": opts,
    }, timeout=1800)
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


def label_prompt(state: str, questions: dict, examples: list[dict] | None = None) -> str:
    head = (
        "You are labeling operational decision cases. Output ONLY JSON, no explanation.\n"
        "For EACH question output a probability distribution over its option keys,\n"
        "summing to 1.\n"
        "ANSWER KEYS MUST BE EXACTLY THE BARE OPTION KEY — the part BEFORE the ':' of the\n"
        "option string, e.g. option 'approve: Matches the order...' gets key 'approve'.\n"
        "Never include the description text in the key.\n"
    )
    mid = ""
    if examples:
        mid = "Here are two worked examples showing the expected style and strictness:\n\n"
        for i, ex in enumerate(examples, 1):
            expected = json.dumps({"answers": ex["answers"]}, ensure_ascii=False)
            mid += (f"EXAMPLE {i}:\nSTATE:\n{ex['state']}\nQUESTIONS:\n"
                    f"{json.dumps(ex['questions'], ensure_ascii=False)}\n"
                    f"EXPECTED OUTPUT:\n{expected}\n\n")
    now = (f"NOW LABEL THIS CASE:\nSTATE:\n{state}\n\nQUESTIONS:\n"
           f"{json.dumps(questions, ensure_ascii=False)}\n\n"
           'OUTPUT: {"answers": {"<qid>": {"<option_key>": p, ...}}}')
    return head + mid + now


def few_shot_examples(seed: int = 42) -> list[dict]:
    """Two worked gold examples from TEST rows of seen workflows — the bake-off
    samples from TRAIN, so these can never leak into the measurement."""
    from datasets import load_dataset

    from drishyam.data_inspect import REPO
    from drishyam.splits import SEEN

    out = []
    for wf, i in ((SEEN[0], 1), (SEEN[1], 2)):
        r = load_dataset(REPO, wf, split="test")[i]
        gold = json.loads(r["gold"])
        answers = {q: g["probabilities"] for q, g in gold.items()}
        questions = json.loads(r["questions"])
        questions = {
            qid: {k: v for k, v in q.items() if k != "criteria"}
            | {"options": q.get("criteria", {})}
            for qid, q in questions.items()
        }
        out.append({"state": r["state"], "questions": questions, "answers": answers})
    return out


def _num(x) -> float:
    """Null-safe coercion: teacher JSON may contain explicit nulls."""
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else 0.0


def build_alias_map(questions: dict) -> dict:
    """qid -> {alias_lower: gold_key}, built from the dataset's own option strings.

    Options appear either as {key: description} (choice/noul) or as a LIST of
    'rank: Name: description' strings (score). Aliases per gold key: the key
    itself, the full option string, the name segment, and the description's
    first word — all lower-cased, period-stripped.
    """
    table: dict = {}

    def add(qid: str, key: str, *aliases) -> None:
        slot = table.setdefault(qid, {})
        for a in aliases:
            if not a:
                continue
            a = str(a).strip().rstrip(".,;:").lower()
            if a and slot.setdefault(a, key) != key:
                pass  # first alias wins; collisions leave the earliest mapping

    for qid, q in questions.items():
        crit = q.get("criteria", q.get("options", {}))
        if isinstance(crit, dict):
            for key, desc in crit.items():
                add(qid, key, key, desc, str(desc).split(";")[0].split(":")[0])
        elif isinstance(crit, list):
            for entry in crit:
                parts = str(entry).split(":", 2)
                key = parts[0].strip()
                name = parts[1].strip() if len(parts) > 1 else ""
                desc = parts[2].strip() if len(parts) > 2 else ""
                name_head = name.split(";")[0].strip()
                add(qid, key, key, entry, name, name.split()[0] if name.split() else "",
                    name_head, name_head.split()[0] if name_head.split() else "",
                    desc.split()[0] if desc.split() else "")
    return table


def answers_to_gold_keys(ans: dict, gold: dict,
                         alias: dict | None = None) -> dict | None:
    """Map teacher answers onto gold option keys. Accepts {qid: {key: p}} or {qid: [p,...]}.

    Resolution order per teacher key: exact gold key -> alias table (built from the
    dataset's option strings, e.g. 'Routine' -> '1' for option '1: Routine; ...') ->
    prefix before ':'. Anything unresolvable FAILS the question (fail-closed) —
    silent zero-fill manufactured the below-random bake-off artifact.
    """
    out: dict = {}
    for qid, g in gold.items():
        keys = list(g.keys())
        amap = (alias or {}).get(qid, {})
        v = ans.get(qid)
        if isinstance(v, dict):
            resolved = {}
            for tk, tv in v.items():
                tk_clean = str(tk).strip().rstrip(".").lower()
                gk = None
                if tk in keys:
                    gk = tk
                elif tk_clean in amap:
                    gk = amap[tk_clean]
                else:
                    bare = str(tk).split(":", 1)[0].strip().lower()
                    if bare in amap:
                        gk = amap[bare]
                    elif bare in [k.lower() for k in keys]:
                        gk = next(k for k in keys if k.lower() == bare)
                if gk is None:
                    return None  # untranslatable — reject, never invent
                resolved[gk] = resolved.get(gk, 0.0) + _num(tv)
            out[qid] = {k: resolved.get(k, 0.0) for k in keys}
        elif isinstance(v, (list, tuple)) and len(v) == len(keys):
            out[qid] = {k: _num(x) for k, x in zip(keys, v, strict=True)}
        else:
            return None
    return out


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


def bakeoff(models: list[str], n_cases: int, samples: int, seed: int,
            num_gpu: int | None = None, few_shot: bool = False) -> dict:
    cases = sample_cases(n_cases, seed)
    ex = few_shot_examples(seed) if few_shot else None
    out = {"n_cases": len(cases), "samples": samples, "few_shot": few_shot, "models": {}}
    for model in models:
        t0 = time.time()
        kls, agree, ok = [], [], 0
        by_type: dict = {}
        for c in cases:
            prompt = label_prompt(c["state"], c["questions"], examples=ex)
            amap = build_alias_map(c["questions"]) if isinstance(c["questions"], dict) else {}
            sample_ps = []
            for s in range(samples):
                txt = call(model, prompt, 0.7, seed * 100 + s, num_gpu=num_gpu)
                j = extract_json(txt)
                if j and "answers" in j:
                    norm = answers_to_gold_keys(j["answers"], c["gold"], alias=amap)
                    if norm is not None:
                        sample_ps.append(norm)
            if not sample_ps:
                continue
            ok += 1
            acc_p = {q: {k: float(np.mean([p[q][k] for p in sample_ps])) for k in ks}
                     for q, ks in ((q, list(g.keys())) for q, g in c["gold"].items())}
            kls.append(mean_kl(acc_p, c["gold"]))
            if isinstance(c["questions"], dict):
                qtypes = {qid: q.get("type", "?") for qid, q in c["questions"].items()}
            else:
                qtypes = {q.get("qid", q.get("id", i)): q.get("type", "?")
                          for i, q in enumerate(c["questions"])}
            for q, g in c["gold"].items():
                if q not in acc_p:
                    continue
                am_ok = max(acc_p[q], key=acc_p[q].get) == max(g, key=g.get)
                qtype = qtypes.get(q, "?")
                slot = by_type.setdefault(qtype, {"n": 0, "agree": 0})
                slot["n"] += 1
                slot["agree"] += int(am_ok)
            agree.append(all(
                max(acc_p[q], key=acc_p[q].get) == max(g, key=g.get)
                for q, g in c["gold"].items() if q in acc_p))
        wall = time.time() - t0
        out["models"][model] = {
            "ok_cases": ok, "mean_soft_kl": float(np.nanmean(kls)),
            "argmax_agree_frac": float(np.mean(agree)) if agree else float("nan"),
            "by_question_type": {k: {"n": v["n"], "agree_frac": v["agree"] / v["n"]}
                                 for k, v in sorted(by_type.items())},
            "cases_per_hour": ok / (wall / 3600), "wall_min": wall / 60,
        }
        print(model, out["models"][model], flush=True)
    return out


def drafter(model: str, budget_epoch: int, max_cases: int, seed: int,
            num_gpu: int | None = None) -> None:
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
            # state + full answers JSON needs ~350-500 tokens; 320 truncated every call
            j = extract_json(call(model, prompt, 0.8, seed + done,
                                  num_predict=900, num_gpu=num_gpu))
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
    ap.add_argument("--few-shot", action="store_true",
                    help="prepend 2 worked gold examples (test rows, no leakage)")
    ap.add_argument("--num-gpu", type=int, default=None,
                    help="ollama num_gpu layers; omit = server default, 0 = CPU-only")
    ap.add_argument("--out", default="artifacts/bakeoff.json")
    a = ap.parse_args()
    if a.mode == "bakeoff":
        rep = bakeoff(a.model.split("+"), a.cases, a.samples, a.seed,
                      num_gpu=a.num_gpu, few_shot=a.few_shot)
        (ROOT / a.out).write_text(json.dumps(rep, indent=2))
        print(json.dumps(rep, indent=2))
    else:
        drafter(a.model, time.time() + a.budget_hours * 3600, a.max_cases, a.seed,
                num_gpu=a.num_gpu)


if __name__ == "__main__":
    main()
