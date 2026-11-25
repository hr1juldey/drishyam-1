"""Gate-2: derive typed decision rows (noul/choice/score) from the deterministic factory gt.

Source of truth for page+gt determinism: artifacts/paper_rig/jit_pipeline.py::gen_page
(same seed -> same pixels + same gt, proven property). We replicate its gt chain here
WITHOUT torch/GPU (pure-CPU generators), and the micro-test cross-checks byte-equality
against jit_pipeline.gen_page on sampled seeds.

Prior-proofing law: every question's answer is determined by THIS page's content —
distractors come from same-family same-field neighbours (near-magnitude for amounts,
same printed format) or cross-family field names, so priors cannot pass.

Deprecation note (jit_pipeline.py:66): legibility/degradation exists on THERMAL only —
score-legibility and noul-readable are therefore thermal-only primitives.

Output: artifacts/gate2_labels_{train,val,test,unseen}.jsonl
Row shape (one CASE per page, mirrors train.py's case dict):
  {"page_key": "<emb-relpath>:<row-idx>", "family": ..., "seed": ...,
   "questions": [{"qid", "primitive", "text", "options": [...]}], "gold": [[floats]]}
"""
import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = ROOT / "artifacts" / "paper_rig"
sys.path.insert(0, str(PAPER))

from content_generator import gen_card, gen_statement  # noqa: E402
import fin_docs  # noqa: E402
from institutions import inr_group  # noqa: E402
from row_labels import TABLES, _leg, UNREADABLE_OPS  # noqa: E402
from thermal_generator import apply_degradation, choose_plan, gen_thermal  # noqa: E402

EMB_DIRS = [PAPER / "embeddings", PAPER / "embeddings_slice2"]
UNSEEN_FAMILY = "table"
TRAIN_FAMILIES = ["thermal", "savings", "card", "electricity", "telecom", "salary"]
AMOUNT_FIELDS = {"total_amt", "total_amount", "total_due", "min_due",
                 "gross_earnings", "net_pay"}
INSTITUTION_FIELD = {"thermal": "shop_name", "card": "issuer", "savings": "bank",
                     "electricity": "provider", "telecom": "provider", "salary": "employer"}
FIELD_LABEL = {
    "total_amt": "total amount", "total_amount": "total amount payable",
    "total_due": "total amount due", "min_due": "minimum amount due",
    "gstin": "GSTIN", "bill_no": "bill number", "invoice_no": "invoice number",
    "date": "date", "due_date": "due date", "consumer_no": "consumer number",
    "mobile": "mobile number", "account_number": "account number",
    "n_items": "item count", "n_transactions": "transaction count",
    "provider": "provider name", "issuer": "issuer name", "bank": "bank name",
    "employer": "employer name", "shop_name": "shop name", "period": "salary period",
    "customer_name": "customer name", "employee_name": "employee name",
    "card_masked": "masked card number", "units_consumed": "units consumed",
    "gross_earnings": "gross earnings", "net_pay": "net pay",
    "subtotal_amt": "subtotal amount",
}
SCORE_OPTIONS = ["L1 most fields unreadable", "L2 several fields unreadable",
                 "L3 about half the fields unreadable", "L4 one field unreadable",
                 "L5 all fields clean and readable"]


def page_gt(family, seed):
    """gt dict for one row — byte-equivalent to jit_pipeline.gen_page(family, seed)[1]
    (rng conventions copied from jit_pipeline.py:49-58; cross-checked by micro-test)."""
    rng = np.random.RandomState(seed)
    if family == "thermal":
        page, gt = gen_thermal(rng, seed)
        drng = np.random.RandomState(seed + 1)
        k = int(drng.randint(0, 4))
        if k:
            fids = list(drng.choice(["total_amt", "date", "bill_no", "gstin",
                                     "item_0_name", "item_1_name", "item_0_amt",
                                     "subtotal_amt"], size=k, replace=False))
            gt["degradation_plan"] = choose_plan(drng, n_ops=k, candidate_fids=list(fids))
        _, applied = apply_degradation(page, gt, np.random.RandomState(seed + 2))
        gt["legibility_applied"] = applied
        return gt
    if family == "savings":
        _, gt = gen_statement(rng, seed)
    elif family == "card":
        _, gt = gen_card(rng, seed)
    else:
        _, gt = getattr(fin_docs, f"gen_{family}")(rng, seed)
    return gt


def load_pages():
    """(family, seed) -> (emb_relpath, idx); parity rule from eval_embeds.load_entries."""
    pages = {}
    for d in EMB_DIRS:
        for emb_path in sorted(glob.glob(str(d / "emb_*.npy"))):
            lab = emb_path.replace("emb_", "lab_", 1).replace(".npy", ".jsonl")
            rows = [json.loads(l) for l in open(lab) if l.strip()]
            if len(rows) != np.load(emb_path, mmap_mode="r").shape[0]:
                continue
            for i, r in enumerate(rows):
                pages[(r["family"], r["seed"])] = (os.path.relpath(emb_path, ROOT), i)
    return pages


def parse_amount(s):
    m = re.search(r"([\d,]+\.\d{2})", str(s))
    if not m:
        return None
    prefix = str(s)[: m.start()]
    return prefix, float(m.group(1).replace(",", ""))


def perturb_amount(gold_str, rng):
    pa = parse_amount(gold_str)
    if pa is None:
        return None
    prefix, val = pa
    newval = round(val * rng.choice([0.6, 0.7, 1.3, 1.4]) + rng.uniform(-9, 9), 2)
    if newval <= 0:
        newval = round(val * 1.3, 2)
    return f"{prefix}{inr_group(newval)}"


def shuffle_options(options, gold_idx, rng):
    order = list(range(len(options)))
    rng.shuffle(order)
    gold = [0.0] * len(options)
    gold[order.index(gold_idx)] = 1.0
    return [options[i] for i in order], gold


class LabelFactory:
    def __init__(self, pages, seed_pool_values, inst_pool, seed=42):
        self.pages = pages
        self.pools = seed_pool_values      # (family, field) -> [gold strings]
        self.inst_pool = inst_pool         # list of institutions across families
        self.rng = random.Random(seed)
        self._fields_cache = {}

    def family_fields(self, family, gt):
        if family not in self._fields_cache:
            try:
                self._fields_cache[family] = [f for f, _, _ in TABLES[family](gt)]
            except Exception:
                self._fields_cache[family] = []
        return self._fields_cache[family]

    def questions_for(self, family, seed, gt, split):
        qs, golds = [], []
        rng = self.rng
        try:
            pairs = TABLES[family](gt)
        except Exception:
            return [], []
        field_map = {f: a for f, _, a in pairs}
        other_families = [f for f in TRAIN_FAMILIES + [UNSEEN_FAMILY] if f != family]
        other_family = other_families[rng.randrange(len(other_families))]
        other_fields = self.family_fields(other_family, page_gt(other_family, seed + 13))

        # noul-present: Yes for 2 real fields, No for 1 field name that belongs to another family
        real = [f for f, _, a in pairs if a != "unreadable"]
        rng.shuffle(real)
        for f in real[:2]:
            opts, gold = shuffle_options(["No", "Yes"], 1, rng)
            qs.append({"qid": f"{family}:{seed}:noul:pres:{f}", "primitive": "noul",
                       "text": f"Is the {FIELD_LABEL.get(f, f)} present and readable on this page?",
                       "options": opts})
            golds.append(gold)
        if other_fields:
            f = other_fields[rng.randrange(len(other_fields))]
            opts, gold = shuffle_options(["No", "Yes"], 0, rng)
            qs.append({"qid": f"{family}:{seed}:noul:pres:{f}", "primitive": "noul",
                       "text": f"Is the {FIELD_LABEL.get(f, f)} present and readable on this page?",
                       "options": opts})
            golds.append(gold)

        # noul-readable (thermal only): per-field legibility from _leg
        if family == "thermal":
            cands = [f for f, _, _ in pairs if f in gt.get("legibility_applied", {})]
            rng.shuffle(cands)
            for f in cands[:2]:
                unreadable = _leg(gt, f) == "unreadable"
                opts, gold = shuffle_options(["No", "Yes"], 1 if not unreadable else 0, rng)
                qs.append({"qid": f"{family}:{seed}:noul:read:{f}", "primitive": "noul",
                           "text": f"Is the {FIELD_LABEL.get(f, f)} clearly readable on this receipt?",
                           "options": opts})
                golds.append(gold)
            # score-legibility (thermal only): severity buckets from UNREADABLE ops count
            n_unread = sum(1 for op in gt.get("legibility_applied", {}).values()
                           if op in UNREADABLE_OPS)
            gold_idx = max(0, min(4, 4 - n_unread))
            qs.append({"qid": f"{family}:{seed}:score:legibility", "primitive": "score",
                       "text": "How legible is this receipt overall?",
                       "options": list(SCORE_OPTIONS)})
            golds.append([1.0 if i == gold_idx else 0.0 for i in range(5)])

        # choice-value: amounts on this page vs near-magnitude perturbation + cross-page same field
        for f in [p[0] for p in pairs if p[0] in AMOUNT_FIELDS][:1]:
            gold_str = field_map.get(f)
            if gold_str in (None, "unreadable"):
                continue
            dist_cross = None
            pool = [v for v in self.pools.get((family, f), []) if v != gold_str]
            if pool:
                dist_cross = pool[rng.randrange(len(pool))]
            dist_pert = perturb_amount(gold_str, rng)
            opts = [gold_str]
            if dist_pert:
                opts.append(dist_pert)
            if dist_cross:
                opts.append(dist_cross)
            if len(opts) < 3:
                continue
            opts, gold = shuffle_options(opts, 0, rng)
            qs.append({"qid": f"{family}:{seed}:choice:val:{f}", "primitive": "choice",
                       "text": f"Which of these values is printed on this page as the "
                               f"{FIELD_LABEL.get(f, f)}?",
                       "options": opts})
            golds.append(gold)

        # choice-issuer: this page's institution vs 2 institutions from other families
        inst_f = INSTITUTION_FIELD.get(family)
        gold_inst = field_map.get(inst_f)
        if gold_inst and gold_inst != "unreadable":
            kind = {"thermal": "shop", "card": "bank", "savings": "bank",
                    "electricity": "electricity board", "telecom": "operator",
                    "salary": "company"}.get(family, "issuer")
            others = [i for i in self.inst_pool if i != gold_inst]
            if len(others) < 2:
                return qs, golds  # need 2 cross-page distractors or the choice is degenerate
            rng.shuffle(others)
            opts = [gold_inst] + others[:2]
            opts, gold = shuffle_options(opts, 0, rng)
            qs.append({"qid": f"{family}:{seed}:choice:issuer", "primitive": "choice",
                       "text": f"Which {kind} issued this page?",
                       "options": opts})
            golds.append(gold)
        return qs, golds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pages-per-family", type=int, default=3000)
    ap.add_argument("--val-pages-per-family", type=int, default=400)
    ap.add_argument("--test-pages-per-family", type=int, default=600)
    ap.add_argument("--unseen-pages", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--limit-pages", type=int, default=0, help="smoke cap")
    a = ap.parse_args()

    pages = load_pages()
    print(f"pages banked: {len(pages)}")
    caps = {"train": a.train_pages_per_family, "val": a.val_pages_per_family,
            "test": a.test_pages_per_family, "unseen": a.unseen_pages}
    sel = {"train": [], "val": [], "test": [], "unseen": []}
    per_fam = defaultdict(lambda: defaultdict(list))
    for (fam, seed), loc in pages.items():
        bucket = "unseen" if fam == UNSEEN_FAMILY else (
            "val" if seed % 10 == 8 else "test" if seed % 10 == 7 else "train")
        if fam == UNSEEN_FAMILY and seed % 10 == 7:
            bucket = "unseen"  # table test-bucket rows still unseen-only
        if len(per_fam[bucket][fam]) < caps[bucket]:
            per_fam[bucket][fam].append((fam, seed))
    for b in sel:
        for fam, items in per_fam[b].items():
            sel[b] += items[:caps[b]]
    if a.limit_pages:
        for b in sel:
            sel[b] = sel[b][: a.limit_pages]

    # pass 1: gt + field pairs -> pools for cross-page distractors
    factory_seed = random.Random(a.seed)
    pool_counts = defaultdict(int)
    pools = defaultdict(list)
    inst_pool = []
    gts = {}
    for b in sel:
        for fam, seed in sel[b]:
            if (fam, seed) in gts:
                continue
            try:
                gt = page_gt(fam, seed)
            except Exception as e:  # noqa: BLE001 — count, report, continue
                pool_counts[f"ERR:{fam}:{type(e).__name__}"] += 1
                continue
            gts[(fam, seed)] = gt
            try:
                for f, _, av in TABLES[fam](gt):
                    pools[(fam, f)].append(av)
                    pool_counts[f"{fam}:{f}"] += 1
                inst_f = INSTITUTION_FIELD.get(fam)
                inst = {f2: a2 for f2, _, a2 in TABLES[fam](gt)}.get(inst_f)
                if inst and inst != "unreadable":
                    inst_pool.append(inst)
            except Exception as e:  # noqa: BLE001
                pool_counts[f"ERR:pairs:{fam}:{type(e).__name__}"] += 1
    inst_pool = sorted(set(inst_pool))
    print(f"gt pages: {len(gts)}, institutions: {len(inst_pool)}, pool errors: "
          f"{ {k: v for k, v in pool_counts.items() if k.startswith('ERR')} or 'none'}")

    lf = LabelFactory(pages, pools, inst_pool, seed=a.seed)
    out_rows = {"train": [], "val": [], "test": [], "unseen": []}
    for b in sel:
        for fam, seed in sel[b]:
            if (fam, seed) not in gts:
                continue
            qs, golds = lf.questions_for(fam, seed, gts[(fam, seed)], b)
            if not qs:
                continue
            out_rows[b].append({
                "page_key": f"{pages[(fam, seed)][0]}:{pages[(fam, seed)][1]}",
                "family": fam, "seed": seed,
                "questions": qs, "gold": golds,
            })

    stats = {}
    for b, rows in out_rows.items():
        path = ROOT / "artifacts" / f"gate2_labels_{b}.jsonl"
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        prim_counts = defaultdict(int)
        for r in rows:
            for q in r["questions"]:
                prim_counts[q["primitive"]] += 1
        fam_counts = defaultdict(int)
        for r in rows:
            fam_counts[r["family"]] += 1
        for r in rows:  # sanity: every gold one-hot sums to 1 and matches options length
            for q, g in zip(r["questions"], r["gold"], strict=False):
                assert abs(sum(g) - 1.0) < 1e-6 and len(g) == len(q["options"]), (q["qid"], g)
        stats[b] = {"pages": len(rows), "decisions": sum(prim_counts.values()),
                    "by_primitive": dict(prim_counts), "by_family": dict(fam_counts)}
        print(b, json.dumps(stats[b]))

    payload = json.dumps(stats, sort_keys=True) + json.dumps(a.__dict__, sort_keys=True)
    stats["cfg_sha256"] = hashlib.sha256(payload.encode()).hexdigest()[:12]
    (ROOT / "artifacts" / "gate2_label_stats.json").write_text(json.dumps(stats, indent=2))
    print("WROTE artifacts/gate2_label_stats.json sha=", stats["cfg_sha256"])
    if a.mlflow:
        import mlflow
        from drishyam.tracking import mlflow_run
        with mlflow_run("g2-labels") as run:
            mlflow.set_tag("gate", "2")
            mlflow.log_params({k: v for k, v in a.__dict__.items()})
            for b, s in stats.items():
                if isinstance(s, dict) and "decisions" in s:
                    mlflow.log_metric(f"{b}_pages", s["pages"])
                    mlflow.log_metric(f"{b}_decisions", s["decisions"])
                    for p, n in s["by_primitive"].items():
                        mlflow.log_metric(f"{b}_{p}", n)
            mlflow.log_artifact(str(ROOT / "artifacts" / "gate2_label_stats.json"))
            mlflow.log_artifact(str(ROOT / "artifacts" / "gate2_labels_train.jsonl"))
            for b in ("val", "test", "unseen"):
                mlflow.log_artifact(str(ROOT / "artifacts" / f"gate2_labels_{b}.jsonl"))
            print("mlflow run:", run.info.run_id)


if __name__ == "__main__":
    main()
