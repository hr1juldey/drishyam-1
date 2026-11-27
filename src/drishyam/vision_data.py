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

import fin_docs  # noqa: E402
from content_generator import gen_card, gen_statement  # noqa: E402
from institutions import inr_group  # noqa: E402
from row_labels import TABLES, UNREADABLE_OPS, _leg  # noqa: E402
from thermal_generator import apply_degradation, choose_plan, gen_thermal  # noqa: E402

EMB_DIRS = [PAPER / "embeddings", PAPER / "embeddings_slice2", PAPER / "embeddings_k5"]
# LOFO unseen family = salary: the banked 90k embeddings contain NO `table` pages
# (generation stopped before the table block of the family-ordered manifest), so the
# originally planned table holdout is impossible. salary = most structurally distinct
# banked family (payslip vs receipt/statement/card/bill). Measured in
# gate2_label_stats.json: train+val+test carry the other five families only.
UNSEEN_FAMILY = "salary"
TRAIN_FAMILIES = ["thermal", "savings", "card", "electricity", "telecom"]
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
    # K5-v4: gt-key names of omittable fields whose extractor pair uses a different
    # name — mapped to the SAME label so twin presence-No rows and base presence-Yes
    # rows share one template text (required by balance_train_presence grouping).
    "printed_total": "total amount payable",
    "min_amount_due": "minimum amount due",
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
        # K5 counterfactual twins: fields the omission registry dropped from THIS page
        # get the exact same question text with gold No — template fixed, page decides.
        for f in sorted(fin_docs.omitted_fields(family, seed))[:2]:
            opts, gold = shuffle_options(["No", "Yes"], 0, rng)
            qs.append({"qid": f"{family}:{seed}:noul:pres:{f}", "primitive": "noul",
                       "text": f"Is the {FIELD_LABEL.get(f, f)} present and readable on this page?",
                       "options": opts})
            golds.append(gold)
        # K5 thermal presence variance: unreadable fields get presence-No (label-only,
        # same template as the Yes variant, answer depends on the page's legibility)
        if family == "thermal":
            unread = [f for f, _, a in pairs if a == "unreadable"]
            rng.shuffle(unread)
            for f in unread[:1]:
                opts, gold = shuffle_options(["No", "Yes"], 0, rng)
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


def _norm_template(text):
    return " ".join(text.lower().split())


def _gold_is_yes(question, gold):
    for i, opt in enumerate(question["options"]):
        if opt.lower().strip() == "yes":
            return gold[i] == 1.0
    return False


def balance_train_presence(rows):
    """K5-v4 train-side presence balancing (opt-in, --balance-presence).

    v3 train data answers most presence questions Yes (fields are almost always
    printed), so a reader can score by base-rate priors. Group every emitted noul
    presence decision by (family, normalized template text); where a group holds
    both answers and Yes-count > 2 x No-count, deterministically drop the excess
    Yes decisions (sorted by qid, keep the first 2*N) so the prior stops paying.
    Mutates rows in place; returns the number of dropped decisions.
    """
    groups = defaultdict(list)          # (family, norm_text) -> [(page_i, q_i, qid, is_yes)]
    for pi, r in enumerate(rows):
        for qi, q in enumerate(r["questions"]):
            if q["primitive"] != "noul" or ":noul:pres:" not in q["qid"]:
                continue
            groups[(r["family"], _norm_template(q["text"]))].append(
                (pi, qi, q["qid"], _gold_is_yes(q, r["gold"][qi])))
    drop = set()
    for _, items in sorted(groups.items()):
        yes = sorted((it for it in items if it[3]), key=lambda t: t[2])
        n_no = sum(1 for it in items if not it[3])
        if not yes or n_no == 0:
            continue
        for pi, qi, _, _ in yes[2 * n_no:]:
            drop.add((pi, qi))
    per_page = defaultdict(list)
    for pi, qi in drop:
        per_page[pi].append(qi)
    for pi, qis in per_page.items():
        keep = [i for i in range(len(rows[pi]["questions"])) if i not in set(qis)]
        rows[pi]["questions"] = [rows[pi]["questions"][i] for i in keep]
        rows[pi]["gold"] = [rows[pi]["gold"][i] for i in keep]
    return len(drop)


def _pass1_item(fam, seed):
    """Pool worker: regenerate one page's gt + field pairs.

    gen_page is fully seed-parameterized and stateless, so a worker process
    reproduces the parent's result byte-identically; the parent replays
    results in submission order, so pools/inst_pool match the serial path.
    """
    try:
        gt = page_gt(fam, seed)
    except Exception as e:
        return fam, seed, None, None, None, f"ERR:{fam}:{type(e).__name__}"
    try:
        pairs = TABLES[fam](gt)
        inst_f = INSTITUTION_FIELD.get(fam)
        inst = {f2: a2 for f2, _, a2 in pairs}.get(inst_f)
        return fam, seed, gt, list(pairs), inst, None
    except Exception as e:
        return fam, seed, gt, None, None, f"ERR:pairs:{fam}:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pages-per-family", type=int, default=3000)
    ap.add_argument("--val-pages-per-family", type=int, default=400)
    ap.add_argument("--test-pages-per-family", type=int, default=600)
    ap.add_argument("--unseen-pages", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--balance-presence", action="store_true",
                    help="K5-v4: cap train Yes presence rows at 2x the No count per "
                         "(family, template) group so priors cannot answer presence")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--limit-pages", type=int, default=0, help="smoke cap")
    ap.add_argument("--workers", type=int, default=8,
                    help="pass-1 gt/pairs process pool (0/1 = serial, legacy path)")
    ap.add_argument("--out-prefix", type=str, default="",
                    help="prefix for gate2_labels_* output files (smoke/parity runs)")
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
    # K5: registry-omitted (twin) pages MUST be in the unseen split — they carry the
    # counterfactual No rows. Dict order puts them last, so the cap would cut them;
    # swap them in explicitly, dropping the tail base pages to stay at the cap.
    twins = [(f, s) for (f, s) in pages
             if f == UNSEEN_FAMILY and fin_docs.omitted_fields(f, s)]
    if twins:
        have = set(sel["unseen"])
        add = [t for t in twins if t not in have]
        keep = max(caps["unseen"] - len(add), len(add))
        sel["unseen"] = sel["unseen"][:keep] + add
        print(f"K5 twins forced into unseen: {len(add)} (kept {keep} base pages)",
              flush=True)
    if a.limit_pages:
        for b in sel:
            sel[b] = sel[b][: a.limit_pages]

    # pass 1: gt + field pairs -> pools for cross-page distractors
    factory_seed = random.Random(a.seed)
    pool_counts = defaultdict(int)
    pools = defaultdict(list)
    inst_pool = []
    gts = {}
    uniq = []
    seen_keys = set()
    for b in sel:
        for key in sel[b]:
            if key not in seen_keys:
                seen_keys.add(key)
                uniq.append(key)
    work = uniq if (getattr(a, "workers", 1) >= 2 and len(uniq) >= 200) else None
    if work is not None:
        from concurrent.futures import ProcessPoolExecutor
        fams, seeds = zip(*work)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            results = list(ex.map(_pass1_item, fams, seeds))
        print(f"pass1 parallel: {len(work)} pages on {a.workers} workers", flush=True)
    else:
        results = [_pass1_item(fam, seed) for fam, seed in uniq]
    # replay in submission order -> pools/inst_pool identical to the serial path
    for fam, seed, gt, pairs, inst, err in results:
        if err is not None:
            if err.startswith("ERR:pairs:"):
                pool_counts[err] += 1
                gts[(fam, seed)] = gt
            else:
                pool_counts[err] += 1
            continue
        gts[(fam, seed)] = gt
        for f, _, av in pairs or []:
            pools[(fam, f)].append(av)
            pool_counts[f"{fam}:{f}"] += 1
        inst_f = INSTITUTION_FIELD.get(fam)
        if inst and inst != "unreadable":
            inst_pool.append(inst)
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

    if a.balance_presence:
        dropped = balance_train_presence(out_rows["train"])
        print(f"balance-presence: dropped {dropped} excess Yes presence rows in train "
              f"(cap: Yes <= 2x No per family+template)", flush=True)

    stats = {}
    for b, rows in out_rows.items():
        path = ROOT / "artifacts" / f"gate2_labels_{a.out_prefix}{b}.jsonl"
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
    (ROOT / "artifacts" / f"gate2_label_stats_{a.out_prefix}.json").write_text(json.dumps(stats, indent=2))
    print("WROTE artifacts/gate2_label_stats_", a.out_prefix or "(none)", ".json sha=", stats["cfg_sha256"])
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
