# BENCHMARK LEDGER — drishyam-1 dated competition record

Rule: every measured score gets a dated row, next to the published reference numbers we are measuring against.
Protocol is stated per row. Reference scores are lab/paper-published numbers (sources linked), not our re-runs.
Machine-readable companion: `artifacts/benchmark_ledger.jsonl` (one JSON object per row).

---

## TEXT — JevBench public-231 (open; 231 tasks; typed choice/noul/score; scored locally)

Reference ladder (published, benchmarkheaven.com/jev-models + repo):

| model | params | acc (public-231) | source |
|---|---|---|---|
| Jev 1.13.0 (closed API) | — | 86.6 | jfbench_research.md (board artifact) |
| system-one-open | 2B | 73.2 | same |
| Kev 0.6B | 0.6B | 66.7 | same |
| bev-decider | 0.4B | 65.8 | self-reported, rev 687e28f protocol |
| jeff | ~0.4B | 62.8 | board |
| Laya | 0.4B | 58.4 | board |
| deberta | ~0.4B | 52.4 | board |
| Kev 0.5B | 0.5B | 49.4 | board |

| date | ckpt | protocol | acc | vs ladder | git sha | mlflow |
|---|---|---|---|---|---|---|
| 2026-11-26 | runs/text-row-sft/best.pt (iteration 0: 1 ep, 58k/231k cases, 512-state train) | eval_jevbench @bebdf8b, state_max 2048 | **39.8** (easy 72.9 / std 36.1 / hard 27.9) | last place; −18.6 vs Laya 58.4 | bebdf8b | 951c4acb |

Planned climb: iteration 1 = full 231k + 2048-state + dict-form training (staged).

---

## IMAGE — ImajevBench v2.0-lite (open but PHOTO-domain: menus/signs/shelves/parcels)

| date | ckpt | protocol | result | reading | git sha |
|---|---|---|---|---|---|
| 2026-11-26 | runs/g2-row-sft/best.pt | stock dev split (173), abstention tau fitted on calibration; OOD by design | **34.1%** (noul 44.6 < coin, choice 26.5 ≈ random, score 0/7; Brier 0.989, ECE 0.407; tau sweep → 0) | total domain gap — benchmark not applicable to a document model; CLOSED as not-our-battlefield | 1feba17 |

Trained-section subset: does not exist (zero family overlap, verified 2026-11-26, deleg_56fcabed).
Official test split (279): labels withheld (submission scoring) — untouched.

---

## IMAGE — SROIE / CORD receipt verification (open DocAI canon; OUR DOMAIN — thermal receipts = home turf)

Reference numbers (published): SROIE KIE best Task-3 Hmean 90.49 (ICDAR 2019 organisers); Qwen2-VL-72B SROIE 75.50; GPT-4o CORD 81.26 (ocr-benchmarks survey). Protocol for our rows: verification of claimed field values vs public gold (TRUE = gold value quoted, FALSE = perturbed; 50/50 balanced), noul head, accuracy + AUC + ECE. Our protocol is self-defined on public gold (precedent: bev-decider's 5k protocol) and published here for comparability.

| date | ckpt | protocol | accuracy | AUC | ECE | git sha | mlflow |
|---|---|---|---|---|---|---|---|
| 2026-11-26 | runs/g2-row-sft/best.pt | SROIE verification v1 (this ledger defines it): **official 400-test pool**, 361/400 receipts with public Task-3 KIE gold (347 original + 14 recovered; HF extension jsdnrs/ICDAR2019-SROIE test split — the pinned Voxel51 snapshot is trainval-side, **0 test-ID overlap**, recorded as the contamination check); per receipt x {total, date, company} x {TRUE gold-verbatim, FALSE cross-receipt perturbation, seed 2026} = 2166 rows, 50/50 exact; raw softmax fp32, one tower encode per receipt | **51.2** (total 51.1 / date 50.4 / company 52.2; TRUE-Yes 82.4, FALSE-catch 20.1) | 0.516 | 0.252 | b85a525 | df8b3094 |

Verdict (2026-11-26): home-turf domain, coin-flip score. The stock champion READS real
scanned receipts (Yes on TRUE claims: 98.3% dates, 95.6% companies) but cannot REJECT wrong
values (FALSE-catch: 2.5% dates, 8.9% companies) — P(Yes) carries no claim signal (AUC 0.52)
and confidence is miscalibrated (ECE 0.25). Same failure mode as the v3 lesson: training-mix
answer priors, not verification. The parked balanced-pair training (v4) targets exactly this;
SROIE 50/50 is the clean probe for it. Totals are the weakest read (53.2% TRUE-recognition:
small totals in dense numeric columns at the 896px stretch). Full protocol + rows digest:
artifacts/sroie_table.json (rows sha256_16 373d16ee6fab6c2d, seed 2026).

---

## IMAGE — future fronts (entered, not yet measured)

| benchmark | status | reference anchors |
|---|---|---|
| RVL-CDIP (16-class doc classification) | staged — needs light train on public split | Donut-176M 95.3, DiT 92.69 |
| DocVQA (50k Qs, ANLS) | staged — needs small extraction decode head (param cost TBD) | human 94.36; Donut-base 65.9; Qwen2-VL-2B 90.1; frontier 95–97 (saturated) |
| ImageJevBench (sealed doc family) | blocked on Gemma-license check + safe loader + submission | board: 68 systems; smallest ranked 600M; Laya Vision 256M 75.9/ECE 0.035 (unranked, NC license) |

---

## INTERNAL GATES (our bench, not comparative — kept for the improvement ledger)

| date | ckpt | verdict | table |
|---|---|---|---|
| 2026-11-25 | g2-row-sft (v2) | 4/5 PASS (flip 0.346, ECE 0.036, jitter 0.0055; val acc 0.894) — CHAMPION | artifacts/gate2_row_table.json |
| 2026-11-26 | g2-row-sft-v3 (K5) | DEAD: flip 0.261 FAIL, g2_4 0.278 | artifacts/gate2_row_table_v3.json |
