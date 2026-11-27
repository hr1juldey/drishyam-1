# PUBLICATION PREP — drishyam-1 (created 2026-11-27)

## The directive (verbatim, user, 2026-11-27)

> "Everything started from this prompt. Then, whatever you are narrating is true, but
> you are actually diluting everything that we did in the middle. We have to actually
> make it publication worthy and good communication — for that we have to know what we
> did at every step and report honestly."
>
> "If this work is properly done then we can prepare ourselves for publishing this
> work universally or in a proper place, because if we publish then we have something
> to show. Where we publish is another problem, but we have to prepare the data and
> everything for proper publication."

## Method (no-dilution discipline)

The paper is written FROM THE RECORD, never from agent memory:
1. `ORIGIN_PROMPT.md` — the verbatim kickoff spec (preserved 2026-11-27).
2. `sources/git_log_full.txt` — all 116 commits, oldest first, with dates.
3. `sources/DECISIONS.md`, `EXEC_LOG.md`, `PROGRESS.md`, `NIGHT_LOG.md`,
   `OVERNIGHT_PLAN.md`, `GATE25_CHECKPOINT_REPORT.md` — the decision trail.
4. `artifacts/benchmark_ledger.jsonl` + `docs/BENCHMARK_LEDGER.md` — every public
   number, dated, beside published references.
5. MLflow (sqlite runs/mlflow.db) — every training/eval run with params+metrics.
6. Session history — the user's verbatim steering words at each pivot (recovered
   via session_search, quoted in the chronology).

## The chronology contract

Every step of the arc gets: date, what was tried, what was measured, verdict
(including NO-GO/DEAD/falsified — reported as honestly as the PASSes), commit sha,
mlflow id. NOTHING is dropped for being unflattering: the NO-GO of Gate 1, the
v1≡v2 degenerate-exam discovery, the v3 DEAD verdict, the v4 public falsification
are chapters, not footnotes.

## Paper outline (draft skeleton — filled only after the chronology is complete)

1. **Introduction** — the question: can a 270M model make typed document decisions
   on one consumer GPU? The four-day, 116-commit answer.
2. **The origin spec** (ORIGIN_PROMPT.md content, condensed with permission-level
   fidelity): decoupled architecture, slot-token decoder, zero-leakage unseen-task
   split, stress suites, anti-shortcut recipe, mandatory ablations, GO/NO-GO bars.
3. **Gate 1** — architecture vindicated, criteria honestly split (what passed:
   exact multiq invariance 0.0000 + 3.81x cache speedup; jitter 0.029; what failed:
   ablation flip 0.194, ECE 0.185, unseen coin-flip) → NO-GO verdict honored.
4. **The middle (no dilution)** — Gate 1.5 teacher factory + label quality;
   generative-wall measurement (270M cannot transcribe digits); Gate 2 vision
   page pathway; Gate 2.5 row decision head; the v1≡v2 identical-below-chance
   signature → "the exam, not the student"; K5 twin fix; v3 counterfactual-minority
   → priors-not-reading → DEAD band honored.
5. **Going public** — benchmark ledger discipline; JevBench 39.8 (measured last
   place, recorded); ImajevBench closed with evidence; SROIE-verification v1
   protocol definition.
6. **The falsification and the fix** — v4 balance: internal fix, public collapse
   (always-No, true-accept 0.28%) → prediction falsified; v5-A domain adaptation:
   95.89% / AUC 0.988 / ECE 0.020 — the synthetic→scan gap WAS the blocker.
7. **Methodology contributions** — gates that can fail; one variable per run;
   pre-registered contracts with DEAD bands; dated public ledger; per-iteration
   checkpoint publishing.
8. **Limitations** — what is still not solved (unseen-workflow generalization,
   calibration on internal benchmark post-adaptation, scale limits).
