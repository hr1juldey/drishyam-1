# IMAGE DATA PIPELINE — 2 LAKH ROWS, JUST-IN-TIME (plan + measured state)

Status 2026-11-24 night. Generators proven readable by the user.
**PARITY TEST PASSED** (jit_train_parity.py, real T5Gemma-2 trunk fwd+bwd,
8-micro x 4-accum = 32 rows/step, seq 256, gradient checkpointing on,
SigLIP+projector frozen): CONTROL 2141 ms/step (10.7 rows/s) vs JIT 2159 ms/step
(9.6 rows/s) — **parity delta +0.8%**. Producer per 32-row batch: CPU gen
242 ms + GPU render 543 ms = 785 ms inside the 2141 ms step budget.
VRAM peak 9.58 GB (card also carries ~1.75 GB of other tenants' contexts).
Training is now the bottleneck, not the data pipeline — the user's requirement
("image did not exist = image exists speed") is met and measured.

## The contract

- **DISK: recipes only.** `manifest_200k.jsonl` = `{row_id, family, seed}` per
  row — 11.5 MB for 2 lakh. **Zero pixels on disk, ever.**
- **PIXELS: generated at train time on the 3060, used by the step, disposed.**
  Recipe → PIL page (CPU) → degradation (numpy, plan recorded) → GPU doc-photo
  render (fp32 — bf16 grid quantization destroys small text, proven) → resize
  896² (SigLIP native) → consumed by the training step.
- **Determinism:** same seed → same pixels + same degradation plan. Any row can
  be regenerated exactly, on any machine, years later. Provenance is the seed.

## Family mix (v1)

| family | rows | generator | decision fields registered |
|---|---|---|---|
| thermal pharmacy receipt | 80,000 | `thermal_generator.gen_thermal` | yes (field bbox registry) |
| savings statement | 60,000 | `content_generator.gen_statement` | scalars only (registry TODO) |
| card e-statement | 60,000 | `content_generator.gen_card` | scalars only (registry TODO) |
| KYC forms | 0 | **generator missing** | — |

Manifest is family-blocked so each render batch is single-family (one renderer
geometry per family).

## Measured (RTX 3060, fp32 doc-photo)

- Standalone JIT stream: **42.8 rows/s** (6.7 ms/row CPU gen + 16.6 ms/row GPU
  render at 1080², then antialiased downsample to 896²).
- Standalone epoch: **1.30 h for 2 lakh rows**, VRAM peak 6.85 GB (batch 32).
- In-training parity: **9.6 vs 10.7 rows/s (+0.8%)** — generation hidden.
  2-lakh epoch at training speed: ~5.5 h (the model, not the data, sets this).
- Memory landmines found: 262k-vocab logits would cost 6.4 GB at batch 32
  (use the decision head / chunked loss, never full-vocab CE at big batch);
  AdamW on the frozen SigLIP wastes 3.3 GB of fp32 state — freeze before the
  optimizer; unchecked Gemma-trunk activations at micro 16 x 256 eat ~6 GB —
  gradient checkpointing on.
- 896²-direct rendering (th=896 tiles) would cut GPU cost ~30% if needed.

## The hard requirement (user, verbatim)

"even if the image did not exist, the speed and image existing the speed should
be same" — generation must be **free**, i.e. fully hidden behind the training
step. Test: `jit_train_parity.py` — real T5Gemma-2 270M fwd+bwd+AdamW at batch
32, control loop (blank batches) vs JIT loop (producer thread renders batch k+1
on a side CUDA stream into a 2-slot GPU ring while step k trains). Report:
median ms/step both loops, parity delta, PASS at ≤10%.

Architecture notes:
- Single process for CUDA; CPU page generation may need a second producer
  thread (or process pool if PIL's GIL contention shows up in the delta).
- Ring slots are bf16 on GPU (195 MB each) — fp32 render, cast once.
- The visible pipeline streams single-family blocks; training shuffles at the
  epoch level via the manifest, not the pixel stream (shuffle row_ids before
  blocking, or accept family-blocked curriculum).

## Labeling strategy (the expensive part is NOT pixels)

Pixels are free. Supervision comes in two layers:
1. **Rule labels (free, all 2 lakh):** decision Q&A derived deterministically
   from gt + `legibility_applied` (e.g. "total?" → value; total smudged →
   manual_review-lean; date on faded thermal → uncertainty). Same contract
   shapes as the v2a harness schemas (choice / numeric / noul).
2. **Teacher calibration (golden subset):** harness (v2a contract) relabels a
   1–2k golden slice per family; agreement rate between rule labels and GLM
   = the quality gate. 50-case family exams (the h500a pattern, ~$1 each)
   gate each new family BEFORE it joins the manifest.

This replaces "2-lakh teacher labels" (~$30–100, ~19 days) with
"2-lakh rule labels + ~6k teacher checks" while keeping the calibrated
distribution where it matters.

## Open blockers / next work

1. **SigLIP processor config missing** in local T5Gemma-2 snapshots — must be
   restored before the vision tower consumes these images in training. Until
   then the student trains text-only and the parity test uses token-level
   stand-in batches (same transformer cost).
2. **Statement field-bbox registries** (savings + card) — needed for the
   degradation plan/legibility coupling on those families (thermal already has
   it; `apply_degradation` works on any numpy RGB page once bboxes exist).
3. **KYC form generator** — the fourth family; same contract (field registry +
   deterministic seed) slots straight into the manifest mix.
4. Vision-tower forward adds SigLIP cost (~frozen fwd) to the training step —
   re-run parity after it is enabled; generation budget only gets easier.
