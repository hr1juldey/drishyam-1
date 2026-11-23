# PROGRESS — drishyam-1 Gate 1

Human ledger. Machine state lives in `docs/EXEC_STATE.json`; append-only journal
in `docs/EXEC_LOG.md` (created by the engine at G1-E0).

## Planning run — 2026-11-23 (no execution, per user directive)

- [x] Repo created: https://github.com/hr1juldey/drishyam-1 (public, MIT,
      cloned at `~/Documents/code/drishyam-1`; renamed from decoupled-t5g2 the
      same day on the user's naming decision — drishyam = Skt. "vision", the
      Gate-2 destination; real-time constraint recorded in DECISIONS §18).
- [x] Environment verified: RTX 3060 12 GB (SM 8.6, driver 580.178.04, native
      bf16) · uv 0.13.7 · ast-grep 0.51.3 · gh 2.105.0 authed (hr1juldey) ·
      635 GB free disk.
- [x] HF assets verified live: dataset `LocalLLaMA/typed-decisions` (Apache-2.0,
      4 workflows × 300 train/100 test, 5 questions/case, official scorer
      shipped in-repo) · `google/t5gemma-2-270m-270m` (multimodal checkpoint —
      Gate 1 freeze/bypass confirmed necessary) · reference baseline
      `ktcapraz/t5gemma2-decision-270m-v2` (exists; custom engine +
      pre-fitted per-primitive temperatures).
- [x] Spec converted ROCm → CUDA (sanity gate kept, expectations adjusted).
- [x] Deviations/improvements + open questions recorded in `docs/DECISIONS.md`.
- [x] Execution roadmap (12 sub-tasks) + exec engine protocol committed.

## Execution checklist (updated by the engine, never by hand)

- [ ] G1-E0 — bootstrap exec engine + environment
- [ ] G1-S0 — ROCm→CUDA sanity gate (20 steps, bf16, finite)
- [ ] G1-D1 — dataset inspection + scorer metric extraction
- [ ] G1-D2 — zero-leakage splits (in-domain + unseen workflow)
- [ ] G1-D3 — stress suites (jitter n=300, multi-question n=100)
- [ ] G1-M1 — DecoupledT5GemmaDecision model + slot token + head + LoRA
- [ ] G1-T1 — training (smoke → full, MLflow, checkpoints)
- [ ] G1-C1 — per-primitive temperature calibration on val
- [ ] G1-V1 — evaluation suites on our model
- [ ] G1-B1 — trivial baselines (majority, TF-IDF + LogReg)
- [ ] G1-B2 — external reference baseline zero-shot
- [ ] G1-G1 — gate table render + GO/NO-GO verdict
