2026-11-23 22:11:57 DONE G1-E0 67e5edd2 engine 6/6 tests, V1-V5 green; probe PROBE PASS sm86 bf16; ast-grep rules fire on fixture; ruff clean; torch 2.15.0+cu132/transformers 5.24.0/peft 0.22.0
2026-11-23 22:14:48 FAIL G1-S0 attempt 1 attempt 1: gated repo 403 — google/t5gemma-2-270m-270m requires HF gate acceptance + authorized token; S0_EXIT=1 before any GPU step
2026-11-23 22:17:16 DONE G1-D1 750a360b counts 300/100 x4 exact; primitives noul/choice/score verified; scorer snapshot sha recorded
2026-11-23 22:17:46 DONE G1-D2 433fd122 240/60 per seen workflow; uniform composition key; disjoint asserts green; unseen train only sha-hashed
2026-11-23 22:17:46 DONE G1-D3 bfb3f3f3 jitter 300 + multiq 100, regeneration byte-identical sha=adf315347de9
2026-11-23 22:21:07 DONE G1-S0 d77a5033 PASS: torch.AdamW 20/20 finite 15.38->0.11; ManualAdamW 20/20 finite 15.09->0.15; artifacts/sanity_gate.json
2026-11-23 22:24:16 DONE G1-M1 86ee3504 smoke: trainable=7800705, slot shapes [2][3][4], vram 2001MB, backward ok; CPU wiring 2/2
2026-11-23 22:48:17 DONE G1-B1 1431fbce TF-IDF LogReg: test acc .633 Brier .161 AUC .826; unseen acc .260 AUC .491 = honest zero-shot floor
2026-11-23 23:13:35 DONE G1-B2 5ae69db8 B2 as-shipped: unseen AUC .910 (bar achievable); fails jitter<5% (.0567), invariance==0 (.0599), ECE<.08 (.183)
2026-11-23 23:57:42 DONE G1-T1 e9ae9b87 night queue: training EXIT marker seen
