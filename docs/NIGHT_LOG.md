- 23:46:42 === NIGHT GPU QUEUE start (pid 569585) ===
- 23:46:42 waiting for runs/g1-t1.log EXIT marker
- 23:46:42 [cpu] === NIGHT CPU LANE start (pid 569711) — waiting for training EXIT ===
G1-T1: done
- 23:57:42 STEP C1 start: uv run python -m drishyam.calibrate
- 23:57:42 [cpu] training EXIT seen — CPU lane starting (num_thread=8 capped so GPU lane keeps headroom)
- 23:57:42 [cpu] bake-off start (lfm3-omni vs gemma4:31b, 50 cases x 3 samples)
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 65, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 58, in main
    table = fit_all()
            ^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 51, in fit_all
    p = np.stack([np.asarray(x, dtype=float) for x in probs[sel]])
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/numpy/_core/shape_base.py", line 458, in stack
    raise ValueError('all input arrays must have the same shape')
ValueError: all input arrays must have the same shape
- 23:57:43 STEP C1 fail#1 rc=1 — retrying
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 65, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 58, in main
    table = fit_all()
            ^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/calibrate.py", line 51, in fit_all
    p = np.stack([np.asarray(x, dtype=float) for x in probs[sel]])
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/numpy/_core/shape_base.py", line 458, in stack
    raise ValueError('all input arrays must have the same shape')
ValueError: all input arrays must have the same shape
- 23:58:03 STEP C1 fail#2 rc=1 — retrying
- 23:58:23 STEP C1 PARKED (3-strikes)
- 23:58:23 STEP V1 start: uv run python -m drishyam.evaluate --ckpt runs/g1-t1/best.pt --out artifacts/gate_table.json
2026/11/23 23:58:27 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 10758.68it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
NIGHT_QUEUE_EXIT=143
- 23:59:58 === NIGHT GPU QUEUE start (pid 601283) ===
- 23:59:58 waiting for runs/g1-t1.log EXIT marker
G1-T1: done
- 23:59:58 STEP C1 start: uv run python -m drishyam.calibrate
{
  "noul": {
    "T": 0.94,
    "n": 300
  },
  "choice": {
    "T": 0.955,
    "n": 240
  },
  "score": {
    "T": 0.9375,
    "n": 360
  }
}
- 23:59:59 STEP C1 PASS (rc=0)
- 23:59:59 STEP V1 start: uv run python -m drishyam.evaluate --ckpt runs/g1-t1/best.pt --out artifacts/gate_table.json
2026/11/24 00:00:03 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 10797.17it/s]
Traceback (most recent call last):
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 179, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 171, in main
    rep = bakeoff(a.model.split("+"), a.cases, a.samples, a.seed)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 107, in bakeoff
    acc_p = {q: {k: float(np.mean([p.get(q, {}).get(k, 0.0) for p in sample_ps]))
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py", line 3862, in mean
    return _methods._mean(a, axis=axis, dtype=dtype,
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/numpy/_core/_methods.py", line 132, in _mean
    ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: unsupported operand type(s) for +: 'NoneType' and 'NoneType'
- 00:00:09 [cpu] bake-off PARKED (rc=1) — drafter still runs with gemma4:31b default
- 00:00:10 [cpu] night drafter start (gemma4:31b, budget 6h, max 400 cases, drafts/ quarantined)
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
- 00:03:44 [cpu] night drafter end — drafts at drafts/seen3_drafts.jsonl (0 cases)
- 00:03:44 [cpu] === NIGHT CPU LANE end ===
NIGHT_CPU_EXIT=0
Traceback (most recent call last):
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 184, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 176, in main
    rep = bakeoff(a.model.split("+"), a.cases, a.samples, a.seed)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/scripts/bakeoff.py", line 112, in bakeoff
    acc_p = {q: {k: float(np.mean([_num(p.get(q, {}).get(k)) for p in sample_ps]))
                                        ^^^^^^^^^^^^^^^^
AttributeError: 'float' object has no attribute 'get'
- 00:10:34 [cpu] bakeoff PARKED rc=0
drafted 0 cases (3 consecutive fails at stop)
- 00:14:35 [cpu] drafter end (0 cases)
- 00:14:35 [cpu] lane done
- 00:15:16 === NIGHT GPU QUEUE start (pid 638553) ===
G1-T1: done
- 00:15:16 STEP V1 start: uv run python -m drishyam.evaluate --ckpt runs/g1-t1/best.pt --out artifacts/gate_table.json
- 00:15:16 [cpu] === NIGHT CPU LANE start (pid 638595) ===
- 00:15:16 [cpu] bake-off start (lfm3-omni vs gemma4:31b, 50 cases x 3 samples)
2026/11/24 00:15:21 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/evaluate.py", line 205, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/evaluate.py", line 159, in main
    model, tok, fixed_len = load_model(ROOT / a.ckpt)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/evaluate.py", line 23, in load_model
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 1616, in load
    return _load(
           ^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2260, in _load
    result = unpickler.load()
             ^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2208, in persistent_load
    typed_storage = load_tensor(
                    ^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 2173, in load_tensor
    wrap_storage = restore_location(storage, location)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 1969, in restore_location
    return default_restore_location(storage, map_location)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 735, in default_restore_location
    result = fn(storage, location)
             ^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/serialization.py", line 669, in _deserialize
    return obj.to(device=device)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/storage.py", line 298, in to
    return _to(self, device, non_blocking)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/_utils.py", line 108, in _to
    untyped_storage = torch.UntypedStorage(self.size(), device=device)
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB. GPU 0 has a total capacity of 11.61 GiB of which 76.81 MiB is free. Process 2627 has 36.53 MiB memory in use. Process 2630 has 34.66 MiB memory in use. Process 3261 has 70.28 MiB memory in use. Process 372545 has 34.16 MiB memory in use. Process 486650 has 154.16 MiB memory in use. Process 3874839 has 232.00 MiB memory in use. Process 3506424 has 232.00 MiB memory in use. Process 601359 has 3.19 GiB memory in use. Process 626876 has 6.59 GiB memory in use. Including non-PyTorch memory, this process has 474.00 MiB memory in use. Of the allocated memory 368.58 MiB is allocated by PyTorch, and 1.42 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
- 00:15:22 STEP V1 fail#1 rc=1 — retrying
2026/11/24 00:15:47 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 9449.01it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
- 00:16:50 STEP V1 fail#1 rc=137 — retrying
2026/11/24 00:17:15 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 11338.01it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
NIGHT_QUEUE_EXIT=137
- 00:18:49 [cpu] === NIGHT CPU LANE start (pid 650476) ===
- 00:18:49 [cpu] bake-off start (lfm3-omni vs gemma4:31b, 50 cases x 3 samples)
lfm3-omni:latest {'ok_cases': 38, 'mean_soft_kl': 1.5437404077785248, 'argmax_agree_frac': 0.0, 'cases_per_hour': 320.2995669573436, 'wall_min': 7.118336192766825}
{
  "ckpt": "runs/g1-t1/best.pt",
  "temperatures": {
    "noul": {
      "T": 0.94,
      "n": 300
    },
    "choice": {
      "T": 0.955,
      "n": 240
    },
    "score": {
      "T": 0.9375,
      "n": 360
    }
  },
  "in_domain_test": {
    "acc": 0.792,
    "brier": 0.054116294843431535,
    "score_mae": 0.25833333333333336,
    "ece15": 0.14696889460086826,
    "ovr_auc": 0.9405537096409438,
    "n": 1500
  },
  "unseen_workflow": {
    "acc": 0.274,
    "brier": 0.24960353601485216,
    "score_mae": 1.165,
    "ece15": 0.16219686758518223,
    "ovr_auc": 0.4987853697006382,
    "n": 500
  },
  "multiq": {
    "n_states": 100,
    "max_delta_full_prec": 0.9946786165237427,
    "max_delta_4dp": 0.9947,
    "bitwise_equal_frac": 0.0,
    "median_speedup": 4.424046744747157
  },
  "state_ablation": {
    "state_ablation_flip_rate": 0.192,
    "real_acc": 0.274,
    "blank_acc": 0.268,
    "acc_drop": 0.006000000000000005,
    "real_brier": 0.24960353601485216,
    "blank_brier": 0.2245711954469344
  },
  "jitter_flip_rate": 0.029333333333333333
}
- 00:48:07 STEP V1 PASS (rc=0)
- 00:48:07 STEP VERDICT start: uv run python scripts/gate_verdict.py --gate artifacts/gate_table.json --out artifacts/verdict.json
Traceback (most recent call last):
  File "/home/riju279/Documents/code/drishyam-1/scripts/gate_verdict.py", line 77, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/scripts/gate_verdict.py", line 15, in main
    gt = json.loads((ROOT / a.gate).read_text())
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/pathlib.py", line 1027, in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/riju279/Documents/code/artifacts/gate_table.json'
- 00:48:07 STEP VERDICT fail#1 rc=1 — retrying
Traceback (most recent call last):
  File "/home/riju279/Documents/code/drishyam-1/scripts/gate_verdict.py", line 77, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/scripts/gate_verdict.py", line 15, in main
    gt = json.loads((ROOT / a.gate).read_text())
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/pathlib.py", line 1027, in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/.local/share/uv/python/cpython-3.12.16-linux-x86_64-gnu/lib/python3.12/pathlib.py", line 1013, in open
    return io.open(self, mode, buffering, encoding, errors, newline)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: '/home/riju279/Documents/code/artifacts/gate_table.json'
- 00:48:27 STEP VERDICT fail#2 rc=1 — retrying
- 00:48:47 STEP VERDICT PARKED (3-strikes)
- 00:48:47 verdict missing — lr screens withheld, queue end (watchdog may retry)
- 00:49:50 === NIGHT GPU QUEUE start (pid 712999) ===
- 00:49:50 verdict chain already done — fast-forward to lr screens
- 00:49:50 lr screen 3e-4 training (budget ok)
- 00:49:50 STEP T1LR3 start: uv run python -m drishyam.train --lr 3e-4 --epochs 40 --seed 42 --out runs/g1-t1-lr3e4
2026/11/24 00:50:00 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
fixed_len=116 train_cases=720 val_cases=180
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights:  66%|██████▋   | 605/911 [00:00<00:00, 6048.53it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 5801.09it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
vision offloaded to CPU: vision_tower
/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py:3204: FutureWarning: `gradient_checkpointing_enable` is deprecated and will be removed in v6. Use `activation_checkpointing_enable` instead.
  warnings.warn(
gradient checkpointing: on
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[W1124 00:50:25.563426913 CUDACachingAllocator.cpp:623] expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 88014848, total: 12470255616).
[W1124 00:50:25.662057362 CUDACachingAllocator.cpp:623] expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 90112000, total: 12470255616).
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 270, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 215, in main
    tr_loss, _ = run_epoch(model, tok, train_cases, device, fixed_len,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 92, in run_epoch
    out = model(s_ids, s_mask, dec_ids, dec_mask, slot_pos, row_case=row_case)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/model.py", line 102, in forward
    out = self.bb(
          ^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/peft/peft_model.py", line 1053, in forward
    return self.get_base_model()(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 1036, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 173, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py", line 1254, in forward
    logits = self.lm_head(hidden_states)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py", line 556, in forward
    logits = self.out_proj(hidden_states)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.27 GiB. GPU 0 has a total capacity of 11.61 GiB of which 1.86 GiB is free. Process 2627 has 36.53 MiB memory in use. Process 2630 has 34.66 MiB memory in use. Process 3261 has 70.28 MiB memory in use. Process 372545 has 34.16 MiB memory in use. Process 486650 has 154.16 MiB memory in use. Process 3874839 has 232.00 MiB memory in use. Process 3506424 has 232.00 MiB memory in use. Process 667166 has 6.60 GiB memory in use. Including non-PyTorch memory, this process has 1.87 GiB memory in use. Of the allocated memory 1.70 GiB is allocated by PyTorch, and 37.56 MiB is reserved by PyTorch but unallocated. expandable_segments is enabled. If instead the process is running out of virtual address space (VSZ), expandable segments may be over-reserving VA (each segment reserves ~9/8 of device memory, times the number of streams); reduce it via PYTORCH_CUDA_ALLOC_CONF expandable_segments_reserve / expandable_segments_reserve_by_class.
- 00:50:28 STEP T1LR3 fail#1 rc=1 — retrying
2026/11/24 00:50:56 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
fixed_len=116 train_cases=720 val_cases=180
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights:  66%|██████▌   | 597/911 [00:00<00:00, 5962.20it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 5866.95it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
vision offloaded to CPU: vision_tower
/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py:3204: FutureWarning: `gradient_checkpointing_enable` is deprecated and will be removed in v6. Use `activation_checkpointing_enable` instead.
  warnings.warn(
gradient checkpointing: on
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[W1124 00:51:22.259863745 CUDACachingAllocator.cpp:623] expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 88014848, total: 12470255616).
[W1124 00:51:22.361174895 CUDACachingAllocator.cpp:623] expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 90112000, total: 12470255616).
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 270, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 215, in main
    tr_loss, _ = run_epoch(model, tok, train_cases, device, fixed_len,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 92, in run_epoch
    out = model(s_ids, s_mask, dec_ids, dec_mask, slot_pos, row_case=row_case)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/model.py", line 102, in forward
    out = self.bb(
          ^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/peft/peft_model.py", line 1053, in forward
    return self.get_base_model()(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 1036, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 173, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py", line 1254, in forward
    logits = self.lm_head(hidden_states)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py", line 556, in forward
    logits = self.out_proj(hidden_states)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.27 GiB. GPU 0 has a total capacity of 11.61 GiB of which 1.86 GiB is free. Process 2627 has 36.53 MiB memory in use. Process 2630 has 34.66 MiB memory in use. Process 3261 has 70.28 MiB memory in use. Process 372545 has 34.16 MiB memory in use. Process 486650 has 154.16 MiB memory in use. Process 3874839 has 232.00 MiB memory in use. Process 3506424 has 232.00 MiB memory in use. Process 667166 has 6.60 GiB memory in use. Including non-PyTorch memory, this process has 1.87 GiB memory in use. Of the allocated memory 1.70 GiB is allocated by PyTorch, and 37.56 MiB is reserved by PyTorch but unallocated. expandable_segments is enabled. If instead the process is running out of virtual address space (VSZ), expandable segments may be over-reserving VA (each segment reserves ~9/8 of device memory, times the number of streams); reduce it via PYTORCH_CUDA_ALLOC_CONF expandable_segments_reserve / expandable_segments_reserve_by_class.
- 00:51:27 STEP T1LR3 fail#2 rc=1 — retrying
- 00:51:47 STEP T1LR3 PARKED (3-strikes)
- 00:51:47 lr screen 5e-5 training
- 00:51:47 STEP T1LR5 start: uv run python -m drishyam.train --lr 5e-5 --epochs 40 --seed 42 --out runs/g1-t1-lr5e5
2026/11/24 00:51:55 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
fixed_len=116 train_cases=720 val_cases=180
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights:  64%|██████▍   | 585/911 [00:00<00:00, 5844.39it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 5536.22it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
vision offloaded to CPU: vision_tower
/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py:3204: FutureWarning: `gradient_checkpointing_enable` is deprecated and will be removed in v6. Use `activation_checkpointing_enable` instead.
  warnings.warn(
gradient checkpointing: on
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[W1124 00:52:20.468055781 CUDACachingAllocator.cpp:623] expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 86638592, total: 12470255616).
[W1124 00:52:20.570437098 CUDACachingAllocator.cpp:623] expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 90112000, total: 12470255616).
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 270, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 215, in main
    tr_loss, _ = run_epoch(model, tok, train_cases, device, fixed_len,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 92, in run_epoch
    out = model(s_ids, s_mask, dec_ids, dec_mask, slot_pos, row_case=row_case)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/model.py", line 102, in forward
    out = self.bb(
          ^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/peft/peft_model.py", line 1053, in forward
    return self.get_base_model()(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 1036, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 173, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py", line 1254, in forward
    logits = self.lm_head(hidden_states)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py", line 556, in forward
    logits = self.out_proj(hidden_states)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.27 GiB. GPU 0 has a total capacity of 11.61 GiB of which 1.86 GiB is free. Process 2627 has 36.53 MiB memory in use. Process 2630 has 34.66 MiB memory in use. Process 3261 has 70.28 MiB memory in use. Process 372545 has 34.16 MiB memory in use. Process 486650 has 154.16 MiB memory in use. Process 3874839 has 232.00 MiB memory in use. Process 3506424 has 232.00 MiB memory in use. Process 667166 has 6.60 GiB memory in use. Including non-PyTorch memory, this process has 1.87 GiB memory in use. Of the allocated memory 1.70 GiB is allocated by PyTorch, and 37.56 MiB is reserved by PyTorch but unallocated. expandable_segments is enabled. If instead the process is running out of virtual address space (VSZ), expandable segments may be over-reserving VA (each segment reserves ~9/8 of device memory, times the number of streams); reduce it via PYTORCH_CUDA_ALLOC_CONF expandable_segments_reserve / expandable_segments_reserve_by_class.
- 00:52:23 STEP T1LR5 fail#1 rc=1 — retrying
2026/11/24 00:52:51 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
fixed_len=116 train_cases=720 val_cases=180
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 17240.18it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
vision offloaded to CPU: vision_tower
/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py:3204: FutureWarning: `gradient_checkpointing_enable` is deprecated and will be removed in v6. Use `activation_checkpointing_enable` instead.
  warnings.warn(
gradient checkpointing: on
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
[W1124 00:53:15.131824147 CUDACachingAllocator.cpp:623] expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 88014848, total: 12470255616).
[W1124 00:53:15.221753357 CUDACachingAllocator.cpp:623] expandable_segments: memory mapping failed with OOM on device 0 while trying to map 20971520 bytes (free: 90112000, total: 12470255616).
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 270, in <module>
    main()
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 215, in main
    tr_loss, _ = run_epoch(model, tok, train_cases, device, fixed_len,
                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/train.py", line 92, in run_epoch
    out = model(s_ids, s_mask, dec_ids, dec_mask, slot_pos, row_case=row_case)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/src/drishyam/model.py", line 102, in forward
    out = self.bb(
          ^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/peft/peft_model.py", line 1053, in forward
    return self.get_base_model()(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/utils/generic.py", line 1036, in wrapper
    output = func(self, *args, **kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/utils/deprecation.py", line 173, in wrapped_func
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py", line 1254, in forward
    logits = self.lm_head(hidden_states)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/models/t5gemma2/modeling_t5gemma2.py", line 556, in forward
    logits = self.out_proj(hidden_states)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.27 GiB. GPU 0 has a total capacity of 11.61 GiB of which 1.86 GiB is free. Process 2627 has 36.53 MiB memory in use. Process 2630 has 34.66 MiB memory in use. Process 3261 has 70.28 MiB memory in use. Process 372545 has 34.16 MiB memory in use. Process 486650 has 154.16 MiB memory in use. Process 3874839 has 232.00 MiB memory in use. Process 3506424 has 232.00 MiB memory in use. Process 667166 has 6.60 GiB memory in use. Including non-PyTorch memory, this process has 1.87 GiB memory in use. Of the allocated memory 1.70 GiB is allocated by PyTorch, and 37.56 MiB is reserved by PyTorch but unallocated. expandable_segments is enabled. If instead the process is running out of virtual address space (VSZ), expandable segments may be over-reserving VA (each segment reserves ~9/8 of device memory, times the number of streams); reduce it via PYTORCH_CUDA_ALLOC_CONF expandable_segments_reserve / expandable_segments_reserve_by_class.
- 00:53:19 STEP T1LR5 fail#2 rc=1 — retrying
- 00:53:39 STEP T1LR5 PARKED (3-strikes)
- 00:53:39 STEP PUSH2 start: git push -q origin main
- 00:53:41 STEP PUSH2 PASS (rc=0)
- 00:53:41 === NIGHT GPU QUEUE end ===
- 00:55:51 === NIGHT GPU QUEUE start (pid 728151) ===
- 00:55:51 verdict chain already done — fast-forward to lr screens
- 00:55:51 lr screen 3e-4 training (budget ok)
- 00:55:51 STEP T1LR3 start: uv run python -m drishyam.train --lr 3e-4 --epochs 40 --seed 42 --out runs/g1-t1-lr3e4
- 00:55:52 [cpu] === NIGHT CPU LANE start (pid 728183) ===
- 00:55:52 [cpu] bake-off start (lfm3-omni vs gemma4:31b, 50 cases x 3 samples)
2026/11/24 00:55:55 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
fixed_len=116 train_cases=720 val_cases=180
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights:  83%|████████▎ | 760/911 [00:00<00:00, 7584.46it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 7440.59it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
vision offloaded to CPU: vision_tower
/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py:3204: FutureWarning: `gradient_checkpointing_enable` is deprecated and will be removed in v6. Use `activation_checkpointing_enable` instead.
  warnings.warn(
gradient checkpointing: on
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
ep00 train=1.2203 val=1.2021 acc=0.4478 brier=0.2385 ece=0.0953 vram=4421MB
ep01 train=1.1027 val=1.0802 acc=0.5544 brier=0.1889 ece=0.0692 vram=4421MB
ep02 train=1.0595 val=1.0662 acc=0.6244 brier=0.1811 ece=0.0614 vram=4425MB
ep03 train=0.9940 val=0.9815 acc=0.6811 brier=0.1394 ece=0.0657 vram=4425MB
ep04 train=0.9404 val=0.9456 acc=0.7089 brier=0.1188 ece=0.0784 vram=4425MB
ep05 train=0.9394 val=1.0310 acc=0.7056 brier=0.1524 ece=0.0642 vram=4425MB
ep06 train=0.9341 val=0.9780 acc=0.6744 brier=0.1318 ece=0.0391 vram=4425MB
ep07 train=0.9297 val=1.0455 acc=0.6244 brier=0.1727 ece=0.0356 vram=4425MB
ep08 train=0.8924 val=1.1199 acc=0.5889 brier=0.1973 ece=0.1052 vram=4425MB
lfm3-omni:latest {'ok_cases': 34, 'mean_soft_kl': 1.3119993554216303, 'argmax_agree_frac': 0.0, 'cases_per_hour': 64.27822034131992, 'wall_min': 31.737032997608186}
ep09 train=0.9180 val=1.1535 acc=0.5422 brier=0.2196 ece=0.0939 vram=4426MB
EARLY STOP at epoch 9 (patience 5)
T1 DONE best_val=0.9456 wall=34.1min vram=4426MB
- 01:30:34 STEP T1LR3 PASS (rc=0)
- 01:30:35 STEP V1LR3 start: uv run python -m drishyam.evaluate --ckpt runs/g1-t1-lr3e4/best.pt --out artifacts/gate_table_lr3e4.json
2026/11/24 01:30:47 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights:  46%|████▋     | 422/911 [00:00<00:00, 4168.09it/s]Loading weights:  92%|█████████▏| 839/911 [00:00<00:00, 3758.21it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 3830.34it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
{
  "ckpt": "runs/g1-t1-lr3e4/best.pt",
  "temperatures": {
    "noul": {
      "T": 1.09,
      "n": 300
    },
    "choice": {
      "T": 1.1075,
      "n": 240
    },
    "score": {
      "T": 1.1949999999999998,
      "n": 360
    }
  },
  "in_domain_test": {
    "acc": 0.704,
    "brier": 0.10936115228194725,
    "score_mae": 0.42333333333333334,
    "ece15": 0.10036138043800992,
    "ovr_auc": 0.8716237061178475,
    "n": 1500
  },
  "unseen_workflow": {
    "acc": 0.368,
    "brier": 0.2390940749132487,
    "score_mae": 1.06,
    "ece15": 0.11706909847259524,
    "ovr_auc": 0.521718991004035,
    "n": 500
  },
  "multiq": {
    "n_states": 100,
    "max_delta_full_prec": 0.9634658694267273,
    "max_delta_4dp": 0.9635,
    "bitwise_equal_frac": 0.0,
    "median_speedup": 4.029975153128694
  },
  "state_ablation": {
    "state_ablation_flip_rate": 0.258,
    "real_acc": 0.368,
    "blank_acc": 0.39,
    "acc_drop": -0.02200000000000002,
    "real_brier": 0.2390940749132487,
    "blank_brier": 0.20309302123361123
  },
  "jitter_flip_rate": 0.021333333333333333
}
- 02:03:34 STEP V1LR3 PASS (rc=0)
- 02:03:34 lr screen 5e-5 training
- 02:03:34 STEP T1LR5 start: uv run python -m drishyam.train --lr 5e-5 --epochs 40 --seed 42 --out runs/g1-t1-lr5e5
2026/11/24 02:03:46 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
fixed_len=116 train_cases=720 val_cases=180
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights:  37%|███▋      | 337/911 [00:00<00:00, 3355.06it/s]Loading weights:  74%|███████▍  | 673/911 [00:00<00:00, 2838.77it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 3140.31it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
vision offloaded to CPU: vision_tower
/home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/transformers/modeling_utils.py:3204: FutureWarning: `gradient_checkpointing_enable` is deprecated and will be removed in v6. Use `activation_checkpointing_enable` instead.
  warnings.warn(
gradient checkpointing: on
[transformers] `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.
ep00 train=1.2837 val=1.3001 acc=0.4044 brier=0.2796 ece=0.0661 vram=4421MB
ep01 train=1.2139 val=1.1686 acc=0.4833 brier=0.2230 ece=0.0476 vram=4421MB
ep02 train=1.1196 val=1.0707 acc=0.5911 brier=0.1778 ece=0.0957 vram=4425MB
ep03 train=1.0627 val=1.0694 acc=0.5700 brier=0.1839 ece=0.1353 vram=4425MB
ep04 train=1.0172 val=1.0405 acc=0.5678 brier=0.1699 ece=0.0721 vram=4425MB
ep05 train=0.9547 val=0.9961 acc=0.6278 brier=0.1435 ece=0.0532 vram=4425MB
ep06 train=0.9277 val=0.9938 acc=0.6633 brier=0.1415 ece=0.0511 vram=4425MB
ep07 train=0.8977 val=0.9641 acc=0.6700 brier=0.1255 ece=0.0647 vram=4425MB
ep08 train=0.8728 val=0.9753 acc=0.6778 brier=0.1269 ece=0.0608 vram=4425MB
ep09 train=0.8512 val=1.1139 acc=0.5644 brier=0.2017 ece=0.0633 vram=4426MB
ep10 train=0.8491 val=0.8823 acc=0.7356 brier=0.0866 ece=0.1010 vram=4431MB
ep11 train=0.8051 val=0.9164 acc=0.6789 brier=0.1050 ece=0.0894 vram=4431MB
ep12 train=0.7942 val=0.8665 acc=0.7400 brier=0.0802 ece=0.1167 vram=4431MB
ep13 train=0.7848 val=0.9111 acc=0.7167 brier=0.0983 ece=0.0644 vram=4431MB
ep14 train=0.7832 val=0.9048 acc=0.7511 brier=0.0951 ece=0.0542 vram=4431MB
ep15 train=0.7627 val=0.9177 acc=0.7400 brier=0.0994 ece=0.0785 vram=4431MB
ep16 train=0.7564 val=0.9226 acc=0.7544 brier=0.1082 ece=0.1085 vram=4431MB
ep17 train=0.7590 val=0.8443 acc=0.7767 brier=0.0689 ece=0.1037 vram=4431MB
ep18 train=0.7470 val=0.8780 acc=0.7211 brier=0.0867 ece=0.0778 vram=4431MB
ep19 train=0.7488 val=0.8798 acc=0.7200 brier=0.0846 ece=0.0920 vram=4431MB
ep20 train=0.7439 val=0.8904 acc=0.7367 brier=0.0861 ece=0.1083 vram=4431MB
ep21 train=0.7409 val=0.8804 acc=0.7778 brier=0.0795 ece=0.1232 vram=4431MB
ep22 train=0.7426 val=0.8452 acc=0.7878 brier=0.0666 ece=0.1277 vram=4431MB
EARLY STOP at epoch 22 (patience 5)
T1 DONE best_val=0.8443 wall=69.7min vram=4431MB
- 03:14:13 STEP T1LR5 PASS (rc=0)
- 03:14:15 STEP V1LR5 start: uv run python -m drishyam.evaluate --ckpt runs/g1-t1-lr5e5/best.pt --out artifacts/gate_table_lr5e5.json
2026/11/24 03:14:25 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights:  61%|██████    | 553/911 [00:00<00:00, 5526.76it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 5541.22it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
- 03:25:52 [cpu] bake-off PARKED (rc=124) — drafter still runs with gemma4:31b default
- 03:25:52 [cpu] night drafter start (gemma4:31b, 11048s budget, max 400 total, drafts/ quarantined)
drafted 0 cases (3 consecutive fails at stop)
- 03:31:23 [cpu] night drafter end — drafts at drafts/seen3_drafts.jsonl (0 cases)
- 03:31:23 [cpu] === NIGHT CPU LANE end ===
lfm3-omni:latest {'ok_cases': 25, 'mean_soft_kl': 1.4656140916768725, 'argmax_agree_frac': 0.0, 'cases_per_hour': 194.3355534009356, 'wall_min': 7.718608220418294}
{
  "ckpt": "runs/g1-t1-lr5e5/best.pt",
  "temperatures": {
    "noul": {
      "T": 1.1999999999999997,
      "n": 300
    },
    "choice": {
      "T": 1.1675,
      "n": 240
    },
    "score": {
      "T": 1.1075,
      "n": 360
    }
  },
  "in_domain_test": {
    "acc": 0.7773333333333333,
    "brier": 0.057212633867256404,
    "score_mae": 0.295,
    "ece15": 0.13731287016471225,
    "ovr_auc": 0.9344813924447237,
    "n": 1500
  },
  "unseen_workflow": {
    "acc": 0.324,
    "brier": 0.2474391349647442,
    "score_mae": 1.16,
    "ece15": 0.15013424021005628,
    "ovr_auc": 0.491679521493839,
    "n": 500
  },
  "multiq": {
    "n_states": 100,
    "max_delta_full_prec": 0.9976332783699036,
    "max_delta_4dp": 0.9976,
    "bitwise_equal_frac": 0.0,
    "median_speedup": 3.3202904154813453
  },
  "state_ablation": {
    "state_ablation_flip_rate": 0.282,
    "real_acc": 0.324,
    "blank_acc": 0.39,
    "acc_drop": -0.066,
    "real_brier": 0.2474391349647442,
    "blank_brier": 0.19112162490360673
  },
  "jitter_flip_rate": 0.036
}
- 03:44:35 STEP V1LR5 PASS (rc=0)
- 03:44:35 STEP PUSH2 start: git push -q origin main
- 03:44:37 STEP PUSH2 PASS (rc=0)
- 03:44:37 === NIGHT GPU QUEUE end ===
2026/11/24 03:45:24 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights:  34%|███▎      | 307/911 [00:00<00:00, 3022.77it/s]Loading weights:  72%|███████▏  | 656/911 [00:00<00:00, 3286.73it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 3335.79it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
{
  "ckpt": "runs/g1-t1/best.pt",
  "temperatures": {
    "noul": {
      "T": 1.1999999999999997,
      "n": 300
    },
    "choice": {
      "T": 1.1675,
      "n": 240
    },
    "score": {
      "T": 1.1075,
      "n": 360
    }
  },
  "in_domain_test": {
    "acc": 0.7913333333333333,
    "brier": 0.05447717744779611,
    "score_mae": 0.25833333333333336,
    "ece15": 0.18513940117756528,
    "ovr_auc": 0.9399846416407058,
    "n": 1500
  },
  "unseen_workflow": {
    "acc": 0.276,
    "brier": 0.23573027879807676,
    "score_mae": 1.16,
    "ece15": 0.13670498710870743,
    "ovr_auc": 0.49734617594060015,
    "n": 500
  },
  "multiq": {
    "n_states": 100,
    "max_delta_full_prec": 0.9946938753128052,
    "max_delta_4dp": 0.9947,
    "bitwise_equal_frac": 0.0,
    "median_speedup": 3.888664622338606
  },
  "state_ablation": {
    "state_ablation_flip_rate": 0.194,
    "real_acc": 0.276,
    "blank_acc": 0.284,
    "acc_drop": -0.007999999999999952,
    "real_brier": 0.23573027879807676,
    "blank_brier": 0.213430679327753
  },
  "jitter_flip_rate": 0.028666666666666663
}
- 04:18:07 [gpu-manual] fp32 re-eval PASS
{
  "ckpt": "runs/g1-t1/best.pt",
  "temperatures": {
    "noul": {
      "T": 1.1999999999999997,
      "n": 300
    },
    "choice": {
      "T": 1.1675,
      "n": 240
    },
    "score": {
      "T": 1.1075,
      "n": 360
    }
  },
  "criteria": {
    "1_state_ablation": {
      "flip_rate": 0.194,
      "bar": "> 0.30",
      "acc_drop": -0.007999999999999952,
      "acc_bar": "> 0.15",
      "pass": false
    },
    "2_invariance": {
      "max_delta_4dp": 0.9947,
      "bar": "== 0.0",
      "speedup": 3.888664622338606,
      "speedup_bar": "> 1.5",
      "pass": false
    },
    "3_robustness_calibration": {
      "jitter_flip": 0.028666666666666663,
      "bar": "< 0.05",
      "post_ts_ece_indomain": 0.18513940117756528,
      "ece_bar": "< 0.08",
      "pass": false
    },
    "4_unseen_generalization": {
      "unseen_auc": 0.49734617594060015,
      "bar": "> 0.75",
      "unseen_acc": 0.276,
      "majority_acc": 0.26,
      "pass": false
    },
    "5_unseen_ece": {
      "unseen_post_ts_ece": 0.13670498710870743,
      "bar": "< 0.12",
      "pass": false
    }
  },
  "gate": "NO-GO",
  "failed": [
    "1",
    "2",
    "3",
    "4",
    "5"
  ],
  "ours_vs_b2": {
    "in_domain": {
      "acc": 0.7913333333333333,
      "brier": 0.05447717744779611,
      "ece15": 0.18513940117756528,
      "ovr_auc": 0.9399846416407058,
      "score_mae": 0.25833333333333336
    },
    "unseen": {
      "acc": 0.276,
      "brier": 0.23573027879807676,
      "ece15": 0.13670498710870743,
      "ovr_auc": 0.49734617594060015,
      "score_mae": 1.16
    },
    "b2_reference": {
      "acc": 0.7586666666666667,
      "auc": 0.9187387525390367,
      "unseen_acc": 0.71,
      "unseen_auc": 0.910485817772079,
      "jitter_flip": 0.056666666666666664,
      "multiq_delta_4dp": 0.0599,
      "ece_indomain": 0.18265156266666666
    }
  },
  "notes": [
    "reliability-curve monotonicity: manual check pending bins artifact",
    "in-domain/unseen numbers are post per-primitive TS (see gate_table)"
  ]
}
- 04:18:10 [gpu-manual] verdict + evidence + push done
2026/11/24 04:34:17 INFO mlflow.agent.hint: Load the `mlflow-trace-instrumentation` skill at /home/riju279/Documents/code/drishyam-1/.venv/lib/python3.12/site-packages/mlflow/assistant/skills/mlflow-trace-instrumentation/SKILL.md before writing any tracing code; it ships with this MLflow install. Set MLFLOW_DISABLE_AGENT_HINT=1 to silence this.
Loading weights:   0%|          | 0/911 [00:00<?, ?it/s]Loading weights:  68%|██████▊   | 618/911 [00:00<00:00, 6173.72it/s]Loading weights: 100%|██████████| 911/911 [00:00<00:00, 5998.30it/s]
[transformers] The new embeddings will be initialized from a multivariate normal distribution that has old embeddings' mean and covariance. As described in this article: https://nlp.stanford.edu/~johnhew/vocab-expansion.html. To disable this, use `mean_resizing=False`
- 05:03:23 [cpu] bakeoff PARKED rc=0
{
  "ckpt": "runs/g1-t1/best.pt",
  "temperatures": {
    "noul": {
      "T": 1.1999999999999997,
      "n": 300
    },
    "choice": {
      "T": 1.1675,
      "n": 240
    },
    "score": {
      "T": 1.1075,
      "n": 360
    }
  },
  "in_domain_test": {
    "acc": 0.7913333333333333,
    "brier": 0.05447717744779611,
    "score_mae": 0.25833333333333336,
    "ece15": 0.18513940117756528,
    "ovr_auc": 0.9399846416407058,
    "n": 1500
  },
  "unseen_workflow": {
    "acc": 0.276,
    "brier": 0.23573027879807676,
    "score_mae": 1.16,
    "ece15": 0.13670498710870743,
    "ovr_auc": 0.49734617594060015,
    "n": 500
  },
  "multiq": {
    "n_states": 100,
    "max_delta_full_prec": 4.172325134277344e-07,
    "max_delta_4dp": 0.0,
    "bitwise_equal_frac": 0.01,
    "median_speedup": 3.805504562200447
  },
  "state_ablation": {
    "state_ablation_flip_rate": 0.194,
    "real_acc": 0.276,
    "blank_acc": 0.284,
    "acc_drop": -0.007999999999999952,
    "real_brier": 0.23573027879807676,
    "blank_brier": 0.213430679327753
  },
  "jitter_flip_rate": 0.028666666666666663
}
- 05:06:06 [gpu-manual] fixed-suite re-eval PASS
{
  "ckpt": "runs/g1-t1/best.pt",
  "temperatures": {
    "noul": {
      "T": 1.1999999999999997,
      "n": 300
    },
    "choice": {
      "T": 1.1675,
      "n": 240
    },
    "score": {
      "T": 1.1075,
      "n": 360
    }
  },
  "criteria": {
    "1_state_ablation": {
      "flip_rate": 0.194,
      "bar": "> 0.30",
      "acc_drop": -0.007999999999999952,
      "acc_bar": "> 0.15",
      "pass": false
    },
    "2_invariance": {
      "max_delta_4dp": 0.0,
      "bar": "== 0.0",
      "speedup": 3.805504562200447,
      "speedup_bar": "> 1.5",
      "pass": true
    },
    "3_robustness_calibration": {
      "jitter_flip": 0.028666666666666663,
      "bar": "< 0.05",
      "post_ts_ece_indomain": 0.18513940117756528,
      "ece_bar": "< 0.08",
      "pass": false
    },
    "4_unseen_generalization": {
      "unseen_auc": 0.49734617594060015,
      "bar": "> 0.75",
      "unseen_acc": 0.276,
      "majority_acc": 0.26,
      "pass": false
    },
    "5_unseen_ece": {
      "unseen_post_ts_ece": 0.13670498710870743,
      "bar": "< 0.12",
      "pass": false
    }
  },
  "gate": "NO-GO",
  "failed": [
    "1",
    "3",
    "4",
    "5"
  ],
  "ours_vs_b2": {
    "in_domain": {
      "acc": 0.7913333333333333,
      "brier": 0.05447717744779611,
      "ece15": 0.18513940117756528,
      "ovr_auc": 0.9399846416407058,
      "score_mae": 0.25833333333333336
    },
    "unseen": {
      "acc": 0.276,
      "brier": 0.23573027879807676,
      "ece15": 0.13670498710870743,
      "ovr_auc": 0.49734617594060015,
      "score_mae": 1.16
    },
    "b2_reference": {
      "acc": 0.7586666666666667,
      "auc": 0.9187387525390367,
      "unseen_acc": 0.71,
      "unseen_auc": 0.910485817772079,
      "jitter_flip": 0.056666666666666664,
      "multiq_delta_4dp": 0.0599,
      "ece_indomain": 0.18265156266666666
    }
  },
  "notes": [
    "reliability-curve monotonicity: manual check pending bins artifact",
    "in-domain/unseen numbers are post per-primitive TS (see gate_table)"
  ]
}
- 05:06:09 [gpu-manual] verdict + evidence + push done
drafted 0 cases (3 consecutive fails at stop)
- 05:07:39 [cpu] drafter end (0 cases)
- 05:07:39 [cpu] lane v6 done
- 05:09:05 [cpu] === NIGHT CPU LANE start (pid 1352405) ===
- 05:09:05 [cpu] night drafter start (gemma4:31b, 3055s to 06:00, max 400 total, num_predict=900)
drafted 0 cases (0 consecutive fails at stop)
- 05:09:11 [cpu] night drafter end — drafts at drafts/seen3_drafts.jsonl (0 cases)
- 05:09:11 [cpu] bake-off start (lfm3-omni vs gemma4:31b, 20 cases x 3 samples, 4849s window)
lfm3-omni:latest {'ok_cases': 15, 'mean_soft_kl': 1.3802826484397388, 'argmax_agree_frac': 0.0, 'cases_per_hour': 628.8543164274483, 'wall_min': 1.4311740835507711}
- 05:18:52 [cpu] === NIGHT CPU LANE start (pid 1372715) ===
- 05:18:52 [cpu] night drafter start (gemma4:31b, 2468s to 06:00, max 400 total, num_predict=900)
drafted 0 cases (0 consecutive fails at stop)
- 05:18:59 [cpu] night drafter end — drafts at drafts/seen3_drafts.jsonl (0 cases)
- 05:18:59 [cpu] bake-off start (lfm3-omni vs gemma4:31b, 20 cases x 3 samples, 4261s window)
lfm3-omni:latest {'ok_cases': 15, 'mean_soft_kl': 1.3802826484397388, 'argmax_agree_frac': 0.0, 'cases_per_hour': 646.0050137257987, 'wall_min': 1.3931780417760213}
gemma4:31b {'ok_cases': 19, 'mean_soft_kl': 1.2199574976824146, 'argmax_agree_frac': 0.10526315789473684, 'cases_per_hour': 23.724178187631605, 'wall_min': 48.05224404335022}
{
  "n_cases": 20,
  "samples": 3,
  "models": {
    "lfm3-omni:latest": {
      "ok_cases": 15,
      "mean_soft_kl": 1.3802826484397388,
      "argmax_agree_frac": 0.0,
      "cases_per_hour": 646.0050137257987,
      "wall_min": 1.3931780417760213
    },
    "gemma4:31b": {
      "ok_cases": 19,
      "mean_soft_kl": 1.2199574976824146,
      "argmax_agree_frac": 0.10526315789473684,
      "cases_per_hour": 23.724178187631605,
      "wall_min": 48.05224404335022
    }
  }
}
- 06:08:33 [cpu] bake-off PASS
- 06:08:33 [cpu] === NIGHT CPU LANE end ===
- 07:22:54 [audit] PHASE B re-run: fixed referee (fail-closed keys) + bare-key prompt; gemma4 20 cases x 1 sample
- 07:32:11 [audit] PHASE C: few-shot anchor (2 test-row examples) + same fixed referee; gemma4 20 cases x 1 sample
gemma4:31b {'ok_cases': 1, 'mean_soft_kl': 2.23726811128732, 'argmax_agree_frac': 0.0, 'by_question_type': {'choice': {'n': 2, 'agree_frac': 1.0}, 'noul': {'n': 1, 'agree_frac': 0.0}, 'score': {'n': 2, 'agree_frac': 1.0}}, 'cases_per_hour': 3.5790557947128057, 'wall_min': 16.76419800122579}
{
  "n_cases": 20,
  "samples": 1,
  "models": {
    "gemma4:31b": {
      "ok_cases": 1,
      "mean_soft_kl": 2.23726811128732,
      "argmax_agree_frac": 0.0,
      "by_question_type": {
        "choice": {
          "n": 2,
          "agree_frac": 1.0
        },
        "noul": {
          "n": 1,
          "agree_frac": 0.0
        },
        "score": {
          "n": 2,
          "agree_frac": 1.0
        }
      },
      "cases_per_hour": 3.5790557947128057,
      "wall_min": 16.76419800122579
    }
  }
}
- 07:39:46 [audit] PHASE B done rc=0
gemma4:31b {'ok_cases': 16, 'mean_soft_kl': 1.5815405839527106, 'argmax_agree_frac': 0.0, 'by_question_type': {'choice': {'n': 22, 'agree_frac': 0.7272727272727273}, 'noul': {'n': 26, 'agree_frac': 0.7307692307692307}, 'score': {'n': 32, 'agree_frac': 0.1875}}, 'cases_per_hour': 75.95056301098202, 'wall_min': 12.639800969759623}
{
  "n_cases": 20,
  "samples": 1,
  "few_shot": false,
  "models": {
    "gemma4:31b": {
      "ok_cases": 16,
      "mean_soft_kl": 1.5815405839527106,
      "argmax_agree_frac": 0.0,
      "by_question_type": {
        "choice": {
          "n": 22,
          "agree_frac": 0.7272727272727273
        },
        "noul": {
          "n": 26,
          "agree_frac": 0.7307692307692307
        },
        "score": {
          "n": 32,
          "agree_frac": 0.1875
        }
      },
      "cases_per_hour": 75.95056301098202,
      "wall_min": 12.639800969759623
    }
  }
}
gemma4:31b {'ok_cases': 20, 'mean_soft_kl': 0.22332041187887636, 'argmax_agree_frac': 0.35, 'by_question_type': {'choice': {'n': 26, 'agree_frac': 0.8076923076923077}, 'noul': {'n': 34, 'agree_frac': 0.7352941176470589}, 'score': {'n': 40, 'agree_frac': 0.65}}, 'cases_per_hour': 91.134321948568, 'wall_min': 13.167377277215321}
{
  "n_cases": 20,
  "samples": 1,
  "few_shot": true,
  "models": {
    "gemma4:31b": {
      "ok_cases": 20,
      "mean_soft_kl": 0.22332041187887636,
      "argmax_agree_frac": 0.35,
      "by_question_type": {
        "choice": {
          "n": 26,
          "agree_frac": 0.8076923076923077
        },
        "noul": {
          "n": 34,
          "agree_frac": 0.7352941176470589
        },
        "score": {
          "n": 40,
          "agree_frac": 0.65
        }
      },
      "cases_per_hour": 91.134321948568,
      "wall_min": 13.167377277215321
    }
  }
}
