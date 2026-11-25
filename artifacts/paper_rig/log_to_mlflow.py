"""Tails runs/overnight/train_v2.log and streams the trainer's own step lines
into MLflow so the CURRENT run (launched without --mlflow) is visible live.
Reads only; never touches the training process.
Usage: uv run python artifacts/paper_rig/log_to_mlflow.py  (exits after DONE)"""
import os, sys, time, re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LOG = os.path.join(ROOT, "runs", "overnight", "train_v2.log")

import mlflow

mlflow.set_tracking_uri("sqlite:///" + os.path.join(ROOT, "runs", "mlflow.db"))
mlflow.set_experiment("drishyam-doc")
run = mlflow.start_run(run_name="img-e4-live")
run_id = run.info.run_id
mlflow.log_params({"source": "log_tail", "log": LOG})
print(f"mlflow run {run_id}", flush=True)

PAT = re.compile(
    r"step (\d+)/(\d+) loss ([\d.]+) gn ([\d.]+) rows/s ([\d.]+) vram (\d+)MB")
offset = 0
new = ""
if os.path.exists(LOG):          # backfill everything already logged
    with open(LOG) as f:
        for line in f:
            m = PAT.search(line)
            if m:
                step, total, loss, gn, rps, vram = m.groups()
                mlflow.log_metrics({"loss": float(loss), "grad_norm": float(gn),
                                    "rows_per_s": float(rps), "vram_mb": float(vram)},
                                   step=int(step))
                offset += len(line)
                total_steps = int(total)
    mlflow.log_metric("progress_frac", 1.0, step=0)  # placeholder to create metric
print(f"backfilled up to byte {offset}", flush=True)

deadline = time.time() + 6 * 3600
while time.time() < deadline:
    time.sleep(20)
    if not os.path.exists(LOG):
        continue
    size = os.path.getsize(LOG)
    if size <= offset:
        continue
    with open(LOG) as f:
        f.seek(offset)
        new = f.read()
        offset = f.tell()
    for line in new.splitlines():
        m = PAT.search(line)
        if m:
            step, total, loss, gn, rps, vram = m.groups()
            mlflow.log_metrics({"loss": float(loss), "grad_norm": float(gn),
                                "rows_per_s": float(rps), "vram_mb": float(vram)},
                               step=int(step))
            print(f"logged step {step}", flush=True)
    if "DONE steps" in new or "SMOKE OK" in new or "divergence:" in new:
        mlflow.log_param("final_line", new.strip().splitlines()[-1][:200])
        break

mlflow.end_run(status="FINISHED" if "DONE steps" in new else "FAILED")
print("tail complete", flush=True)
