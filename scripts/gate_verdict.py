"""G1-G1: assemble the verdict — criteria 1-5 from gate_table + baselines + B2."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="artifacts/gate_table.json")
    ap.add_argument("--out", default="artifacts/verdict.json")
    a = ap.parse_args()
    gt = json.loads((ROOT / a.gate).read_text())
    b1 = json.loads((ROOT / "artifacts/baseline_report.json").read_text())
    b2 = json.loads((ROOT / "artifacts/b2_report.json").read_text())

    ab = gt["state_ablation"]
    mq = gt["multiq"]
    unseen = gt["unseen_workflow"]
    indomain = gt["in_domain_test"]

    c1 = {
        "flip_rate": ab["state_ablation_flip_rate"], "bar": "> 0.30",
        "acc_drop": ab["acc_drop"], "acc_bar": "> 0.15",
        "pass": ab["state_ablation_flip_rate"] > 0.30 and ab["acc_drop"] > 0.15,
    }
    c2 = {
        "max_delta_4dp": mq["max_delta_4dp"], "bar": "== 0.0",
        "speedup": mq["median_speedup"], "speedup_bar": "> 1.5",
        "pass": mq["max_delta_4dp"] == 0.0 and mq["median_speedup"] > 1.5,
    }
    c3 = {
        "jitter_flip": gt["jitter_flip_rate"], "bar": "< 0.05",
        "post_ts_ece_indomain": indomain["ece15"], "ece_bar": "< 0.08",
        "pass": gt["jitter_flip_rate"] < 0.05 and indomain["ece15"] < 0.08,
    }
    c4 = {
        "unseen_auc": unseen["ovr_auc"], "bar": "> 0.75",
        "unseen_acc": unseen["acc"], "majority_acc": b1["unseen_test"]["acc"],
        "pass": unseen["ovr_auc"] > 0.75 and unseen["acc"] > b1["unseen_test"]["acc"],
    }
    c5 = {
        "unseen_post_ts_ece": unseen["ece15"], "bar": "< 0.12",
        "pass": unseen["ece15"] < 0.12,
    }

    b2_row = {
        "acc": b2["test"]["acc"], "auc": b2["test"]["ovr_auc"],
        "unseen_acc": b2["unseen_test"]["acc"], "unseen_auc": b2["unseen_test"]["ovr_auc"],
        "jitter_flip": b2["jitter_flip_rate"], "multiq_delta_4dp": b2["multiq"]["max_delta_4dp"],
        "ece_indomain": b2["test"]["ece15"],
    }
    verdict = {
        "ckpt": gt.get("ckpt"), "temperatures": gt.get("temperatures"),
        "criteria": {"1_state_ablation": c1, "2_invariance": c2, "3_robustness_calibration": c3,
                     "4_unseen_generalization": c4, "5_unseen_ece": c5},
        "gate": "GO" if all(c["pass"] for c in (c1, c2, c3, c4, c5)) else "NO-GO",
        "failed": [k for k, c in {"1": c1, "2": c2, "3": c3,
                                  "4": c4, "5": c5}.items() if not c["pass"]],
        "ours_vs_b2": {"in_domain": {"acc": indomain["acc"], "brier": indomain["brier"],
                                     "ece15": indomain["ece15"], "ovr_auc": indomain["ovr_auc"],
                                     "score_mae": indomain["score_mae"]},
                       "unseen": {"acc": unseen["acc"], "brier": unseen["brier"],
                                  "ece15": unseen["ece15"], "ovr_auc": unseen["ovr_auc"],
                                  "score_mae": unseen["score_mae"]},
                       "b2_reference": b2_row},
        "notes": ["reliability-curve monotonicity: manual check pending bins artifact",
                  "in-domain/unseen numbers are post per-primitive TS (see gate_table)"],
    }
    (ROOT / a.out).write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
