"""Stage F: morning report — always written, pass or fail. States only measured facts."""
import os, sys, json, argparse

def tail(path, n=15):
    if not os.path.exists(path):
        return "(no log)"
    return "\n".join(open(path).read().strip().splitlines()[-n:])

def main():
    ap = argparse.ArgumentParser()
    for k in ("chain-log", "train-log", "eval-json", "eval-log", "verify-a",
              "verify-b", "slice2-log", "smoke-log", "out"):
        ap.add_argument(f"--{k}", default=None)
    for s in "ABCDE":
        ap.add_argument(f"--stage-{s.lower()}", default="NOT_RUN")
    a = ap.parse_args()

    L = []
    L.append("# Overnight Report — " + os.popen("date '+%Y-%m-%d %H:%M'").read().strip())
    L.append("")
    L.append("| stage | what | result |")
    L.append("|---|---|---|")
    L.append(f"| A | slice-1 embeddings 50k (thermal/savings/card) | **{a.stage_a}** |")
    L.append(f"| B | slice-2 embeddings 24k (electricity/telecom/salary/table) | **{a.stage_b}** |")
    L.append(f"| C | trainer GPU smoke | **{a.stage_c}** |")
    L.append(f"| D | decoder training (projector+SigLIP frozen) | **{a.stage_d}** |")
    L.append(f"| E | LOIO eval (holdout seed%10==7) | **{a.stage_e}** |")
    L.append("")

    # data state
    for name, vf in (("slice-1", a.verify_a), ("slice-2", a.verify_b)):
        if vf and os.path.exists(vf):
            try:
                v = json.load(open(vf))
                L.append(f"**{name}:** " + "; ".join(
                    f"{r['dir']}: {r.get('rows','?')} rows / {r.get('shards','?')} shards"
                    + ("" if r["ok"] else f" — VERIFY ISSUE: {r['why']}")
                    for r in v))
                L.append("")
            except Exception as e:
                L.append(f"**{name}:** verify file unreadable: {e}")
                L.append("")

    # eval numbers
    if a.eval_json and os.path.exists(a.eval_json):
        ev = json.load(open(a.eval_json))
        L.append("## Eval (greedy decode, embeddings in = answer out)")
        L.append("")
        L.append("| split | n | exact match % |")
        L.append("|---|---|---|")
        for split, v in ev.items():
            L.append(f"| {split} | {v['n']} | {v['exact_match']} |")
        L.append("")
        L.append("Read: `train_inst` = institutions seen in training; `holdout_loio` = "
                 "held-out institutions (the generalization number). Gap between them "
                 "is the honest measure of memorization vs learning.")
        L.append("")

    L.append("## Morning usage")
    L.append("")
    L.append("```bash")
    L.append("cd ~/Documents/code/drishyam-1")
    L.append("uv run python artifacts/paper_rig/infer_docs.py --image <png> \\")
    L.append('  --q "What is the total amount?"')
    L.append("```")
    L.append("")
    if a.train_log and os.path.exists(a.train_log):
        L.append("## Training log tail")
        L.append("```")
        L.append(tail(a.train_log))
        L.append("```")
        L.append("")

    nxt = []
    if a.stage_d in ("PASS",):
        nxt.append("Model trained end-to-end. Run the full-pipeline spot check on a "
                   "freshly rendered page (render_896 -> infer_docs) — embedding-space "
                   "eval passing does not prove the pixel path; render 5 pages and check answers.")
    if a.stage_d == "HALTED_DIVERGENCE":
        nxt.append("Training halted by divergence guard. Inspect runs/overnight/train.log; "
                   "do NOT trust the final checkpoint before rerunning shorter LR.")
    if a.stage_d == "FAIL":
        nxt.append("Training failed. Checkpoint may not exist; check runs/overnight/train.log.")
    if a.stage_e == "FAIL":
        nxt.append("Eval failed but training may be fine — rerun eval_embeds manually.")
    if a.stage_c == "FAIL":
        nxt.append("Trainer smoke failed — training never started; embeddings are intact "
                   "and still the night's deliverable. Debug the trainer in daylight.")
    if nxt:
        L.append("## Next actions")
        L.append("")
        for n in nxt:
            L.append(f"- {n}")
        L.append("")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"report written: {a.out}")

if __name__ == "__main__":
    main()
