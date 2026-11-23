#!/usr/bin/env python
"""Amnesia-proof loop engine — docs/EXEC_ENGINE.md: next / --done / --fail / --verify."""

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "docs" / "EXEC_STATE.json"
ROADMAP = ROOT / "docs" / "ROADMAP.md"
LOG = ROOT / "docs" / "EXEC_LOG.md"
ID_RE = re.compile(r"^G1-[A-Z0-9]+\b", re.M)  # Gate-1 ids only; the G2 preview section is excluded


def roadmap_ids() -> list[str]:
    body = ROADMAP.read_text().split("## Gate 2 preview")[0]
    return ID_RE.findall(body)


def load_state() -> dict:
    return json.loads(STATE.read_text())


def save_state(s: dict) -> None:
    STATE.write_text(json.dumps(s, indent=2) + "\n")


def head_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def append_log(line: str) -> None:
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{datetime.datetime.now():%F %T} {line}\n")


def verify() -> int:
    s, ids, errs = load_state(), roadmap_ids(), []
    st = s["sub_tasks"]
    if set(st) - set(ids):
        errs.append(f"V2 orphan ids: {sorted(set(st) - set(ids))}")
    if set(ids) - set(st):
        errs.append(f"V2 missing ids: {sorted(set(ids) - set(st))}")
    for i, r in st.items():
        if r["status"] == "done":
            if not r["commit"]:
                errs.append(f"V1 {i}: done without commit sha")
            if r["artifact"] and not (ROOT / r["artifact"]).exists():
                errs.append(f"V3 {i}: artifact missing {r['artifact']}")
    if sorted(s["parked"]) != sorted(i for i, r in st.items() if r["status"] == "parked"):
        errs.append("V5 parked list mismatch")
    if errs:
        print("VERIFY FAIL")
        for e in errs:
            print(" -", e)
        return 1
    print("VERIFY OK: V1-V5 green")
    return 0


def next_id(s: dict) -> str | None:
    for i in roadmap_ids():
        r = s["sub_tasks"][i]
        if r["status"] == "in_progress":
            return i  # resume an interrupted attempt first
        if r["status"] == "todo" and i not in s["parked"]:  # V4: parked is never returned
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--done")
    ap.add_argument("--fail")
    ap.add_argument("--note", default="")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--init", action="store_true")
    a = ap.parse_args()
    if a.init:
        print("state initialised at docs/EXEC_STATE.json")
        return
    if a.verify:
        sys.exit(verify())
    s = load_state()
    if a.done or a.fail:
        i = a.done or a.fail
        rec = s["sub_tasks"][i]
        rec["attempts"] += 1
        rec["note"] = a.note
        if a.done:
            rec["status"], rec["commit"] = "done", head_sha()
            append_log(f"DONE {i} {rec['commit'][:8]} {a.note}")
        else:
            if rec["attempts"] >= 3 and i not in s["parked"]:
                rec["status"] = "parked"
                s["parked"].append(i)
            append_log(f"FAIL {i} attempt {rec['attempts']} {a.note}")
        save_state(s)
        print(f"{i}: {s['sub_tasks'][i]['status']}")
        return
    i = next_id(s)
    if i is None:
        print("ALL SUB-TASKS DONE OR PARKED")
        sys.exit(verify())
    s["sub_tasks"][i]["status"] = "in_progress"
    save_state(s)
    txt = ROADMAP.read_text()
    k = txt.find(f"\n{i} ")
    spec = txt[k + 1: txt.find("\n\n", k + 1)] if k >= 0 else "(see docs/ROADMAP.md)"
    print(f"NEXT {i} (attempt {s['sub_tasks'][i]['attempts'] + 1})\n{spec[:1200]}")


if __name__ == "__main__":
    main()
