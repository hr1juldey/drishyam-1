import contextlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_engine(tmp_path):
    spec = importlib.util.spec_from_file_location("next_step", ROOT / "scripts" / "next_step.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    road = tmp_path / "roadmap.md"
    road.write_text(
        "G1-A do a thing\n\nbody\n\nG1-B do b thing\n\nbody\n\n"
        "## Gate 2 preview\n\nG2-TF1 later\n"
    )
    st = tmp_path / "state.json"
    st.write_text(json.dumps({"schema": 1, "parked": [], "sub_tasks": {
        "G1-A": {"status": "todo", "attempts": 0, "note": "", "commit": "", "artifact": ""},
        "G1-B": {"status": "todo", "attempts": 0, "note": "", "commit": "", "artifact": ""}}}))
    mod.ROOT = tmp_path
    mod.STATE = st
    mod.ROADMAP = road
    mod.LOG = tmp_path / "log.md"
    mod.head_sha = lambda: "deadbee"  # no git needed in tests
    return mod, json.loads(st.read_text())


def _run(mod, *args):
    sys.argv = ["next_step.py", *args]
    with contextlib.suppress(SystemExit):
        mod.main()


def test_v1_done_without_commit_fails(tmp_path):
    mod, st = load_engine(tmp_path)
    st["sub_tasks"]["G1-A"].update(status="done", commit="")
    mod.STATE.write_text(json.dumps(st))
    _run(mod, "--verify")
    assert mod.verify() == 1


def test_v2_orphan_id(tmp_path):
    mod, st = load_engine(tmp_path)
    st["sub_tasks"]["G1-Z"] = st["sub_tasks"]["G1-A"].copy()
    mod.STATE.write_text(json.dumps(st))
    assert mod.verify() == 1


def test_v3_done_artifact_missing(tmp_path):
    mod, st = load_engine(tmp_path)
    st["sub_tasks"]["G1-A"].update(status="done", commit="abc", artifact="nope.json")
    mod.STATE.write_text(json.dumps(st))
    assert mod.verify() == 1


def test_v4_parked_never_next(tmp_path):
    mod, st = load_engine(tmp_path)
    st["sub_tasks"]["G1-A"]["status"] = "parked"
    mod.STATE.write_text(json.dumps(st))
    assert mod.next_id(json.loads(mod.STATE.read_text())) == "G1-B"


def test_three_fails_park(tmp_path):
    mod, _ = load_engine(tmp_path)
    for _ in range(3):
        _run(mod, "--fail", "G1-A", "--note", "x")
    st = json.loads(mod.STATE.read_text())
    assert st["sub_tasks"]["G1-A"]["status"] == "parked"
    assert st["parked"] == ["G1-A"]


def test_v2_ignores_gate2_preview(tmp_path):
    mod, st = load_engine(tmp_path)
    mod.STATE.write_text(json.dumps(st))
    assert mod.verify() == 0  # G2-TF1 in roadmap tail must NOT read as orphan
