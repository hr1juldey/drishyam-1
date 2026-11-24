import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "artifacts", "paper_rig"))
from jit_pipeline import build_manifest

def test_producer_fills_three_slots():
    from producer import JITProducer
    rows, _ = build_manifest(64)
    p = JITProducer(rows, micro=8)
    p.start()
    got = []
    for _ in range(3):
        b = p.get(timeout=120)
        assert b["pixel_values"].shape == (8, 3, 896, 896)
        assert b["pixel_values"].dtype == torch.bfloat16
        assert b["input_ids"].shape[0] == 8
        assert b["attention_mask"].shape == b["input_ids"].shape
        assert b["labels"].shape[0] == 8
        got.append(b["family"])
    assert all(f in ("thermal", "savings", "card") for f in got)
    assert got[0] == "thermal" and len(set(got)) == 1   # 64-row manifest: all-thermal prefix
    p.stop.set()
