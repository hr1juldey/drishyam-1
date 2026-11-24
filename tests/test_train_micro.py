import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "artifacts", "paper_rig"))

def test_micro_step_grads_and_loss():
    from train_2lakh import build_model, build_opt, trainables
    from producer import JITProducer
    from jit_pipeline import build_manifest
    rows, _ = build_manifest(8)
    p = JITProducer(rows, micro=8)
    p.start()
    cfg = {"model_snapshot": os.path.expanduser(
        "~/.cache/huggingface/hub/models--google--t5gemma-2-270m-270m/snapshots/7c38f16641f455ef0685b18431faf1b17722d5a1")}
    model = build_model(cfg)
    opt = build_opt(model, cfg)
    b = p.get(timeout=180)
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                pixel_values=b["pixel_values"],
                decoder_input_ids=b["dec_in"], decoder_attention_mask=b["dmask"],
                labels=b["labels"])
    assert torch.isfinite(out.loss), f"loss not finite: {out.loss}"
    out.loss.backward()
    tr = trainables(model)
    assert any("decoder" in n for n in tr), "decoder must be trainable"
    assert any("multi_modal_projector" in n for n in tr), "projector must be trainable"
    assert not any("vision_tower" in n for n in tr), "SigLIP must stay frozen"
    assert not any("encoder.text_model" in n for n in tr), "text encoder must stay frozen"
    p.stop.set()
