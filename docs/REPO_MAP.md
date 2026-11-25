# REPO_MAP (regenerate: uv run python scripts/repo_map.py --save; committed artifact — absence claims cite this)

```
scripts/bakeoff.py  [331L]  call, extract_json, label_prompt, few_shot_examples, _num, build_alias_map, answers_to_gold_keys, mean_kl, sample_cases, bakeoff, drafter, main
scripts/gate_verdict.py  [77L]  main
scripts/harness.py  [257L]  contract_text, few_shot_examples_v2, prepare, score, main
scripts/next_step.py  [122L]  roadmap_ids, load_state, save_state, head_sha, append_log, verify, next_id, main
scripts/repo_map.py  [40L]  main
src/drishyam/__init__.py  [1L]  
src/drishyam/augs.py  [24L]  shuffled_options, jitter_state
src/drishyam/b2_eval.py  [155L]  load_engine, raw_request, align_probs, main
src/drishyam/baselines.py  [243L]  jitter_states_by_case, majority_dist, ovr_auc, metrics_rows, question_key, fit_baselines, predict_rows, fit_priors, predict_unseen_rows, main
src/drishyam/calibrate.py  [75L]  _softmax, apply_temperature, _rows, fit_temperature, fit_all, main
src/drishyam/data_inspect.py  [123L]  parquet_sha, q_prim, to_questions, to_gold, main
src/drishyam/dataset.py  [44L]  CaseDataset, load_seen_splits
src/drishyam/evaluate.py  [209L]  load_model, predict_rows, state_batch, flip_rate, max_prob_delta, timed_cuda, multiq_suite, ablation_suite, main
src/drishyam/losses.py  [25L]  ce_soft, brier, rps, question_loss
src/drishyam/model.py  [120L]  build_backbone, attach_lora, SlotHead, DecoupledDecision, trainable_groups
src/drishyam/probe.py  [20L]  main
src/drishyam/sanity_gate.py  [121L]  ManualAdamW, load_backbone, run_arm, main
src/drishyam/schemas.py  [75L]  Primitive, Question, SanityStep, SanityReport, SplitSpec, DecisionRow, GateInputs
src/drishyam/splits.py  [57L]  stratify_240_60, main
src/drishyam/stress.py  [57L]  jitter_state, sha, build, main
src/drishyam/tokenize.py  [36L]  add_slot_token, question_text, collate_questions
src/drishyam/tracking.py  [13L]  mlflow_run
src/drishyam/train.py  [270L]  git_sha, fixed_len_for, state_batch, ece15, run_epoch, evaluate_split, main
tests/test_baselines.py  [33L]  test_majority_dist_hand_computed, test_metrics_rows_hand_computed, test_metrics_flatten_identical
tests/test_calibrate.py  [37L]  test_temperature_identity, _sm, test_temperature_recovery, test_temperature_direction
tests/test_engine.py  [78L]  load_engine, _run, test_v1_done_without_commit_fails, test_v2_orphan_id, test_v3_done_artifact_missing, test_v4_parked_never_next, test_three_fails_park, test_v2_ignores_gate2_preview
tests/test_evaluate_units.py  [37L]  test_flip_rate_hand_computed, test_max_prob_delta_ragged, test_blank_state_flag_shapes
tests/test_losses.py  [41L]  test_ce_brier_hand_computed, test_rps_hand_computed, test_choice_order_shuffled_loss_invariant
tests/test_model_shapes.py  [63L]  tiny_setup, encode_q, rows_of, test_slot_shapes_and_invariance, test_question_text_format
tests/test_pack_batch.py  [33L]  _proc, test_placeholders_are_256, test_labels_mask_prompt_and_keep_answer, test_deterministic_same_seed, test_answer_fits_budget
tests/test_producer.py  [21L]  test_producer_fills_three_slots
tests/test_row_labels.py  [59L]  _thermal_gt, test_thermal_total_value_when_clean, test_thermal_total_unreadable_when_smudged_heavy, test_one_question_per_call_and_rotation_changes_field, test_savings_and_card_answer_from_scalar_gt, test_new_families_answer_from_gt, test_loio_holdout_bucket
tests/test_train_micro.py  [27L]  test_micro_step_grads_and_loss
TOTAL: 33 py files
```
