# Phase29 Completion Audit

## Status

Phase29 implementation, evidence generation, real probe, scoring, and verification gates are complete.
The remaining release step is blocked by local git metadata permissions in this Codex session.

## Completed Requirements

- Prior evidence review: `evidence/phase29_prereq_review.json` and `evidence/phase29_prereq_review.md`.
- Benefit contract: `evidence/phase29_benefit_contract.json`.
- Feedback source separation: `evidence-feedback/phase29_signal_routing_report.json`.
- Contract review scenario: `evidence/source_manifest.json`, `evidence/holdout.json`, 40 training-source tasks, and 30 holdout prompts.
- Feedback signals: `evidence-feedback/phase29_feedback_batch.jsonl` and `evidence-feedback/phase29_review_decisions.json`.
- Candidate generation: `evidence-candidates/selected_sft_samples.jsonl`, `evidence-candidates/selected_dpo_pairs.jsonl`, `candidate_manifest.json`, `candidate_quality_report.json`, and `holdout_integrity_check.json`.
- Model selection: `evidence/model_selection.json`.
- Primary real training probe: attempted 12-step MLX Qwen3-8B path and saved failure evidence in `evidence-training/training_attempt.json`.
- Fallback real DPO probe: completed Qwen2.5-0.5B DPO training and validated adapter artifact in `evidence-training/dpo-fallback-qwen25-0_5b/training_attempt.json`.
- Adapter artifact: `trainer_job_outputs/phase29-dpo-fallback-qwen25-0_5b/dpo_adapter/adapter_model.safetensors`.
- Holdout eval: `evidence-training/dpo-fallback-qwen25-0_5b/eval_report_rescored.json`.
- Runtime reference: real Ollama qwen3.6 outputs in `evidence-runtime-reference/ollama_qwen36_reference.json`.
- Output examples and preference report: `evidence-eval/output_examples.md` and `evidence-eval/preference_adherence_report.json`.
- Final decision: `phase29-final-decision.md`.

## Real Probe Result

- Primary Qwen3-8B MLX training: failed because Metal device is unavailable in the current headless/sandboxed session.
- Qwen2.5-0.5B DPO fallback: completed real training and produced a valid PEFT adapter artifact.
- Adapter improved citation hit rate and unsupported assertion count, but regressed structure adherence and did not improve preference adherence.
- Final recommendation: `archive`.

## Verified Commands

- `py_compile`: passed for Phase29 core, DPO executor, tool, and tests.
- `pytest tests/test_phase29_feedback_tuning_benefit.py tests/test_dpo_executor_unit.py -q`: 22 passed.
- `pytest tests/test_phase28_real_feedback_loop_engineering.py tests/test_phase28_real_feedback_loop_surface.py -q`: 13 passed.
- `python tools/phase29_feedback_driven_tuning_benefit.py --clean-evidence`: passed.
- `python tools/phase29_feedback_driven_tuning_benefit.py --clean-evidence --run-real-training --training-steps 12 --train-sample-limit 40 --eval-holdout-limit 30 --run-runtime-reference --runtime-reference-limit 5 --clean-training-output --run-dpo-fallback`: passed and saved real evidence.
- `git diff --check`: passed.
- `make test-unit test-surface test-e2e-mock smoke-beta`: passed.

## Blocked Release Step

The requested branch, commit, push, and draft PR could not be completed because `.git` is not writable in this session.

Observed command:

```bash
touch .git/.codex-write-test
```

Observed error:

```text
touch: .git/.codex-write-test: Operation not permitted
```

Current branch remains `codex/phase28-real-feedback-loop-engineering`.
