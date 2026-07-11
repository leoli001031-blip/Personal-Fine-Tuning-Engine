# PFE Phase42 Runbook

## Scope

Phase42 hardens four product paths: adapter serving quality, real local PEFT
training, OpenAI-compatible streaming/context accounting, and local privacy.
It does not claim that the newly trained probe improves product behavior.

## Preconditions

- Worktree: `/Users/lichenhao/Desktop/PFE`
- Branch: `codex/phase42-trustworthy-training-runtime-hardening`
- Local probe model: `models/Qwen2.5-0.5B-Instruct`
- Do not modify `videos/` or any existing Hermes profile.

## Loop A: Adapter Serving Gate

Capture the promoted adapter before changing lifecycle state:

```bash
.venv/bin/python tools/phase42_trustworthy_training_runtime_hardening.py --clean-evidence
```

The command runs the same 20 generic prompts against base and explicit adapter
`20260617-006`. Inspect:

- `evidence-adapter-gate/base_outputs.jsonl`
- `evidence-adapter-gate/adapter_outputs.jsonl`
- `evidence-adapter-gate/adapter_holdout_report.json`

Only after the report is persisted, apply the lifecycle decision:

```bash
.venv/bin/python tools/phase42_trustworthy_training_runtime_hardening.py \
  --apply-adapter-decision --reuse-existing-holdout
.venv/bin/pfe next --workspace user_default
```

Expected state: `base_ready`. Archival removes `latest` but retains the version
directory, manifest, quality report, and adapter weights.

## Loop B: Real Local PEFT

```bash
.venv/bin/python tools/phase42_real_local_peft_smoke.py --clean
```

Completion requires all of the following:

- 12 optimizer/backward steps;
- finite loss history;
- trainable parameter SHA256 changes;
- a parseable safetensors adapter with non-empty LoRA tensors;
- no JSON or placeholder file with a `.safetensors` suffix.

The 34 MB adapter stays under ignored `trainer_job_outputs/`. The committed
evidence stores hashes, loss, tensor counts, paths, and parameter fingerprints.

## Loop C: Hermes and Context

```bash
.venv/bin/python tools/server_live_smoke.py --request-timeout 15
.venv/bin/python tools/phase42_context_budget_smoke.py
```

The live smoke validates raw SSE and consumes the same endpoint through the
standard OpenAI Python SDK, matching the protocol shape used by a Hermes custom
provider. PFE emits an immediate assistant-role chunk, keep-alive comments,
content deltas, a terminal `finish_reason`, and `data: [DONE]`.

The current backend capability is explicitly `buffered_backend`; PFE does not
claim token-level incremental generation. The 4K smoke proves only the configured
4096-token path, not the model's declared maximum context.

## Loop D: Security and Candidate Quality

```bash
.venv/bin/pytest -q \
  tests/test_phase42_reliability_hardening.py \
  tests/test_phase41_simulated_review_preferences.py \
  tests/test_trainer_real_peft_job.py \
  tests/test_inference_runtime.py \
  tests/test_server_http.py \
  tests/test_server_adapters.py
```

The tests cover path-safe user IDs, explicit long-term-memory consent, temporary
interaction separation, all five Dashboard data endpoints, wrong API keys,
SSE completion/disconnect behavior, token budgets, and preference diversity.

Phase41's original 24 pairs must be blocked. Phase41-v2 must remain labelled
`simulated_usage=true`, `actual_model_call=false`, and cannot launch training.

## Final Verification

```bash
.venv/bin/python tools/phase42_finalize_evidence.py
make test-unit test-surface test-e2e-mock smoke-beta
git diff --check
```

`reliability_gate_passed` does not authorize automatic training or promotion.
