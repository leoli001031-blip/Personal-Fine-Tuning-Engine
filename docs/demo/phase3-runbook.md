# Phase3 Signal Loop Runbook

更新时间：2026-06-19

Phase3 moves PFE from one-off demo material into a continuous signal-driven
finetuning loop:

```text
persona/scenario -> interaction feedback -> signal inbox -> routing policy
-> training candidates -> candidate adapter plan -> eval gate -> promote/archive
```

This first cut does not require real large-model training. It proves the
interfaces, safety policy, Studio surface, and smoke path that a later trainer
can consume.

## Demo Assumptions

- You are working in `/Users/lichenhao/Desktop/PFE`.
- `videos/` remains untracked local media and is not part of Phase3.
- `.venv` exists and can run the test suite.
- Phase3 starts generic. The built-in vertical is only an example:
  `合同摘要与风险标注`.

The contract scenario is deliberately bounded:

- It summarizes and flags risk.
- It does not provide legal conclusions, lawsuit strategy, or deterministic
  compliance judgment.
- It requires human confirmation for real contracts or high-risk decisions.

## 1. Run The Core/API Smoke

```bash
.venv/bin/python tools/phase3_api_smoke.py
```

Expected shape:

```json
{
  "ok": true,
  "persona": "ops-analyst",
  "scenario": "contract-risk-summary",
  "signal_type": "correction",
  "eligible_for_training": true,
  "candidate_sample_count": 1,
  "plan_state": "planned",
  "eval_gate": "ready_for_eval"
}
```

## 2. Verify The Tests

Run the targeted Phase3 gate:

```bash
.venv/bin/python -m pytest \
  tests/test_phase3_signal_loop.py \
  tests/test_server_http.py::ServerHttpSmokeTests::test_phase3_api_exposes_signal_loop_and_candidate_plan \
  tests/test_server_http.py::ServerHttpSmokeTests::test_feedback_endpoint_mirrors_phase3_inbox \
  tests/test_server_http.py::ServerHttpSmokeTests::test_studio_frontend_serves_user_facing_control_surface \
  -q
```

The test set proves:

- persona/scenario schema validation
- signal routing for accept/reject/edit/preference/correction/safety_block
- PII and high-risk domain exclusion
- candidate sample generation
- Phase3 API smoke
- Studio minimal UI exposure

## 3. Start Studio

```bash
env \
  PFE_HOME=/tmp/pfe-phase3-demo-home \
  PFE_WORKSPACE=phase3_demo \
  PYTHONPATH="$PWD/pfe-core:$PWD/pfe-cli:$PWD/pfe-server" \
  .venv/bin/python -m pfe_cli.main serve \
    --host 127.0.0.1 \
    --port 8921 \
    --workspace phase3_demo \
    --live
```

Open:

```text
http://127.0.0.1:8921/studio
```

The right side of Studio should include `信号闭环` with:

- Persona
- Scenario
- recent signal inbox rows
- candidate training state
- eval gate state
- `采集示例`
- `生成计划`

## 4. Drive The Phase3 Loop

Use the Studio buttons for the shortest path:

1. Click `采集示例`.
2. Confirm one `correction` signal appears.
3. Confirm the row is marked `train`.
4. Click `生成计划`.
5. Confirm candidate training shows `planned / 1 samples`.
6. Confirm Eval gate is `ready_for_eval`.

Or run the same flow through API:

```bash
curl -s http://127.0.0.1:8921/pfe/phase3/signals \
  -H 'content-type: application/json' \
  -d '{
    "signal_type": "edit",
    "user_input": "请整理合同交付条款：乙方需 7 日内交付。",
    "model_output": "该条款没有风险，可以直接签。",
    "corrected_output": "摘要：乙方需 7 日内交付。风险提示：违约金和验收口径需人工确认。本输出不是法律结论。",
    "confidence": 0.9
  }'
```

Export training candidates:

```bash
curl -s http://127.0.0.1:8921/pfe/phase3/training-candidates
```

Generate a candidate adapter plan:

```bash
curl -s http://127.0.0.1:8921/pfe/phase3/candidate-plan \
  -H 'content-type: application/json' \
  -d '{"persona_id":"ops-analyst","scenario_id":"contract-risk-summary"}'
```

## 5. Routing Policy Checklist

Expected routing:

- `accept`: training candidate when context and model output exist.
- `edit` or `correction`: training candidate when a corrected output exists.
- `reject`: review-only until paired with a chosen output.
- `preference`: profile-first; training only after repeated or confirmed
  reinforcement.
- `safety_block`: review/discard, never training.
- PII or high-risk legal/medical/financial conclusion: excluded from training.

The plan exposes existing lifecycle handoff URLs:

- `/pfe/training/jobs`
- `/pfe/eval`
- `/pfe/candidate/promote`
- `/pfe/candidate/archive`

## 6. Pass Criteria

Phase3 is ready for demo when:

- API smoke prints `ok: true`.
- targeted tests pass.
- Studio shows `信号闭环`.
- at least one signal can become a training candidate.
- candidate plan returns `plan_state=planned`.
- eval gate returns `ready_for_eval`.
- no files under `videos/` are modified.
