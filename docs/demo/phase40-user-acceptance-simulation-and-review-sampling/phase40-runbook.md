# Phase40 Runbook

Generate deterministic Phase40 user-acceptance simulation and review-sampling evidence:

```bash
.venv/bin/python tools/phase40_user_acceptance_simulation_and_review_sampling.py --clean-evidence
```

Default output contains simulated usage scenarios and pending manual review items only. It does not connect Hermes, train 27B, auto-promote, or claim actual product benefit.

To test reviewed-preference readiness, pass a JSON or JSONL file with explicit human review decisions:

```bash
.venv/bin/python tools/phase40_user_acceptance_simulation_and_review_sampling.py --review-decisions-json path/to/decisions.jsonl
```
