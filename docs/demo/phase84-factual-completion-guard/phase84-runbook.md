# Phase84 Runbook

```bash
.venv/bin/python tools/phase84_factual_completion_guard.py prepare --clean
.venv/bin/python tools/phase84_factual_completion_guard.py api-smoke --clean
.venv/bin/python tools/phase84_factual_completion_guard.py generate --variant base_api_length_control_160 --clean
.venv/bin/python tools/phase84_factual_completion_guard.py generate --variant persona_api_contract_v3 --clean
.venv/bin/python tools/phase84_factual_completion_guard.py full-regression
.venv/bin/python tools/phase84_factual_completion_guard.py finalize
.venv/bin/python tools/phase84_factual_completion_guard.py validate
```

The model revision, Phase83 canonical reference and holdout, fresh Phase84 holdout, scorer and runtime source hashes, per-turn route audit, API contract, decoding controls, and complete gate thresholds are frozen before generation. Both variants use identical model and decoding controls; only the V3 response contract and factual-completion guard differ.
