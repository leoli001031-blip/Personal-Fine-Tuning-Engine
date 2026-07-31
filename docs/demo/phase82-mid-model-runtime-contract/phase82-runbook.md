# Phase82 Runbook

```bash
.venv/bin/python tools/phase82_mid_model_runtime_contract.py prepare --clean
.venv/bin/python tools/phase82_mid_model_runtime_contract.py api-smoke --clean
.venv/bin/python tools/phase82_mid_model_runtime_contract.py generate --variant base_api_length_control --clean
.venv/bin/python tools/phase82_mid_model_runtime_contract.py generate --variant persona_api_contract --clean
.venv/bin/python tools/phase82_mid_model_runtime_contract.py full-regression
.venv/bin/python tools/phase82_mid_model_runtime_contract.py finalize
.venv/bin/python tools/phase82_mid_model_runtime_contract.py validate
```

The model revision, fresh holdout, product API surface, decoding controls, and thresholds are frozen before generation. Both variants use the same model and decoding controls; only the persona response contract differs.
