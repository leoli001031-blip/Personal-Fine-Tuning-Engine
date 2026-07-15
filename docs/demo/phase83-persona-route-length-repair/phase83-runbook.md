# Phase83 Runbook

```bash
.venv/bin/python tools/phase83_persona_route_length_repair.py prepare --clean
.venv/bin/python tools/phase83_persona_route_length_repair.py api-smoke --clean
.venv/bin/python tools/phase83_persona_route_length_repair.py generate --variant base_api_length_control_160 --clean
.venv/bin/python tools/phase83_persona_route_length_repair.py generate --variant persona_api_contract_v2 --clean
.venv/bin/python tools/phase83_persona_route_length_repair.py full-regression
.venv/bin/python tools/phase83_persona_route_length_repair.py finalize
.venv/bin/python tools/phase83_persona_route_length_repair.py validate
```

The model revision, fresh holdout, per-turn route audit, API contract, decoding controls, and thresholds are frozen before generation. Both variants use identical model and decoding controls; only the V2 response contract differs.
