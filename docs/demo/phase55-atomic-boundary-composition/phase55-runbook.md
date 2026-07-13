# Phase55 Runbook

```bash
.venv/bin/python tools/phase55_prepare.py --clean-evidence
.venv/bin/python tools/phase55_atomic_evaluator.py --split calibration
# Only after calibration status=qualified:
.venv/bin/python tools/phase55_atomic_evaluator.py --split holdout
.venv/bin/python tools/phase55_finalize_evidence.py
.venv/bin/python tools/phase55_validate.py
```

The independent holdout remains sealed until calibration qualifies. Models emit atomic fields only; the deterministic composer derives boundary completeness and owns every final label.
