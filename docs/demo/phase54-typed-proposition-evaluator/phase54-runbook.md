# Phase54 Runbook

```bash
.venv/bin/python tools/phase54_prepare.py --clean-evidence
.venv/bin/python tools/phase54_typed_evaluator.py --split calibration
# Only after calibration status=qualified:
.venv/bin/python tools/phase54_typed_evaluator.py --split holdout
.venv/bin/python tools/phase54_finalize_evidence.py
.venv/bin/python tools/phase54_validate.py
```

The independent holdout remains sealed until calibration qualifies. Model outputs contain typed proposition fields only; the deterministic composer owns every final label.
