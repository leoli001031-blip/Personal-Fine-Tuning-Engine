# Phase56 Runbook

```bash
.venv/bin/python tools/phase56_prepare.py --clean-evidence
.venv/bin/python tools/phase56_span_evaluator.py --split calibration
# Only after calibration status=qualified:
.venv/bin/python tools/phase56_span_evaluator.py --split holdout
.venv/bin/python tools/phase56_finalize_evidence.py
.venv/bin/python tools/phase56_validate.py
```

The independent holdout remains sealed until calibration qualifies. Models emit atomic fields plus exact evidence spans; deterministic grounding validates each atom before the composer owns every final label.
