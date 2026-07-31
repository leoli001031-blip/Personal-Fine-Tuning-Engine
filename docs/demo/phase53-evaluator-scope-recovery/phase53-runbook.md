# Phase53 Runbook

```bash
.venv/bin/python tools/phase53_prepare.py --clean-evidence
.venv/bin/python tools/phase53_dual_evaluator.py --split calibration
# Only after calibration status=qualified:
.venv/bin/python tools/phase53_dual_evaluator.py --split holdout
.venv/bin/python tools/phase53_finalize_evidence.py
.venv/bin/python tools/phase53_validate.py
```

Attempts 01 and 02 are preserved under `evidence-evaluator-debug/` and are not formal evidence. The independent holdout remains untouched until the formal calibration qualifies. Phase53 never runs Phase51 runtime replay or creates a boundary-clause design.
