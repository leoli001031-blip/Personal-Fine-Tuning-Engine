# Phase52 Runbook

```bash
.venv/bin/python tools/phase52_prepare.py --clean-evidence
.venv/bin/python tools/phase52_dual_evaluator.py --split calibration
.venv/bin/python tools/phase52_dual_evaluator.py --split holdout
.venv/bin/python tools/phase52_dual_evaluator.py --split replay
.venv/bin/python tools/phase52_finalize_evidence.py
.venv/bin/python tools/phase52_validate.py
```

Calibration attempt-01 is preserved under `evidence-evaluator-debug/` and is not formal evidence. Holdout was untouched until the revised calibration qualified. Phase51 runtime replay was not called until the independent holdout qualified.
