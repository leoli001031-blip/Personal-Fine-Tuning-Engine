# Phase51 Runbook

```bash
.venv/bin/python tools/phase51_prepare.py --clean-evidence
.venv/bin/python tools/phase51_dual_evaluator.py --split calibration
.venv/bin/python tools/phase51_dual_evaluator.py --split holdout
.venv/bin/python tools/phase51_qwen3_4b_generate.py --variant base_compact_v1 --clean
.venv/bin/python tools/phase51_qwen3_4b_generate.py --variant base_global_v2 --clean
.venv/bin/python tools/phase51_qwen3_4b_generate.py --variant base_conditional_guard --clean
.venv/bin/python tools/phase51_dual_evaluator.py --split runtime
.venv/bin/python tools/phase51_finalize_evidence.py
.venv/bin/python tools/phase51_validate.py
```

Calibration attempt-01 is preserved under `evidence-evaluator-debug/` and is not formal evidence. The independent holdout was not called until calibration attempt-02 qualified. Runtime generation was not called until the independent evaluator holdout qualified.
