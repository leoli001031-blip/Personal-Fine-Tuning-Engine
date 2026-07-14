# Phase59 Runbook

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

In a second terminal:

```bash
.venv/bin/python tools/phase59_prepare.py --clean-evidence
.venv/bin/python tools/phase59_candidate_evaluator.py --split calibration --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase59_candidate_evaluator.py --split holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase59_finalize_evidence.py
.venv/bin/python tools/phase59_validate.py
```

Do not change candidate generation, fixture audits, prompts, schemas, or gates after any calibration call. A failed calibration seals Phase59 without holdout.
