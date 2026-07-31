# Phase58 Runbook

Start an isolated four-slot Ollama service in terminal A:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

Run the frozen gate in terminal B:

```bash
.venv/bin/python tools/phase58_prepare.py --clean-evidence
.venv/bin/python tools/phase58_clause_evaluator.py --split calibration --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase58_clause_evaluator.py --split holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase58_finalize_evidence.py
.venv/bin/python tools/phase58_validate.py
```

Do not modify the rubric, schema, fixture, grounding code, or composer after any calibration call. A failed calibration seals Phase58 without holdout.
