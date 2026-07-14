# Phase61 Runbook

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

In a second terminal:

```bash
.venv/bin/python tools/phase61_prepare.py --clean-evidence
.venv/bin/python tools/phase61_execute.py --stage preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase61_execute.py --stage calibration --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase61_execute.py --stage holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase61_finalize_evidence.py
.venv/bin/python tools/phase61_validate.py
```

Do not change candidate generation, fixtures, prompts, wire parser, retry count, or gates after prepare. Stop after the first failed gate and finalize that path.
