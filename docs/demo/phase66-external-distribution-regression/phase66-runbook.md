# Phase66 Runbook

Start an isolated Ollama service:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

Freeze and run the one-shot stages:

```bash
.venv/bin/python tools/phase66_prepare.py --clean-evidence
.venv/bin/python tools/phase66_execute.py --stage preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase66_execute.py --stage external_holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase66_execute.py --stage historical_replay --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase66_finalize_evidence.py
.venv/bin/python tools/phase66_validate.py
```

Do not rerun or tune a scored split after revealing its labels. Failed wire attempts remain evidence and are not parser-normalized.
