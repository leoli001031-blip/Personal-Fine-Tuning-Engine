# Phase57 Runbook

Start an isolated four-slot Ollama service in terminal A:

```bash
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 ollama serve
```

Run the frozen replay in terminal B:

```bash
.venv/bin/python tools/phase57_prepare.py --clean-evidence
.venv/bin/python tools/phase57_historical_replay.py --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase57_finalize_evidence.py
.venv/bin/python tools/phase57_validate.py
```

The historical replay is a one-shot external qualification. Its results may be analyzed and archived, but never used to tune the frozen Phase56 evaluator in Phase57.
