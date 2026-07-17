# Phase109 Runbook

```bash
.venv/bin/python tools/phase109_personal_engineering_copilot.py prepare --clean
.venv/bin/python tools/phase109_personal_engineering_copilot.py train --steps 1 --clean
.venv/bin/python tools/phase109_personal_engineering_copilot.py train --steps 12 --clean
.venv/bin/python tools/phase109_personal_engineering_copilot.py train --steps 30 --clean
.venv/bin/python tools/phase109_personal_engineering_copilot.py eval --variant base
.venv/bin/python tools/phase109_personal_engineering_copilot.py eval --variant phase107_dpo
.venv/bin/python tools/phase109_personal_engineering_copilot.py eval --variant phase109_personal_dpo
.venv/bin/python tools/phase109_personal_engineering_copilot.py analyze
.venv/bin/python tools/phase109_personal_engineering_copilot.py decide
.venv/bin/python tools/phase109_personal_engineering_copilot.py validate
```

The experiment uses 42 historical-signal-derived simulated preference pairs and 35 fresh simulated multi-turn holdout sessions. All 105 model calls are local Qwen3-4B. Raw generated transcripts stay under `/private/tmp`; no external provider, push, deployment, automatic retraining, or automatic promotion is permitted.
