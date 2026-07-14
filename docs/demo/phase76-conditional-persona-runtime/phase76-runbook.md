# Phase76 Runbook

```bash
.venv/bin/python tools/phase76_conditional_persona_runtime.py prepare --clean-evidence
.venv/bin/python tools/phase76_conditional_persona_runtime.py generate --variant base_minimal --clean
.venv/bin/python tools/phase76_conditional_persona_runtime.py generate --variant conditional_persona_runtime --clean
.venv/bin/python tools/phase76_conditional_persona_runtime.py prepare-eval
.venv/bin/python tools/phase76_conditional_persona_runtime.py judge --model gemma4:31b --ollama-endpoint http://127.0.0.1:11435 --clean
.venv/bin/python tools/phase76_conditional_persona_runtime.py judge --model qwen3.6 --ollama-endpoint http://127.0.0.1:11435 --clean
.venv/bin/python tools/phase76_conditional_persona_runtime.py finalize
.venv/bin/python tools/phase76_conditional_persona_runtime.py full-regression
.venv/bin/python tools/phase76_conditional_persona_runtime.py validate
```

Phase76 freezes 36 persona targets and 12 ordinary controls. The conditional arm must beat base on the target slice while producing byte-identical ordinary transcripts. No training, adapter lifecycle change, Hermes attachment, automatic promotion, or real-user claim is allowed.
