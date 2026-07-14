# Phase75 Runbook

```bash
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py prepare --clean-evidence
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py generate --variant base_minimal --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py generate --variant base_persona_runtime --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py generate --variant archived_adapter_minimal --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py generate --variant archived_adapter_persona_runtime --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py prepare-eval
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py judge --model gemma4:31b --ollama-endpoint http://127.0.0.1:11435 --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py judge --model qwen3.6 --ollama-endpoint http://127.0.0.1:11435 --clean
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py finalize
.venv/bin/python tools/phase75_personalization_benefit_benchmark.py validate
```

Phase75 uses 48 frozen simulated_usage sessions, four real Qwen3-4B arms, and two independent local Ollama judges. The Phase45 adapter is an archived eval-only negative control. No new training, Hermes attachment, default change, automatic promotion, or real-user claim is allowed.
