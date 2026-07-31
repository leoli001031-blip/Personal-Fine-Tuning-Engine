# Phase108 Runbook

```bash
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py prepare --clean
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py eval --variant base
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py eval --variant phase106_sft
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py eval --variant phase107_dpo
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py eval --variant phase107_dpo_no_runtime
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py analyze
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py decide
.venv/bin/python tools/phase108_runtime_adapter_causal_value_proof.py validate
```

All 260 planned evaluation calls are local Qwen3-4B simulated usage. The remaining 40-call reserve is unavailable unless one predeclared failure clearly dominates. No automatic retry, training, promotion, Hermes integration, external provider, push, or deployment is allowed.
