# Phase87-89 Failure-driven Adapter Loop

This loop uses only local Qwen2.5-1.5B and simulated_usage evidence.

```bash
.venv/bin/python tools/phase85_metric_schema_v2_overlay.py finalize
.venv/bin/python tools/phase85_metric_schema_v2_overlay.py validate
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py prepare --clean
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py train --steps 5 --clean
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py generate --scope sanity --variant base --clean
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py generate --scope sanity --variant adapter --clean
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py sanity
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py train --steps 25 --clean
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py generate --scope full --variant base --clean
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py generate --scope full --variant adapter --clean
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py review-template --clean
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py review-validate
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py full-regression
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py finalize
.venv/bin/python tools/phase87_89_failure_driven_adapter_loop.py validate
```

No automatic promotion, deployment, Hermes attachment, or actual-user-benefit claim is permitted.
