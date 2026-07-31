# Phase100-104 Autonomous Loop Runbook

All commands are local-only and use `models/Qwen3-4B`.

```bash
.venv/bin/python tools/phase100_qwen3_generation_boundary_closure.py prepare --clean
.venv/bin/python tools/phase100_qwen3_generation_boundary_closure.py diagnose --attempt 1 --clean
.venv/bin/python tools/phase100_qwen3_generation_boundary_closure.py diagnose --attempt 2 --clean
.venv/bin/python tools/phase100_qwen3_generation_boundary_closure.py generate --clean
.venv/bin/python tools/phase100_qwen3_generation_boundary_closure.py decide
.venv/bin/python tools/phase101_failure_targeted_sft.py prepare --clean
.venv/bin/python tools/phase101_failure_targeted_sft.py train --steps 1 --clean
.venv/bin/python tools/phase101_failure_targeted_sft.py train --steps 12 --clean
.venv/bin/python tools/phase101_failure_targeted_sft.py train --steps 30 --clean
.venv/bin/python tools/phase101_failure_targeted_sft.py eval --variant base --clean
.venv/bin/python tools/phase101_failure_targeted_sft.py eval --variant sft --clean
.venv/bin/python tools/phase101_failure_targeted_sft.py decide
.venv/bin/python tools/phase102_failure_targeted_dpo.py prepare --clean
.venv/bin/python tools/phase102_failure_targeted_dpo.py train --steps 12 --clean
.venv/bin/python tools/phase102_failure_targeted_dpo.py train --steps 30 --clean
.venv/bin/python tools/phase102_failure_targeted_dpo.py eval --clean
.venv/bin/python tools/phase102_failure_targeted_dpo.py decide
.venv/bin/python tools/phase103_simulated_user_acceptance.py prepare --clean
.venv/bin/python tools/phase103_simulated_user_acceptance.py eval --variant base --clean
.venv/bin/python tools/phase103_simulated_user_acceptance.py eval --variant dpo --clean
.venv/bin/python tools/phase103_simulated_user_acceptance.py decide
.venv/bin/python tools/phase104_finalize_autonomous_loop.py finalize
.venv/bin/python tools/phase104_finalize_autonomous_loop.py validate
```

Do not promote or deploy any Phase101/102 adapter. Phase100 runtime boundary is the retained product path.
