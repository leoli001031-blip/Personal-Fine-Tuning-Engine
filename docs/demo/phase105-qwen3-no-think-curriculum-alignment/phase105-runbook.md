# Phase105 Runbook

```bash
.venv/bin/python tools/phase105_qwen3_curriculum_alignment.py prepare --clean
.venv/bin/python tools/phase105_qwen3_curriculum_alignment.py train --steps 1 --clean
.venv/bin/python tools/phase105_qwen3_curriculum_alignment.py train --steps 12 --clean
.venv/bin/python tools/phase105_qwen3_curriculum_alignment.py train --steps 30 --clean
.venv/bin/python tools/phase105_qwen3_curriculum_alignment.py eval --variant base --clean
.venv/bin/python tools/phase105_qwen3_curriculum_alignment.py eval --variant candidate --clean
.venv/bin/python tools/phase105_qwen3_curriculum_alignment.py decide
.venv/bin/python tools/phase105_qwen3_curriculum_alignment.py validate
```

All data is simulated_usage. Do not promote or deploy automatically.
