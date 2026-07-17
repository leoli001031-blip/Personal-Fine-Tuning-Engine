# Phase106 Runbook

```bash
.venv/bin/python tools/phase106_stratified_curriculum_repair.py prepare --clean
.venv/bin/python tools/phase106_stratified_curriculum_repair.py train --steps 1 --clean
.venv/bin/python tools/phase106_stratified_curriculum_repair.py train --steps 12 --clean
.venv/bin/python tools/phase106_stratified_curriculum_repair.py train --steps 30 --clean
.venv/bin/python tools/phase106_stratified_curriculum_repair.py eval --variant base --clean
.venv/bin/python tools/phase106_stratified_curriculum_repair.py eval --variant candidate --clean
.venv/bin/python tools/phase106_stratified_curriculum_repair.py decide
.venv/bin/python tools/phase106_stratified_curriculum_repair.py validate
```

This is a single-variable sampling repair. Do not auto-promote or deploy.
