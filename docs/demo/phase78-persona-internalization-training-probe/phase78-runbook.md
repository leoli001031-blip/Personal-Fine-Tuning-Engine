# Phase78 Runbook

```bash
.venv/bin/python tools/phase78_persona_internalization_training.py prepare --clean
.venv/bin/python tools/phase78_persona_internalization_training.py train --steps 12 --clean
.venv/bin/python tools/phase78_persona_internalization_training.py full-regression
.venv/bin/python tools/phase78_persona_internalization_training.py finalize-blocked
.venv/bin/python tools/phase78_persona_internalization_training.py validate
```

The real training command was attempted twice. The current process reported no MPS device; neither CPU attempt created an adapter artifact. Failure evidence is retained. The unused generation and judge commands remain available only for a future run that first creates a valid adapter.
