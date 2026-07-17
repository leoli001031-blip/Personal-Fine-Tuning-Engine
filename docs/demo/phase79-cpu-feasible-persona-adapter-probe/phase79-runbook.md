# Phase79 Runbook

```bash
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py prepare --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py train --steps 12 --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py sanity --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py sanity-diagnostic --clean
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py full-regression
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py finalize-sanity-blocked
.venv/bin/python tools/phase79_cpu_feasible_persona_probe.py validate
```

The 12-step probe creates a new Phase79 adapter from the frozen Phase78 privacy-safe curriculum. If sanity fails, the 120-step command is forbidden. The diagnostic compares base, adapter, and same-model Phase77 conditional runtime on the same seven fresh sessions. Training success and benefit remain separate.
