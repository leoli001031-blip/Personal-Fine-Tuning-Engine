# Phase107 Runbook

```bash
.venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py prepare --clean
.venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py train --steps 1 --clean
.venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py train --steps 12 --clean
.venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py train --steps 30 --clean
.venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py eval --variant base --clean
.venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py eval --variant phase106_sft --clean
.venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py eval --variant phase107_dpo --clean
.venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py decide
.venv/bin/python tools/phase107_runtime_provenance_and_token_faithful_dpo.py validate
```

All model calls are local and simulated. Never auto-promote this candidate.
