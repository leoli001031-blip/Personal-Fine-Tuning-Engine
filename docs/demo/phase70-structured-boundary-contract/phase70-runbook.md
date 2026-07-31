# Phase70 Runbook

```bash
.venv/bin/python tools/phase70_prepare.py --clean-evidence
.venv/bin/python tools/phase70_execute_eval.py --stage sparse_preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase70_finalize_evidence.py
.venv/bin/python tools/phase70_validate.py
```

The sparse prerequisite did not qualify. Do not run downstream regression, generation, or product evaluation, and do not edit the frozen fixture or gates after model calls.
