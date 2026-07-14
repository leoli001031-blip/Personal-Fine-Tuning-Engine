# Phase68 Runbook

Use the isolated Ollama endpoint containing both frozen judge models:

```bash
.venv/bin/python tools/phase68_prepare.py --clean-evidence
.venv/bin/python tools/phase68_execute.py --stage preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase68_execute.py --stage calibration --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase68_execute.py --stage holdout --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase68_execute.py --stage phase55_regression --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase68_finalize_evidence.py
.venv/bin/python tools/phase68_validate.py
```

Do not edit the candidate rule, prompts, fixtures, protocol, or gates after prepare. Phase55 is label-contract regression only; fresh Phase68 splits retain the full typed and candidate-exact gates.
