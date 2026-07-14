# Phase73 Runbook

```bash
.venv/bin/python tools/phase73_exact_descriptor_normalization.py prepare --clean-evidence
.venv/bin/python tools/phase73_exact_descriptor_normalization.py eval --stage sparse_preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase73_exact_descriptor_normalization.py eval --stage phase68_regression --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase73_exact_descriptor_normalization.py finalize
.venv/bin/python tools/phase73_exact_descriptor_normalization.py validate
```

Do not run the regression unless the fresh preflight is qualified. Do not count historical Phase72 replay rows as new model outputs.
