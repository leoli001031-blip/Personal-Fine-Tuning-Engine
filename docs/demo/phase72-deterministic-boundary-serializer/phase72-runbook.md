# Phase72 Runbook

```bash
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py prepare --clean-evidence
.venv/bin/python tools/phase72_deterministic_boundary_serializer.py eval --stage sparse_preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase72_finalize_blocked.py
.venv/bin/python tools/phase72_validate_blocked.py
```

Do not run downstream stages or edit the frozen wire protocol after this failure.
