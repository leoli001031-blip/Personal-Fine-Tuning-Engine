# Phase74 Runbook

```bash
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py prepare --clean-evidence
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py generate --clean
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py prepare-product
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py eval-product --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py finalize
.venv/bin/python tools/phase74_shared_raw_deterministic_serializer_ab.py validate
```

The two arms must be derived from the same shared raw transcript. Do not regenerate one arm independently.
