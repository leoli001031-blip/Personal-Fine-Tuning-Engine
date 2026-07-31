# Phase71 Runbook

```bash
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py prepare --clean-evidence
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py eval --stage sparse_preflight --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py eval --stage phase68_regression --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py generate --variant natural_boundary_contract --clean
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py generate --variant structured_boundary_contract --clean
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py prepare-product
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py eval --stage product --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py finalize
.venv/bin/python tools/phase71_qualified_structured_contract_ab.py validate
```

Do not edit fixtures, contracts, transports, decoding, or gates after prepare.
