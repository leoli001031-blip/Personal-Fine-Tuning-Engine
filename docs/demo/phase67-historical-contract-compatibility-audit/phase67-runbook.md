# Phase67 Runbook

Phase67 is deterministic and makes no model calls:

```bash
.venv/bin/python tools/phase67_prepare.py --clean-evidence
.venv/bin/python tools/phase67_finalize_evidence.py
.venv/bin/python tools/phase67_validate.py
```

Do not relabel individual Phase51-55 rows. Only Phase55 labels may be used as aligned legacy regression under the current three-atom contract; Phase51-54 remain diagnostic-only.
