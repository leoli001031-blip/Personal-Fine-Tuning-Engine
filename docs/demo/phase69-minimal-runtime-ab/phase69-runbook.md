# Phase69 Runbook

```bash
.venv/bin/python tools/phase69_prepare.py --clean-evidence
.venv/bin/python tools/phase69_generate.py --variant baseline_runtime --clean
.venv/bin/python tools/phase69_generate.py --variant candidate_boundary_contract --clean
.venv/bin/python tools/phase69_prepare_eval.py
.venv/bin/python tools/phase69_execute_eval.py --ollama-endpoint http://127.0.0.1:11435 --timeout 900
.venv/bin/python tools/phase69_finalize_evidence.py
.venv/bin/python tools/phase69_validate.py
```

Do not edit tasks, the candidate contract, decoding, evaluator sources, or gates after `phase69_prepare.py`. Do not edit blinded outputs after `phase69_prepare_eval.py`. Resume interrupted judge calls only with `--resume`.
