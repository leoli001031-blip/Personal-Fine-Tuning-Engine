# Phase47 Runbook

```bash
.venv/bin/python -m py_compile pfe-core/pfe_core/phase47_simulated_user_review.py tools/phase47_simulated_user_review.py tests/test_phase47_simulated_user_review.py
.venv/bin/pytest -q tests/test_phase47_simulated_user_review.py tests/test_phase46_runtime_first_latest_intent.py
.venv/bin/python tools/phase47_simulated_user_review.py --clean-evidence
.venv/bin/python tools/phase47_validate.py
```

Phase47 performs no model generation and no training. It converts the Phase46 candidate pack into explicit accept/edit/reject decisions from a simulated-user perspective while preserving that no actual human review occurred.
