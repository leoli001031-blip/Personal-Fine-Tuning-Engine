# Phase111-112 Runbook

Phase111 closes the cross-platform CI/evidence gap. Phase112 establishes a
deterministic, explainable failure-eval contract. Neither phase trains or calls a
model.

## Generate and validate

```bash
.venv/bin/python tools/phase111_112_evidence_failure_eval.py generate --clean
.venv/bin/python tools/phase111_112_evidence_failure_eval.py validate
```

The external FDE pack is read-only. Generated files contain claim/eval metadata,
source hashes, and narrow allowed wording; no private source body is copied.

## Focused checks

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS=--basetemp="$PWD/.pytest-tmp" \
  .venv/bin/python -m pytest -p no:cacheprovider \
  tests/test_phase85_driver_safety.py \
  tests/test_phase111_112_evidence_failure_eval.py \
  tests/test_phase111_112_driver_safety.py -q
```

## Release checks

```bash
make test-unit test-surface test-e2e-mock smoke-beta
git diff --check
```

Do not interpret deterministic fixtures, historical evidence, or a green Fast
beta gate as model improvement or real-user validation.
