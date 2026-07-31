# Phase24 Runbook

## Goal

Validate a longer PFE product loop: runtime contract interactions -> explicit feedback provenance -> review queue -> strict routing -> candidate sample specs -> training value decision.

## Commands

```bash
.venv/bin/python tools/phase24_real_signal_review_candidate_value_probe.py --clean-evidence
.venv/bin/python -m pytest tests/test_phase24_real_signal_review_candidate_value.py tests/test_phase24_real_signal_review_surface.py tests/test_phase23_runtime_contract_product_loop.py tests/test_phase23_runtime_contract_loop_surface.py -q
make test-unit test-surface test-e2e-mock smoke-beta
```

## Current Result

- Runtime interactions: 80
- Feedback signals: 80
- Runtime holdout: 100
- Runtime scores: {"citation_hit_rate": 1.0, "explicit_boundary_rate": 1.0, "external_law_reference_rate": 0.0, "extra_text_after_first_block_rate": 0.0, "safety_boundary_rate": 1.0, "structure_hit_rate": 1.0, "think_leak_rate": 0.0, "unsupported_assertions": 0}
- Final recommendation: runtime_contract_primary_training_candidate_archived

## Important Boundary

Phase24 generated real PFE runtime-contract outputs, but feedback is labelled as curated/scripted lab review. It is not represented as actual user feedback, so real product-value adapter training is blocked and archived.
