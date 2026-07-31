# Phase85 Low-Fallback Semantic Guard Runbook

Phase85 is a simulated benchmark. It does not use actual user feedback, does not train an adapter,
and cannot prove training benefit. Raw pre-guard model text and private source text are not retained.

## Frozen run

```bash
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real prepare --clean-evidence
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real api-smoke --clean-evidence
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real generate --variant base_api_length_control_160 --clean-evidence
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real generate --variant persona_api_contract_v3_fresh --clean-evidence
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real generate --variant persona_api_contract_v4 --clean-evidence
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real review-template --clean-evidence
```

Review every routed V4 returned output from the temporary review cache. Record only hashes and
findings in `docs/demo/phase85-low-fallback-semantic-guard/manual-semantic-review.json`. Manual review may turn an automated
pass into failure; it may never upgrade a failed deterministic gate.

```bash
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real full-regression
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real finalize
.venv/bin/python tools/phase85_low_fallback_semantic_guard.py --mode real validate
```

## Strict boundaries

- Expected format denominator: 68 routed turns per variant.
- V4 native rate >= 0.75; repair <= 0.25; fallback <= 0.10.
- V4 target score >= 0.80; every target category >= 0.75; gain vs base >= 0.04.
- Independent pre-labeled block recall must be 1.0 and false-block rate 0.0.
- No automatic promote, deployment, Hermes attachment, or product-default change.
