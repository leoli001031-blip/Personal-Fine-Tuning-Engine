2026-07-31 start: package Phase110 archive evidence as a GitHub draft PR, not a product promotion.
Goal: push `codex/phase110-task-grounded-sft-dpo-causal-proof`, create a draft PR, and check CI.
Order: local focused tests -> `make test-unit` -> `git diff --check` -> push -> draft PR -> CI checks.
Facts to preserve: `archive_phase110_sft_not_qualified`, `product_gate_qualified=false`, `automatic_promotion_allowed=false`.
Risk: the branch is a large historical evidence chain, so PR wording must not frame it as a small feature or usable model.
Local validation: focused Phase110 tests passed 11/11, `make test-unit` passed 1843/1843, and `git diff --check` was clean.
Remote: branch pushed to origin and draft PR #96 created for CI review.
CI round 1: Fast beta gate failed during collection because `tests/test_phase99_driver_safety.py` imported optional `transformers` at module load time.
Fix: removed the top-level optional dependency and kept the tokenizer boundary contract covered with a lightweight tokenizer double.
CI round 2: Fast beta gate reached unit execution but failed on dependency-light dry-run paths (`torch`/`transformers`) and a Linux temp-root Phase85 safety expectation.
Fix: made dry-run/stopping/boundary helpers dependency-light.
Boundary: a direct edit to `tests/test_phase85_driver_safety.py` fixes Linux temp-root behavior but breaks Phase85 frozen source-hash overlay; do not keep that edit without an allowed evidence rebaseline path.
Local validation after dependency-light fixes: Phase110 focused tests passed 11/11; CI-failure spot checks passed 4/4; `make test-unit` passed 1843/1843 with 30 deselected and 1 warning; `git diff --check` was clean.
Commit: `e0e1689 test: keep CI fast gate dependency-light` pushed to `origin/codex/phase110-task-grounded-sft-dpo-causal-proof`.
CI round 3: Fast beta gate failed after `make test-unit` with only the Linux Phase85 temp-root expectation remaining (`1 failed, 1820 passed, 22 skipped, 30 deselected in 81.40s`). Stopped per 3-round limit and recorded details in `BLOCKED.md`.

2026-08-09 Phase111-112 start from verified HEAD `e0e1689`.
Goal: make CI/evidence portable and turn failures into deterministic, explainable eval cases; do not train.
Order: verify freeze -> fix Linux pytest temp semantics -> import evidence -> build taxonomy/evals -> gates -> draft PR.
Verified: 28 unique claims, 30 unique eval briefs, 15 failure modes, and zero missing evidence paths.
Frozen facts: Phase110 remains archived/unqualified; feedback/model calls/training stay at zero this phase.
Largest risk: fixing Linux by editing the Phase85 frozen test would invalidate its source hash, so that file stays byte-identical.
Phase111 result: reproduced the /tmp failure (1 failed), moved Fast-gate pytest basetemp under github.workspace, then the same test passed; Phase85 hash unchanged.
Phase112 result: 28 claims, 30 eval briefs, 70 unique simulated cases (10/category), 0 duplicate IDs, 0 missing evidence paths, 0 holdout collisions.
Reverse evidence-class check: historical -> authorized_real failed closed for PFE-001; restored ledger validates.
Focused checks: Phase111-112/Phase85 33 passed; Phase110 regression 11 passed.
Local gates: test-unit 1858 passed/30 deselected; surface 162 passed; e2e-mock 13 passed/22 deselected; smoke-beta passed.
The first sandboxed e2e attempt failed only on denied loopback bind; the authorized loopback rerun passed without code changes.
Remote CI round 1: Fast beta passed on commit `5f800c6` in 3m30s; run 31274453714. Strict release gate remained policy-skipped.
