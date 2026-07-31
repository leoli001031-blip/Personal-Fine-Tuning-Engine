# Phase33 Runbook

Phase33 simulates real multi-turn Agent usage sessions to evaluate whether the Phase32 personal preference adapter profile behaves better than the base profile in the same scenarios.

All Phase33 sessions and transcripts are `simulated_usage`. They are not actual user feedback and must not enter training as realtime user feedback.

## Default Evidence

```bash
.venv/bin/python tools/phase33_simulated_usage_replay_eval.py --clean-evidence --session-count 64
```

## What This Proves

- Same-session base vs adapter replay comparison.
- Multi-turn behavior: user goal, Agent answer, user correction, continued execution, final acceptance.
- Privacy boundary: no raw Obsidian or AgentMemory private text is committed.
- Decision gate never auto-promotes; the best possible recommendation is `promote_after_manual_review`.

## What This Does Not Prove

- It does not claim actual user feedback was collected.
- It does not replace real online feedback.
- It does not auto-promote the Phase32 adapter.
