# Phase41 Runbook

Generate deterministic simulated user-review preference evidence from Phase40 blind review items:

```bash
.venv/bin/python tools/phase41_simulated_review_preference_candidates.py --clean-evidence
```

Phase41 uses anonymous Phase40 review payloads to simulate a user acceptance review. It creates preference candidates when at least 12 simulated reviewed preferences pass validation. These candidates remain `simulated_usage`; they are not actual user feedback and do not justify an actual product benefit claim.
