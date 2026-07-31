# Phase34 Runbook

Phase34 adds a simulated real-user acceptance judge on top of Phase33 replay transcripts.

It does not train a model, does not collect actual user feedback, and does not auto-promote. The judge simulates whether the user would accept, edit, reject, or block a response based on reduced effort, trust, correction recovery, evidence, privacy boundary, and false-completion risk.

## Default Evidence

```bash
.venv/bin/python tools/phase34_simulated_user_acceptance_judge.py --clean-evidence --scenario-count 100
```

## Boundaries

- Every scenario, blind pair, and judgement is `simulated_user_judgement`.
- `actual_user_feedback_count` must remain 0.
- The public blind-pair payload must not expose which variant is base or adapter.
- The best possible recommendation is `promote_after_manual_review`.
