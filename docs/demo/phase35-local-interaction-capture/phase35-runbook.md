# Phase35 Runbook

Phase35 adds a lightweight local interaction capture lane before any Hermes integration.

## Simulated Evidence

```bash
.venv/bin/python tools/phase35_local_interaction_capture.py --clean-evidence
```

## Local CLI Capture

```bash
pfe phase35 interact \
  --workspace personal-agent \
  --user-goal "帮我整理当前工作区并判断下一步" \
  --feedback-action correction \
  --user-feedback "这次回答还是太泛，先跑真实检查。"
```

To mark a real local interaction as reviewable actual feedback, the operator must explicitly add:

```bash
--operator-id local-user \
--confirm-actual-user-feedback \
--consent-for-training-candidate-review \
--not-scripted-or-curated
```

Even then, Phase35 only stores the record as pending review. Training remains blocked until Phase36 review approves it.
