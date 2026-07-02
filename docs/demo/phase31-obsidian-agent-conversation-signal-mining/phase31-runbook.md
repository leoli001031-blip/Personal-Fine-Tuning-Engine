# Phase31 Runbook

Phase31 mines the user's Obsidian/AgentMemory conversation archive for historical collaboration signals.

It does not train by default, does not modify the vault, and does not label historical conversations as realtime `actual_user_feedback`.

## Default Smoke

```bash
.venv/bin/python tools/phase31_obsidian_agent_signal_mining.py --clean-evidence
```

## Alternate Vault

```bash
.venv/bin/python tools/phase31_obsidian_agent_signal_mining.py \
  --vault-path [AGENT_MEMORY_VAULT] \
  --max-conversations 80 \
  --clean-evidence
```
