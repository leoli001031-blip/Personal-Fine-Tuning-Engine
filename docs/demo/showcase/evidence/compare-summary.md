# Base vs Local API Comparison

Prompt: `PFE Phase2 Demo 记忆代号是什么？只回答代号。`

Expected memory: `DEMO-PHASE2-042`

| Model | Answer | Adapter version | Adapter requested | Adapter loaded |
|---|---|---|---|---|
| `base` | `记忆代号是“Memory”。` | `None` | `None` | `False` |
| `local` | `DEMO-PHASE2-042` | `20260618-001` | `True` | `True` |

Result:

- base does not reveal memory: `True`
- local answers memory: `True`
