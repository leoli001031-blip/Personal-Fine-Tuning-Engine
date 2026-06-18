# Phase2 Demo Runbook

更新时间：2026-06-18

This runbook turns the Phase2 engineering loop into a live demo path:

```text
Studio -> model path -> memory sample -> training -> auto eval -> promote -> base/local API comparison
```

Phase2 is already on `main`. PR #16 delivered the local adapter eval gate and
memory golden smoke. PR #17 fixed the strict release tokenizer fallback. The
latest manual `main` release gate was green for both Fast beta gate and Strict
release gate.

## Demo Assumptions

- You are on a clean `main` checkout.
- `videos/` is a local, untracked media folder and is not part of this demo
  package.
- `.venv` exists and has the training dependencies installed.
- An unquantized Hugging Face model directory is available for training. The
  default local path is:

```bash
models/Qwen2.5-0.5B-Instruct
```

Do not use the `4bit` or `gguf` model as the PEFT training happy path.

## 1. Sync And Verify The Demo Gate

```bash
git switch main
git pull --ff-only
git status --short --branch
```

Choose the model path for the demo:

```bash
export PFE_GOLDEN_SMOKE_MODEL="$PWD/models/Qwen2.5-0.5B-Instruct"
export PFE_REAL_LOCAL_MODEL="$PFE_GOLDEN_SMOKE_MODEL"
```

Run the demo-level gate:

```bash
make demo-phase2-smoke
```

This runs:

- `tools/studio_model_path_smoke.py`
- `tools/memory_golden_smoke.py --strict`

Expected result:

```text
STUDIO MODEL PATH SMOKE PASSED
MEMORY GOLDEN SMOKE PASSED
```

The memory report is written to:

```bash
/tmp/pfe-phase2-demo-memory-golden.json
```

## 2. Create A Clean Demo Workspace

Use an isolated home so the demo can be reset without touching normal local
workspaces:

```bash
export PFE_DEMO_HOME=/tmp/pfe-phase2-demo-home
export PFE_DEMO_WORKSPACE=phase2_demo
export PFE_DEMO_MODEL="${PFE_GOLDEN_SMOKE_MODEL:-$PWD/models/Qwen2.5-0.5B-Instruct}"
```

For a fresh run only, clear the isolated demo home:

```bash
rm -rf "$PFE_DEMO_HOME"
```

Initialize the workspace:

```bash
PFE_HOME="$PFE_DEMO_HOME" \
.venv/bin/python -m pfe_cli.main init \
  --workspace "$PFE_DEMO_WORKSPACE" \
  --base-model "$PFE_DEMO_MODEL" \
  --home "$PFE_DEMO_HOME"
```

## 3. Import The Memory Sample

The current Studio screen is the live demo cockpit. The memory sample import is
done with the CLI so the demo does not invent a UI that is not shipped yet.

```bash
PFE_HOME="$PFE_DEMO_HOME" \
.venv/bin/python -m pfe_cli.main collect ingest \
  --workspace "$PFE_DEMO_WORKSPACE" \
  --event-id evt-phase2-demo-memory-1 \
  --request-id req-phase2-demo-memory-1 \
  --session-id sess-phase2-demo-memory-1 \
  --source-event-id evt-phase2-demo-chat-1 \
  --user-input "PFE Phase2 Demo 记忆代号是什么？只回答代号。" \
  --model-output "DEMO-PHASE2-042" \
  --action accept \
  --confidence 0.99 \
  --scenario phase2-demo-memory
```

Expected output includes:

```text
Curated Samples: 1
```

## 4. Launch Studio

Start the local server:

```bash
env \
  PFE_HOME="$PFE_DEMO_HOME" \
  PFE_WORKSPACE="$PFE_DEMO_WORKSPACE" \
  PFE_ENABLE_REAL_LOCAL_INFERENCE=1 \
  PYTHONPATH="$PWD/pfe-core:$PWD/pfe-cli:$PWD/pfe-server" \
  .venv/bin/python -m pfe_cli.main serve \
    --host 127.0.0.1 \
    --port 8921 \
    --workspace "$PFE_DEMO_WORKSPACE" \
    --live
```

Open:

```text
http://127.0.0.1:8921/studio
```

The first viewport should show:

- current workspace
- current base model
- API handoff
- adapter overview
- training panel
- version list

## 5. Drive The Live Demo

Use this talk track:

```text
PFE starts from a local base model. It collects one memory signal, trains a
local LoRA adapter, evaluates the adapter, blocks promotion until eval passes,
then serves the promoted adapter through an OpenAI-compatible API.
```

In Studio:

1. Confirm the workspace is `phase2_demo`.
2. Confirm the model path points to `Qwen2.5-0.5B-Instruct`.
3. Click `检查条件`.
4. Click `生成版本` and confirm.
5. Wait for the training job to complete.
6. Watch the new adapter appear as a candidate or pending eval version.
7. Wait for automatic eval to finish, or click `评估` on the version.
8. Click `设为当前` only after the eval gate allows promotion.
9. Confirm `Latest adapter`, `Adapter loaded`, and API handoff are visible.

## 6. Prove Base vs Local

Ask the base model:

```bash
curl -s http://127.0.0.1:8921/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "base",
    "messages": [
      {"role": "user", "content": "PFE Phase2 Demo 记忆代号是什么？只回答代号。"}
    ],
    "temperature": 0,
    "max_tokens": 32,
    "metadata": {"enable_real_local": true}
  }'
```

Ask the promoted local adapter:

```bash
curl -s http://127.0.0.1:8921/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "local",
    "messages": [
      {"role": "user", "content": "PFE Phase2 Demo 记忆代号是什么？只回答代号。"}
    ],
    "temperature": 0,
    "max_tokens": 32,
    "metadata": {"enable_real_local": true}
  }'
```

Expected result:

- `model=base` should not reveal `DEMO-PHASE2-042`.
- `model=local` should answer `DEMO-PHASE2-042`.
- The response metadata should show an adapter-backed local path.

## 7. Browser Acceptance Checklist

Before recording or presenting, check:

- `http://127.0.0.1:8921/studio` loads.
- `studio.css` and `studio.js` load.
- no horizontal page overflow.
- the adapter overview shows base model, latest adapter, pending eval, and
  adapter loaded state.
- the version card shows eval gate state and promote availability.
- the address section shows the web URL, chat API, feedback API, model
  parameter, and copyable handoff.

## 8. Reset

Stop the server with `Ctrl-C`.

To replay from scratch:

```bash
rm -rf "$PFE_DEMO_HOME"
```

Then repeat from section 2.
