# PFE Phase2 Product Showcase

更新时间：2026-06-18

This showcase records one clean Phase2 demo rehearsal for the loop described in
[../phase2-demo-runbook.md](../phase2-demo-runbook.md):

```text
Studio -> memory sample -> training -> eval gate -> promote -> base/local API comparison
```

## Demo Run

- Workspace: `phase2_showcase`
- Demo home: `/tmp/pfe-phase2-showcase-home`
- Base model: `/Users/lichenhao/Desktop/PFE/models/Qwen2.5-0.5B-Instruct`
- Memory prompt: `PFE Phase2 Demo 记忆代号是什么？只回答代号。`
- Expected local answer: `DEMO-PHASE2-042`
- Adapter version: `20260618-001`

## Screenshots

| Screenshot | What It Shows |
|---|---|
| [01-studio-overview.png](screenshots/01-studio-overview.png) | Studio loaded with the demo workspace, base model, pending adapter, and API handoff. |
| [02-model-ready.png](screenshots/02-model-ready.png) | The selected 0.5B model and real-local-ready Studio header. |
| [03-training-version-area.png](screenshots/03-training-version-area.png) | Training and version-generation controls. |
| [04-candidate-adapter.png](screenshots/04-candidate-adapter.png) | Candidate adapter `20260618-001` before eval passes. |
| [05-eval-gate.png](screenshots/05-eval-gate.png) | Eval gate passed and `设为当前` is available. |
| [06-promoted-adapter.png](screenshots/06-promoted-adapter.png) | Adapter promoted to latest and loaded by Studio. |
| [07-api-handoff.png](screenshots/07-api-handoff.png) | Web URL, chat API, feedback API, model parameter, and curl handoff. |
| [08-base-vs-local-compare.png](screenshots/08-base-vs-local-compare.png) | Rendered comparison from the saved base/local API JSON responses. |
| [09-simplified-studio.png](screenshots/09-simplified-studio.png) | Simplified Studio layout with workspace, endpoint details, and status details collapsed by default. |
| [10-warm-simple-effective-studio.png](screenshots/10-warm-simple-effective-studio.png) | Warm Workbench version of the simplified Studio: short status beans, one clear work order, and base/local proof visible in the main surface. |

## Evidence

| Evidence | Result |
|---|---|
| [demo-phase2-smoke.txt](evidence/demo-phase2-smoke.txt) | `STUDIO MODEL PATH SMOKE PASSED` and `MEMORY GOLDEN SMOKE PASSED`. |
| [smoke-memory-golden.txt](evidence/smoke-memory-golden.txt) | Core memory golden smoke passed. |
| [demo-memory-golden-report.json](evidence/demo-memory-golden-report.json) | `model=base` did not reveal the golden memory; `model=local` answered it. |
| [demo-init-and-ingest.txt](evidence/demo-init-and-ingest.txt) | Clean workspace initialized and one curated memory sample ingested. |
| [demo-training.txt](evidence/demo-training.txt) | Real PEFT training completed for adapter `20260618-001`. |
| [demo-eval-summary.json](evidence/demo-eval-summary.json) | Eval completed with `recommendation=deploy` and `promotion_allowed=true`. |
| [demo-promote-summary.json](evidence/demo-promote-summary.json) | Adapter `20260618-001` promoted to latest and loaded. |
| [base-response.json](evidence/base-response.json) | `model=base` answered without the demo memory. |
| [local-response.json](evidence/local-response.json) | `model=local` answered `DEMO-PHASE2-042` with adapter metadata. |
| [compare-summary.md](evidence/compare-summary.md) | Human-readable base/local comparison summary. |
| [base-vs-local-summary.json](evidence/base-vs-local-summary.json) | Machine-readable comparison result. |

## Browser Acceptance

- `http://127.0.0.1:8921/studio` loaded successfully.
- `studio.css` and `studio.js` loaded.
- No horizontal overflow was observed.
- Studio displayed the base model, latest adapter, pending adapter state, eval
  gate state, promote action, and API handoff.

## Notes

- The Studio screenshots were captured from the in-app browser against the live
  local server.
- The base/local comparison image is rendered from the saved API response JSON,
  because browser policy blocks opening local `data:` and `file:` comparison
  pages in this environment.
- `videos/` remains unrelated to this showcase package and is not included.
