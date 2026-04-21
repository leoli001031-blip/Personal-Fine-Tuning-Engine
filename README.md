# Personal Finetune Engine (PFE)

English | [简体中文](README.zh-CN.md)

Personal Finetune Engine is a local-first engine for turning user feedback and behavior into a continuous small-model personalization loop.

```text
collect -> curate -> train -> eval -> promote -> serve
```

PFE is best understood as infrastructure rather than a turnkey consumer app. The main surface is the `pfe` CLI, with local HTTP and browser companions for serving and observability.

## What PFE Includes

- Local readiness, diagnostics, and operator views
- Signal collection, curation, and data controls
- SFT and DPO training paths
- Evaluation, candidate handling, promotion, and archive workflows
- Queue, trigger, daemon, and recovery controls
- OpenAI-compatible local serving plus dashboard / chat surfaces

## Quick Start

```bash
tools/bootstrap_py311_env.sh
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Recommended first commands:

```bash
pfe doctor
pfe status --json
pfe console --cycles 1
```

Start the local server:

```bash
pfe serve --port 8921 --live
```

Open observability:

```bash
pfe dashboard
```

or visit:

```text
http://127.0.0.1:8921/dashboard
http://127.0.0.1:8921/
```

Notes:

- `pfe serve --port 8921` without `--live` shows the serve plan only.
- If no promoted adapter is available, serving can stay in safe / mock mode.
- Real local model loading is gated behind explicit runtime configuration such as `--real-local`.

## Common CLI Commands

- `pfe doctor`
  Check runtime readiness, backend availability, local model state, and capability blockers.
- `pfe status --json`
  Inspect adapters, signals, trigger policy, queue state, observability, and trainer planning.
- `pfe console`
  Open the Rich-based operator console for a live terminal view of the system.
- `pfe serve`
  Start the local FastAPI / OpenAI-compatible inference surface.
- `pfe dashboard`
  Open the observability dashboard in a browser.
- `pfe train`, `pfe dpo`, `pfe eval`
  Run the core training and evaluation workflows.
- `pfe adapter`, `pfe candidate`
  Manage promotion, archive, and candidate lifecycle operations.
- `pfe trigger`, `pfe daemon`, `pfe eval-trigger`
  Control automation gates, worker loops, and recovery flows.
- `pfe collect`, `pfe data`
  Manage collection and privacy / data compliance surfaces.

## Typical Workflow

```bash
# 1. Check whether the local runtime is actually ready
pfe doctor

# 2. Inspect the current engine state
pfe status --json

# 3. Open the operator console
pfe console --cycles 1

# 4. Start local serving
pfe serve --port 8921 --live

# 5. Open observability
pfe dashboard
```

When your workspace, base model, and adapter flow are configured, continue with:

```bash
pfe train --help
pfe dpo --help
pfe eval --help
pfe adapter --help
pfe trigger --help
```

## Screenshots

Real screenshots from a local run in this repository:

`pfe console --cycles 1`

![PFE CLI console](docs/assets/screenshots/cli-console.png)

`/dashboard` after `pfe serve --port 8921 --live`

![PFE dashboard](docs/assets/screenshots/dashboard.png)

## HTTP And Browser Surfaces

PFE also exposes local HTTP and browser companions:

- `GET /healthz`
- `GET /pfe/status`
- `GET /dashboard`
- `POST /v1/chat/completions`

Bundled browser pages live under:

- `pfe-server/pfe_server/static/dashboard.html`
- `pfe-server/pfe_server/static/chat.html`

## Repository Layout

```text
pfe-core/    Core engine and training pipeline
pfe-cli/     CLI entrypoints and console workflows
pfe-server/  FastAPI server and HTTP surfaces
tests/       Unit, surface, integration, and e2e coverage
docs/        Public docs, guides, references, and archive
examples/    Example assets and scenarios
tools/       Repository-local helper scripts
```

## Project Status

- Phase 1 complete
- Phase 2 complete
- Public repository prepared with large local artifacts intentionally excluded

See [docs/reference/phase2-closeout.md](docs/reference/phase2-closeout.md) for the closeout note.

## Docs

- [README.zh-CN.md](README.zh-CN.md)
- [ENGINE_DEV_DOC.md](ENGINE_DEV_DOC.md)
- [docs/README.md](docs/README.md)
- [docs/01-overview.md](docs/01-overview.md)
- [docs/02-architecture.md](docs/02-architecture.md)
- [docs/04-roadmap.md](docs/04-roadmap.md)
- [docs/reference/phase2-closeout.md](docs/reference/phase2-closeout.md)

## License

MIT. See [LICENSE](LICENSE).

## Repository Boundaries

This repository does not include:

- local model weights
- training outputs
- virtual environments
- package caches
- vendored `llama.cpp` checkout and build artifacts

Those assets are environment-specific and should stay outside the published source repository.
