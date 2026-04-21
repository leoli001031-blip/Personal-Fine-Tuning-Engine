# Personal Finetune Engine (PFE)

English | [简体中文](README.zh-CN.md)

Personal Finetune Engine is a local-first engine for turning user feedback and behavior into a continuous small-model personalization loop.

```text
collect -> curate -> train -> eval -> promote -> serve
```

[Quick Start](#quick-start) • [Core CLI Workflow](#core-cli-workflow) • [Screenshots](#screenshots) • [Platform Support](#platform-support) • [Docs](#docs)

PFE is best understood as operator infrastructure rather than a turnkey consumer app. The main surface is the `pfe` CLI, with local HTTP and browser companions for serving and observability.

## What PFE Covers

- Local readiness checks, diagnostics, and operator views
- Signal collection, curation, and data controls
- SFT and DPO training paths
- Evaluation, candidate handling, promotion, and archive workflows
- Queue, trigger, daemon, and recovery controls
- OpenAI-compatible local serving plus dashboard and chat surfaces

## Quick Start

Bootstrap a local environment:

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

Start local serving:

```bash
pfe serve --port 8921 --live
```

Open observability:

```bash
pfe dashboard
```

Default local pages:

```text
http://127.0.0.1:8921/dashboard
http://127.0.0.1:8921/
```

Notes:

- `pfe serve --port 8921` without `--live` shows the serve plan only.
- `127.0.0.1:8921` is the default local bind, not a fixed requirement.
- If no promoted adapter is available, serving can stay in safe or mock mode.
- Real local model loading is gated behind explicit runtime configuration such as `--real-local`.

## Core CLI Workflow

Typical operator path:

```bash
# 1. Verify the local runtime
pfe doctor

# 2. Inspect current engine state
pfe status --json

# 3. Open the live terminal surface
pfe console --cycles 1

# 4. Start local serving
pfe serve --port 8921 --live

# 5. Open observability
pfe dashboard
```

Command families:

- Inspect: `pfe doctor`, `pfe status --json`, `pfe console`
- Train and evaluate: `pfe train`, `pfe dpo`, `pfe eval`
- Lifecycle: `pfe adapter`, `pfe candidate`
- Automation: `pfe trigger`, `pfe daemon`, `pfe eval-trigger`
- Collection and data: `pfe collect`, `pfe data`

When your workspace, base model, and adapter flow are configured, continue with:

```bash
pfe train --help
pfe dpo --help
pfe eval --help
pfe adapter --help
pfe trigger --help
```

## Screenshots

Real visuals from a local run in this repository.

CLI surfaces generated from real `pfe --help` and `pfe doctor` output:

<p align="center">
  <img src="docs/assets/screenshots/cli-surfaces.png" alt="PFE CLI surfaces" width="1100">
</p>

Dashboard at `/dashboard` after `pfe serve --port 8921 --live`:

<p align="center">
  <img src="docs/assets/screenshots/dashboard.png" alt="PFE observability dashboard" width="1100">
</p>

The browser dashboard is a companion surface. The main control plane remains the CLI.

## Platform Support

- Best-supported local path today: `macOS` on Apple Silicon
- Also supported in the codebase: `Linux/CUDA` and `CPU-only` fallback paths
- `Windows` is not currently documented as a primary target

## Default Network Settings

- Default host: `127.0.0.1`
- Default port: `8921`
- Both can be overridden, for example: `pfe serve --host 127.0.0.1 --port 3000 --live`

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
- [docs/README.md](docs/README.md)
- [ENGINE_DEV_DOC.md](ENGINE_DEV_DOC.md)
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
