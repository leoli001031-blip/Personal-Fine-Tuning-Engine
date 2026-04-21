# Personal Finetune Engine (PFE)

Personal Finetune Engine is a local-first framework for turning user behavior into continuous small-model personalization.

Its core closed loop is:

```text
collect -> curate -> train -> eval -> promote -> serve
```

PFE focuses on the engine layer rather than a single end-user app. The repository includes the core runtime, CLI, server surface, tests, and design docs needed to run and extend the personalization loop locally.

## Status

As of 2026-04-21:

- Phase 1 complete
- Phase 2 complete
- Phase 2 closeout validated with unit, surface, integration, and end-to-end coverage
- Repository is in a good state for publishing, with local model weights and generated artifacts intentionally excluded

See the closeout note in [docs/11-phase2-closeout-2026-04-21.md](docs/11-phase2-closeout-2026-04-21.md).

## License

MIT. See [LICENSE](LICENSE).

## What PFE Includes

- Signal collection and curation
- SFT and DPO training paths
- Adapter lifecycle management
- Evaluation, promotion, rollback, and status surfaces
- Local inference serving with OpenAI-compatible chat endpoints
- Reliability, auditing, and observability layers

## Repository Layout

```text
pfe-core/    Core engine and training pipeline
pfe-cli/     CLI entrypoints and console workflows
pfe-server/  FastAPI server and HTTP surfaces
tests/       Unit, surface, integration, and e2e coverage
docs/        Architecture, roadmap, handoff, and closeout docs
examples/    Example assets and scenarios
tools/       Small repository-local helper scripts
```

## Quick Start

```bash
tools/bootstrap_py311_env.sh
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Common commands:

```bash
pfe doctor
pfe status --json
pfe serve --port 8921
```

Representative test commands:

```bash
env PFE_DISABLE_AUTO_LOCAL_BASE_MODEL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/pytest tests/test_server_adapters.py tests/test_conflict_detector.py -q
env PFE_DISABLE_AUTO_LOCAL_BASE_MODEL=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./.venv/bin/pytest tests -q -m "not integration and not e2e" --tb=short
```

## Documentation

- [ENGINE_DEV_DOC.md](ENGINE_DEV_DOC.md)
- [docs/01-overview.md](docs/01-overview.md)
- [docs/02-architecture.md](docs/02-architecture.md)
- [docs/07-development-roadmap-v2.md](docs/07-development-roadmap-v2.md)
- [docs/11-phase2-closeout-2026-04-21.md](docs/11-phase2-closeout-2026-04-21.md)

## Repository Hygiene

This repository does not include:

- local model weights
- training outputs
- virtual environments
- package caches
- vendored `llama.cpp` checkout and build artifacts

Those assets are environment-specific and should stay outside the published source repository.
