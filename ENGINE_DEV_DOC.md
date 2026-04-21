# Personal Finetune Engine (PFE) — Documentation Guide

PFE is a local-first framework for turning user behavior into continuous small-model personalization.

The main documentation set has been organized into:

- `core docs` for project positioning, architecture, module design, and roadmap
- `guides` for concrete integration and usage workflows
- `reference` for durable supporting material
- `archive` for historical handoff and planning records

## Start Here

- [docs/README.md](docs/README.md)
- [docs/01-overview.md](docs/01-overview.md)
- [docs/02-architecture.md](docs/02-architecture.md)
- [docs/03-module-design.md](docs/03-module-design.md)
- [docs/04-roadmap.md](docs/04-roadmap.md)
- [docs/05-tech-and-risk.md](docs/05-tech-and-risk.md)

## Guides

- [docs/guides/openai-closed-loop-integration.md](docs/guides/openai-closed-loop-integration.md)
- [docs/guides/auto-loop-policy.md](docs/guides/auto-loop-policy.md)
- [docs/guides/chat-collector.md](docs/guides/chat-collector.md)
- [docs/guides/dpo-training.md](docs/guides/dpo-training.md)

## Reference

- [docs/reference/data-layering-strategy.md](docs/reference/data-layering-strategy.md)
- [docs/reference/phase2-closeout.md](docs/reference/phase2-closeout.md)

## Development Setup

```bash
tools/bootstrap_py311_env.sh
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Useful commands:

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
