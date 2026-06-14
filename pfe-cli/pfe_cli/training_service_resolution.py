"""Training service resolution shared by training CLI commands."""

from __future__ import annotations

from typing import Any

from .training_command_deps import TrainingCommandDeps


def training_service(deps: TrainingCommandDeps) -> Any | None:
    return deps.load_service("pfe_core.trainer", "pfe_core.pipeline", "pfe_core.services.pipeline")


__all__ = ["training_service"]
