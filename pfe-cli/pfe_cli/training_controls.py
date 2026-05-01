"""Small helpers for CLI training controls."""

from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Iterator

REAL_TRAINING_ENV = "PFE_REAL_TRAINING"
TRAIN_BACKEND_OPTIONS = frozenset({"auto", "mock_local", "peft", "dpo", "unsloth", "mlx"})


def normalize_train_backend_option(value: str | None) -> str:
    normalized = str(value or "auto").strip().lower().replace("-", "_")
    return normalized or "auto"


def validate_train_backend_option(value: str | None, *, train_type: str) -> str:
    normalized = normalize_train_backend_option(value)
    if normalized not in TRAIN_BACKEND_OPTIONS:
        raise ValueError("Unsupported training backend. Use one of: auto, mock_local, peft, dpo, unsloth, mlx.")
    if normalized == "dpo" and str(train_type or "sft").strip().lower() != "dpo":
        raise ValueError("--backend dpo is only valid with --train-type dpo or the pfe dpo command.")
    return normalized


@contextmanager
def real_training_env(*, real_local: bool) -> Iterator[None]:
    previous = os.environ.get(REAL_TRAINING_ENV)
    if real_local:
        os.environ[REAL_TRAINING_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(REAL_TRAINING_ENV, None)
        else:
            os.environ[REAL_TRAINING_ENV] = previous
