"""Shared exception hierarchy for PFE."""

from __future__ import annotations


class PFEError(Exception):
    """Base error for user-facing PFE failures."""

    code = "E000"

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class TrainingError(PFEError):
    code = "E100"


class AdapterError(PFEError):
    code = "E200"


class EvalError(PFEError):
    code = "E300"


class EvaluationError(EvalError):
    code = "E300"


class InferenceError(PFEError):
    code = "E400"


class ServerError(PFEError):
    code = "E410"


class PipelineError(PFEError):
    code = "E420"


class DataError(PFEError):
    code = "E500"


class ConfigError(PFEError):
    code = "E600"


class AuthError(PFEError):
    code = "E700"
