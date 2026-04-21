"""Adapter lifecycle state and transition helpers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal


class AdapterLifecycleState(str, Enum):
    TRAINING = "training"
    PENDING_EVAL = "pending_eval"
    PROMOTED = "promoted"
    FAILED_EVAL = "failed_eval"
    ARCHIVED = "archived"


class AdapterArtifactFormat(str, Enum):
    PEFT_LORA = "peft_lora"
    MLX_LORA = "mlx_lora"
    GGUF_MERGED = "gguf_merged"


ALLOWED_TRANSITIONS: dict[AdapterLifecycleState, set[AdapterLifecycleState]] = {
    AdapterLifecycleState.TRAINING: {AdapterLifecycleState.PENDING_EVAL},
    AdapterLifecycleState.PENDING_EVAL: {
        AdapterLifecycleState.PROMOTED,
        AdapterLifecycleState.FAILED_EVAL,
        AdapterLifecycleState.ARCHIVED,
    },
    AdapterLifecycleState.PROMOTED: {AdapterLifecycleState.ARCHIVED},
    AdapterLifecycleState.FAILED_EVAL: {
        AdapterLifecycleState.PROMOTED,
        AdapterLifecycleState.ARCHIVED,
    },
    AdapterLifecycleState.ARCHIVED: set(),
}

PROMOTABLE_STATES = frozenset(
    {
        AdapterLifecycleState.PENDING_EVAL,
        AdapterLifecycleState.FAILED_EVAL,
    }
)


@dataclass
class LifecycleDecision:
    current: AdapterLifecycleState
    target: AdapterLifecycleState
    allowed: bool
    reason: str = ""


def normalize_state(value: str | AdapterLifecycleState) -> AdapterLifecycleState:
    return value if isinstance(value, AdapterLifecycleState) else AdapterLifecycleState(value)


def can_transition(current: str | AdapterLifecycleState, target: str | AdapterLifecycleState) -> bool:
    current_state = normalize_state(current)
    target_state = normalize_state(target)
    return target_state in ALLOWED_TRANSITIONS[current_state]


def can_promote(current: str | AdapterLifecycleState) -> bool:
    return normalize_state(current) in PROMOTABLE_STATES


def can_promote_from(current: str | AdapterLifecycleState) -> bool:
    """Return whether promote() is allowed from the given lifecycle state."""

    return can_promote(current)


def llama_cpp_requires_gguf_merged(target_backend: str | None = None) -> bool:
    """llama.cpp only consumes gguf_merged artifacts."""

    if target_backend is None:
        return False
    normalized = target_backend.replace("-", "_").lower()
    return normalized in {"llama_cpp", "llama.cpp"}


def llama_cpp_requires_merged_gguf(target_backend: str | None = None) -> bool:
    """Alias for the merged GGUF contract used by llama.cpp exports."""

    return llama_cpp_requires_gguf_merged(target_backend)


def artifact_role_for_format(artifact_format: str | AdapterArtifactFormat | None) -> str:
    """Return the semantic role for an artifact format."""

    if artifact_format is None:
        return "base_model"
    normalized = artifact_format.value if isinstance(artifact_format, AdapterArtifactFormat) else str(artifact_format)
    normalized = normalized.strip().lower().replace("-", "_")
    if normalized in {AdapterArtifactFormat.PEFT_LORA.value, AdapterArtifactFormat.MLX_LORA.value}:
        return "lora_adapter"
    if normalized == AdapterArtifactFormat.GGUF_MERGED.value:
        return "merged_gguf"
    return "base_model"


def validate_transition(current: str | AdapterLifecycleState, target: str | AdapterLifecycleState) -> None:
    current_state = normalize_state(current)
    target_state = normalize_state(target)
    if not can_transition(current_state, target_state):
        raise ValueError(f"Invalid adapter transition: {current_state.value} -> {target_state.value}")


def transition_decision(
    current: str | AdapterLifecycleState,
    target: str | AdapterLifecycleState,
) -> LifecycleDecision:
    current_state = normalize_state(current)
    target_state = normalize_state(target)
    allowed = can_transition(current_state, target_state)
    reason = "" if allowed else f"{current_state.value} cannot transition to {target_state.value}"
    return LifecycleDecision(current=current_state, target=target_state, allowed=allowed, reason=reason)


def promoted_state() -> AdapterLifecycleState:
    return AdapterLifecycleState.PROMOTED


def pending_eval_state() -> AdapterLifecycleState:
    return AdapterLifecycleState.PENDING_EVAL


def archived_state() -> AdapterLifecycleState:
    return AdapterLifecycleState.ARCHIVED


def rollback_to_version(
    store: Any,
    current_version: str,
    fallback_version: str | None = None,
    reason: str = "forget_detected",
) -> dict[str, Any]:
    """Rollback to a previous adapter version when forgetting is detected.

    This function implements automatic rollback for incremental training when
    forget detection indicates significant knowledge loss. It archives the
    problematic version and promotes the fallback version.

    Args:
        store: AdapterStore instance
        current_version: The version that exhibited forgetting
        fallback_version: Version to rollback to (default: previous promoted)
        reason: Reason for rollback

    Returns:
        Dictionary with rollback operation results

    Raises:
        AdapterError: If rollback cannot be performed
    """
    from ..errors import AdapterError

    result: dict[str, Any] = {
        "success": False,
        "current_version": current_version,
        "fallback_version": fallback_version,
        "reason": reason,
        "actions": [],
    }

    try:
        # Archive the current problematic version
        try:
            archive_result = store.archive(current_version)
            result["actions"].append({"action": "archive", "version": current_version, "result": archive_result})
        except Exception as e:
            # If already archived or cannot archive, continue
            result["actions"].append({"action": "archive", "version": current_version, "error": str(e)})

        # Determine fallback version if not provided
        if fallback_version is None:
            # Try to get the most recent non-archived version
            try:
                rows = store.list_version_records(limit=10)
                for row in rows:
                    row_version = row.get("version")
                    row_state = row.get("state")
                    if row_version and row_version != current_version and row_state in {"promoted", "pending_eval"}:
                        fallback_version = row_version
                        break
            except Exception as e:
                result["fallback_error"] = f"Failed to find fallback: {e}"

        # Promote fallback version if found
        if fallback_version:
            try:
                promote_result = store.promote(fallback_version)
                result["actions"].append({"action": "promote", "version": fallback_version, "result": promote_result})
                result["success"] = True
                result["promoted_version"] = fallback_version
            except Exception as e:
                result["promote_error"] = str(e)
        else:
            result["fallback_error"] = "No suitable fallback version found"

    except Exception as e:
        result["error"] = str(e)
        raise AdapterError(f"Rollback failed: {e}") from e

    return result
