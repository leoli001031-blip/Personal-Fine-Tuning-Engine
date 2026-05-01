"""Status and guidance segment rendering facade for the Rich console prompt panel."""

from __future__ import annotations

from .console_app_prompt_guidance_segments import append_prompt_guidance_segments
from .console_app_prompt_runtime_segments import append_prompt_runtime_segments
from .console_app_prompt_status_segments import append_prompt_status_segments


__all__ = [
    "append_prompt_guidance_segments",
    "append_prompt_runtime_segments",
    "append_prompt_status_segments",
]
