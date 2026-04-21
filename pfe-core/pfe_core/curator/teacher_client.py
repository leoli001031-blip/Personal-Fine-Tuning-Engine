"""Teacher inference client supporting cloud, local, and mock backends."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal, Optional


@dataclass
class TeacherClientConfig:
    """Configuration for teacher inference client."""

    backend: Literal["cloud", "local", "mock"] = "mock"
    model: str = "mock-teacher"
    temperature: float = 0.7
    max_tokens: int = 512
    # Cloud-specific
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    # Local-specific
    local_base_model: Optional[str] = None
    local_adapter_path: Optional[str] = None


class TeacherInferenceClient:
    """Pluggable teacher inference client.

    Supports three backends:
    - cloud: OpenAI-compatible API (requires openai package)
    - local: Reuse InferenceEngine for local model inference
    - mock: Deterministic template responses for tests and fallback
    """

    def __init__(self, config: TeacherClientConfig) -> None:
        self.config = config
        self._cloud_client: Any = None
        self._local_engine: Any = None

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        """Generate a teacher response.

        Returns a dict with:
        - text: generated text
        - backend: which backend was used
        - model: model name
        - usage: token usage estimate (if available)
        """
        if self.config.backend == "cloud":
            return self._generate_cloud(messages, **kwargs)
        if self.config.backend == "local":
            return self._generate_local(messages, **kwargs)
        return self._generate_mock(messages, **kwargs)

    def _generate_cloud(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "openai package is required for cloud teacher backend. "
                "Install it with: pip install openai"
            ) from exc

        if self._cloud_client is None:
            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key and not self.config.base_url:
                raise RuntimeError(
                    "Cloud teacher requires OPENAI_API_KEY environment variable or explicit api_key"
                )
            client_kwargs: dict[str, Any] = {}
            if api_key:
                client_kwargs["api_key"] = api_key
            if self.config.base_url:
                client_kwargs["base_url"] = self.config.base_url
            self._cloud_client = OpenAI(**client_kwargs)

        temperature = kwargs.get("temperature", self.config.temperature)
        max_tokens = kwargs.get("max_tokens", self.config.max_tokens)

        response = self._cloud_client.chat.completions.create(
            model=self.config.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )

        text = response.choices[0].message.content or ""
        usage: dict[str, int] = {}
        if getattr(response, "usage", None):
            usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
            }

        return {
            "text": text,
            "backend": "cloud",
            "model": self.config.model,
            "usage": usage,
        }

    def _generate_local(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        if self._local_engine is None:
            from ..inference.engine import InferenceConfig, InferenceEngine

            base_model = self.config.local_base_model or "Qwen/Qwen2.5-3B-Instruct"
            inference_config = InferenceConfig(
                base_model=base_model,
                adapter_path=self.config.local_adapter_path,
                backend="auto",
                quantization="4bit",
                max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                device="auto",
            )
            self._local_engine = InferenceEngine(inference_config)

        text = self._local_engine.generate(
            messages,
            temperature=kwargs.get("temperature", self.config.temperature),
            max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
        )

        return {
            "text": text,
            "backend": "local",
            "model": self.config.local_base_model or "local",
            "usage": {},
        }

    def _generate_mock(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        # Mock backend returns a deterministic hint based on the last user message
        last_message = messages[-1]["content"] if messages else ""
        text = (
            f"[mock-teacher] 针对问题「{last_message[:32]}…」"
            "给出的回应：先确认情绪，再提供一个具体、可操作的小步骤。"
        )
        return {
            "text": text,
            "backend": "mock",
            "model": self.config.model,
            "usage": {},
        }

    def generate_explanation(
        self,
        prompt: str,
        base_response: str,
        user_feedback: str,
    ) -> str:
        """Generate a teacher explanation for why the user prefers a certain direction.

        The explanation is used as distillation provenance to document the teacher's
        understanding of the user's preference.

        Args:
            prompt: The original user prompt.
            base_response: The model's original response.
            user_feedback: The user's feedback or edited text.

        Returns:
            A textual explanation of the user's preference direction.
        """
        system_msg = (
            "You are a teacher model analyzing user preferences. "
            "Explain briefly why the user prefers their version over the base response."
        )
        user_msg = (
            f"Prompt: {prompt}\n\n"
            f"Base response: {base_response}\n\n"
            f"User feedback: {user_feedback}\n\n"
            "Explain the user's preference direction in one sentence."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        try:
            result = self.generate(messages, temperature=self.config.temperature, max_tokens=256)
            explanation = str(result.get("text", "")).strip()
            if explanation:
                return explanation
        except Exception:
            pass
        # Fallback to local heuristic explanation
        return (
            f"The user prefers a response that better aligns with their feedback: "
            f"'{user_feedback[:80]}...' over the base response."
        )

    def generate_preference_pair(
        self,
        prompt: str,
        rejected: str,
    ) -> dict[str, Any]:
        """Generate a teacher-improved response to form a DPO preference pair.

        Args:
            prompt: The user prompt.
            rejected: The rejected (worse) response.

        Returns:
            A dict with keys:
            - prompt: the original prompt
            - chosen: the teacher-generated better response
            - rejected: the original rejected response
            - explanation: brief rationale from the teacher
            - backend: which backend produced the chosen text
        """
        system_msg = (
            "You are a helpful teacher model. Given a user prompt and a rejected response, "
            "generate a clearly better response that is more helpful, accurate, and aligned."
        )
        user_msg = (
            f"Prompt: {prompt}\n\n"
            f"Rejected response: {rejected}\n\n"
            "Generate a better response."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        try:
            result = self.generate(messages, temperature=self.config.temperature, max_tokens=self.config.max_tokens)
            chosen = str(result.get("text", "")).strip()
            backend = result.get("backend", "mock")
        except Exception:
            chosen = ""
            backend = "fallback"

        if not chosen:
            # Local fallback: produce a generic improved response
            chosen = (
                f"[teacher-improved] A more thoughtful answer to: {prompt[:64]}. "
                "It addresses the core concern with empathy and actionable detail."
            )
            backend = "fallback"

        explanation = (
            f"Teacher ({backend}) generated a preferred response over the rejected one "
            f"because it better addresses the prompt."
        )
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "explanation": explanation,
            "backend": backend,
        }
