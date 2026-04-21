"""Adapters that connect core services to the pfe-server contract."""

from __future__ import annotations

from typing import Any

from .adapter_store.store import create_adapter_store
from .collector import ChatCollector, ChatInteraction, CollectorConfig
from .config import PFEConfig
from .inference.engine import InferenceConfig, InferenceEngine
from .pipeline import PipelineService, service as pipeline_service
from .router import ScenarioRouter, RoutingResult, create_router
from .user_memory import get_user_memory_store, UserMemoryStore


class InferenceServiceAdapter:
    def __init__(self, pipeline: PipelineService | None = None):
        self.pipeline = pipeline or pipeline_service
        # Initialize ChatCollector for implicit signal extraction
        config = PFEConfig.load()
        collector_config = config.collector if hasattr(config, 'collector') else CollectorConfig()
        self.collector = ChatCollector(
            workspace="user_default",
            config=collector_config,
            home=str(config.home) if hasattr(config, 'home') else None
        )
        self._pending_interactions: dict[str, ChatInteraction] = {}
        # Initialize user memory store
        self.user_memory: UserMemoryStore = get_user_memory_store(
            str(config.home) if hasattr(config, 'home') else None
        )
        # Initialize scenario router for multi-adapter routing
        self.router: ScenarioRouter = create_router(config=config)
        self._routing_cache: dict[str, RoutingResult] = {}
        self._router_enabled = config.router.enable_scenario_routing if hasattr(config.router, 'enable_scenario_routing') else True

    def _get_adapter_for_request(self, request: Any) -> tuple[str, dict[str, Any] | None]:
        """Determine the appropriate adapter version using scenario routing.

        Returns:
            Tuple of (adapter_version, routing_info)
        """
        # If explicit adapter_version is provided, use it
        if request.adapter_version and request.adapter_version != "latest":
            return request.adapter_version, None

        # If routing is disabled, use latest
        if not self._router_enabled:
            return "latest", None

        # Extract the last user message for routing
        user_message = ""
        if request.messages:
            for msg in reversed(request.messages):
                if hasattr(msg, 'role') and msg.role == 'user':
                    user_message = msg.content or ""
                    break
                if isinstance(msg, dict) and msg.get('role') == 'user':
                    user_message = msg.get('content', '')
                    break

        if not user_message:
            return "latest", None

        # Check cache for identical inputs
        cache_key = user_message.strip().lower()
        if cache_key in self._routing_cache:
            cached_result = self._routing_cache[cache_key]
            return cached_result.adapter_version, {
                "scenario_id": cached_result.scenario_id,
                "confidence": cached_result.confidence,
                "fallback": cached_result.fallback,
                "cached": True,
            }

        # Route to appropriate scenario
        routing_result = self.router.route(user_message)

        # Cache the result
        if len(self._routing_cache) >= 1000:  # Limit cache size
            self._routing_cache.clear()
        self._routing_cache[cache_key] = routing_result

        routing_info = {
            "scenario_id": routing_result.scenario_id,
            "confidence": routing_result.confidence,
            "fallback": routing_result.fallback,
            "reasoning": routing_result.reasoning,
            "cached": False,
        }

        return routing_result.adapter_version, routing_info

    async def generate_chat_completion(self, request: Any) -> Any:
        from pfe_server.models import ChatCompletionResponse

        # Get user profile for memory injection
        user_id = request.session_id or "default_user"
        user_profile_text = self.user_memory.get_profile_for_prompt(user_id)

        # Prepare messages with user memory
        messages = [message.model_dump(mode="json") for message in request.messages]

        # Inject user profile into system message or add as first message
        if user_profile_text:
            # Check if first message is system
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = messages[0].get("content", "") + user_profile_text
            else:
                # Insert as first system message
                messages.insert(0, {
                    "role": "system",
                    "content": f"你是一个 helpful 的AI助手。{user_profile_text}"
                })

        # Determine adapter version using scenario routing
        adapter_version, routing_info = self._get_adapter_for_request(request)

        import os
        workspace = os.environ.get("PFE_WORKSPACE")
        payload = self.pipeline.chat_completion(
            messages=messages,
            model=request.model,
            adapter_version=adapter_version,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            metadata={**(request.metadata or {}), "routing": routing_info} if routing_info else request.metadata,
            request_id=request.request_id,
            session_id=request.session_id,
            workspace=workspace,
        )

        # Get user message and assistant response
        user_message = self._extract_last_user_message(request.messages) if request.messages else ""
        assistant_message = ""
        if payload.get("choices"):
            assistant_message = payload["choices"][0].get("message", {}).get("content", "")

        # Extract user facts from conversation
        if user_message and assistant_message:
            self.user_memory.extract_facts_from_conversation(
                user_id=user_id,
                user_message=user_message,
                assistant_message=assistant_message,
                request_id=request.request_id or payload.get("id", ""),
            )

            # Store interaction for signal extraction
            interaction = ChatInteraction(
                session_id=request.session_id or "",
                request_id=request.request_id or payload.get("id", ""),
                user_message=user_message,
                assistant_message=assistant_message,
                adapter_version=request.adapter_version,
            )
            self._pending_interactions[request.session_id or request.request_id or ""] = interaction

        return ChatCompletionResponse.model_validate(payload)

    def _extract_last_user_message(self, messages: list[Any]) -> str:
        """Extract the last user message from the conversation."""
        for msg in reversed(messages):
            if hasattr(msg, 'role') and msg.role == 'user':
                return msg.content or ""
            if isinstance(msg, dict) and msg.get('role') == 'user':
                return msg.get('content', '')
        return ""

    def on_user_action(
        self,
        session_id: str,
        action: str,
        next_message: str | None = None,
        edited_text: str | None = None,
        user_message: str | None = None,
        assistant_message: str | None = None,
    ) -> list[Any]:
        """Process user action to extract implicit signals.

        Args:
            session_id: The session identifier
            action: The user action (continue, edit, delete, regenerate)
            next_message: The next user message if continuing
            edited_text: The edited text if editing
            user_message: Optional user message for explicit feedback
            assistant_message: Optional assistant message for explicit feedback

        Returns:
            List of extracted signals
        """
        # Try to get interaction from pending interactions
        interaction = self._pending_interactions.pop(session_id, None)

        # If not found and explicit messages provided, create a new interaction
        if not interaction and user_message and assistant_message:
            from uuid import uuid4
            interaction = ChatInteraction(
                session_id=session_id,
                request_id=f"explicit-{uuid4().hex[:8]}",
                user_message=user_message,
                assistant_message=assistant_message,
                adapter_version=None,
            )

        if not interaction:
            return []

        return self.collector.on_interaction(
            interaction=interaction,
            next_user_message=next_message,
            edited_text=edited_text,
            action=action,  # type: ignore[arg-type]
        )

    async def status(self) -> dict[str, Any]:
        snapshot = self.pipeline.status()
        adapter_path = None
        latest_adapter = snapshot.get("latest_adapter")
        workspace = str(
            (latest_adapter or {}).get("workspace")
            or snapshot.get("workspace")
            or "user_default"
        )
        if latest_adapter:
            try:
                adapter_path = create_adapter_store(workspace=workspace).load("latest")
            except Exception:
                adapter_path = None
        engine = InferenceEngine(
            InferenceConfig(
                base_model="local-default",
                adapter_path=adapter_path,
            )
        )
        return {**engine.status(), **snapshot}

    def test_route(self, text: str) -> dict[str, Any]:
        """Test scenario routing for a given input text.

        Args:
            text: Input text to test routing

        Returns:
            Detailed routing test results
        """
        return self.router.test_route(text)

    def list_scenarios(self) -> list[dict[str, Any]]:
        """List all available scenarios."""
        return self.router.list_scenarios()

    def bind_scenario_to_adapter(self, scenario_id: str, adapter_version: str) -> bool:
        """Bind a scenario to a specific adapter version.

        Args:
            scenario_id: The scenario ID to bind
            adapter_version: The adapter version to use

        Returns:
            True if successful, False otherwise
        """
        return self.router.bind_scenario_to_adapter(scenario_id, adapter_version)

    def clear_routing_cache(self) -> None:
        """Clear the routing cache."""
        self._routing_cache.clear()
        self.router.clear_cache()


class PipelineServiceAdapter:
    def __init__(self, pipeline: PipelineService | None = None):
        self.pipeline = pipeline or pipeline_service

    async def ingest_signal(self, request: Any) -> Any:
        import os
        from pfe_server.models import SignalIngestResponse

        payload = request.model_dump(mode="json")
        payload.setdefault("metadata", {})
        workspace = os.environ.get("PFE_WORKSPACE") or payload.get("metadata", {}).get("workspace")
        if workspace:
            payload["metadata"]["workspace"] = workspace
        result = self.pipeline.signal(payload)
        return SignalIngestResponse(
            request_id=request.request_id,
            session_id=request.session_id,
            adapter_version=request.adapter_version,
            signal_id=result["event_id"],
            source_event_id=result.get("source_event_id"),
            source_event_ids=list(result.get("source_event_ids") or []),
            event_chain_complete=bool(result.get("event_chain_complete", False)),
            curated_samples=int(result.get("curated_samples", 0) or 0),
            curated_sample_ids=list(result.get("curated_sample_ids") or []),
            dataset_split_counts=dict(result.get("dataset_split_counts") or {}),
            metadata={
                "curated_samples": result.get("curated_samples", 0),
                "curated_sample_ids": result.get("curated_sample_ids", []),
                "curation_state": result.get("curation_state"),
                "curation_reason": result.get("curation_reason"),
                "curation_detail": result.get("curation_detail"),
                "event_chain_complete": result.get("event_chain_complete"),
                "source_event_id": result.get("source_event_id"),
                "source_event_ids": result.get("source_event_ids", []),
                "auto_train": dict(result.get("auto_train") or {}),
            },
        )

    async def run_distillation(self, request: Any) -> Any:
        from pfe_server.models import DistillRunResponse

        result = self.pipeline.run_distillation(
            teacher_model=request.teacher_model,
            scenario=request.scenario,
            style=request.style,
            num_samples=request.num_samples,
            output=None,
        )
        return DistillRunResponse(
            teacher_model=result["teacher_model"],
            teacher_prompt_version=result["teacher_prompt_version"],
            requested_samples=request.num_samples,
            generated_samples=result["generated_samples"],
            train_samples=result["train_samples"],
            val_samples=result["val_samples"],
            test_samples=result["test_samples"],
            metadata={
                "scenario": result["scenario"],
                "style": result["style"],
                "generation_config": request.generation_config,
                "use_cloud_teacher": request.use_cloud_teacher,
            },
        )

    async def status(self) -> dict[str, Any]:
        snapshot = self.pipeline.status()
        return {
            "backend": "sqlite_local",
            "signals": snapshot.get("signal_count", 0),
            "distill_runs": snapshot.get("sample_counts", {}).get("train", 0),
            **snapshot,
        }

    async def reset_auto_train_trigger(self) -> dict[str, Any]:
        return self.pipeline.reset_auto_train_trigger()

    async def retry_auto_train_trigger(self) -> dict[str, Any]:
        return self.pipeline.retry_auto_train_trigger()

    async def process_next_train_queue(self) -> dict[str, Any]:
        return self.pipeline.process_next_train_queue()

    async def process_train_queue_batch(self, limit: int = 5) -> dict[str, Any]:
        return self.pipeline.process_train_queue_batch(limit=limit)

    async def process_train_queue_until_idle(self, max_iterations: int | None = None) -> dict[str, Any]:
        return self.pipeline.process_train_queue_until_idle(max_iterations=max_iterations)

    async def run_train_queue_worker_loop(
        self,
        max_cycles: int | None = None,
        idle_rounds: int | None = None,
        poll_interval_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self.pipeline.run_train_queue_worker_loop(
            max_cycles=max_cycles,
            idle_rounds=idle_rounds,
            poll_interval_seconds=poll_interval_seconds,
        )

    async def run_train_queue_worker_runner(
        self,
        max_seconds: float | None = None,
        idle_sleep_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self.pipeline.run_train_queue_worker_runner(
            max_seconds=max_seconds,
            idle_sleep_seconds=idle_sleep_seconds,
        )

    async def stop_train_queue_worker_runner(self) -> dict[str, Any]:
        return self.pipeline.stop_train_queue_worker_runner()

    async def train_queue_worker_runner_status(self) -> dict[str, Any]:
        return self.pipeline.train_queue_worker_runner_status()

    async def train_queue_worker_runner_timeline(self, limit: int = 5) -> dict[str, Any]:
        return self.pipeline.train_queue_worker_runner_timeline(limit=limit)

    async def train_queue_worker_runner_history(self, limit: int = 10) -> dict[str, Any]:
        return self.pipeline.train_queue_worker_runner_history(limit=limit)

    async def start_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        return self.pipeline.start_train_queue_daemon(note=note)

    async def stop_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        return self.pipeline.stop_train_queue_daemon(note=note)

    async def train_queue_daemon_status(self) -> dict[str, Any]:
        return self.pipeline.train_queue_daemon_status()

    async def train_queue_daemon_history(self, limit: int = 10) -> dict[str, Any]:
        return self.pipeline.train_queue_daemon_history(limit=limit)

    async def recover_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        return self.pipeline.recover_train_queue_daemon(note=note)

    async def restart_train_queue_daemon(self, note: str | None = None) -> dict[str, Any]:
        return self.pipeline.restart_train_queue_daemon(note=note)

    async def approve_next_train_queue(self, note: str | None = None) -> dict[str, Any]:
        return self.pipeline.approve_next_train_queue(note=note)

    async def reject_next_train_queue(self, note: str | None = None) -> dict[str, Any]:
        return self.pipeline.reject_next_train_queue(note=note)

    async def promote_candidate(self, note: str | None = None) -> dict[str, Any]:
        return self.pipeline.promote_candidate(note=note)

    async def archive_candidate(self, note: str | None = None) -> dict[str, Any]:
        return self.pipeline.archive_candidate(note=note)

    async def rollback_candidate(self, version: str | None = None, note: str | None = None) -> dict[str, Any]:
        return self.pipeline.rollback_candidate(version=version, note=note)

    async def candidate_history(self, limit: int = 10) -> dict[str, Any]:
        return self.pipeline.candidate_history(limit=limit)

    async def candidate_timeline(self, limit: int = 10) -> dict[str, Any]:
        return self.pipeline.candidate_timeline(limit=limit)

    async def train_queue_history(self, job_id: str | None = None, limit: int = 10) -> dict[str, Any]:
        return self.pipeline.train_queue_history(job_id=job_id, limit=limit)

    def train(
        self,
        *,
        method: str = "qlora",
        epochs: int = 3,
        base_model: str | None = None,
        train_type: str = "sft",
        workspace: str | None = None,
    ) -> str:
        return self.pipeline.train(
            method=method,
            epochs=epochs,
            base_model=base_model,
            train_type=train_type,
            workspace=workspace,
        )

    def train_dpo(
        self,
        *,
        method: str = "qlora",
        epochs: int = 3,
        base_model: str | None = None,
        workspace: str | None = None,
        backend_hint: str | None = None,
        base_adapter_path: str | None = None,
        min_confidence: float | None = None,
    ) -> str:
        return self.pipeline.train_dpo(
            method=method,
            epochs=epochs,
            base_model=base_model,
            workspace=workspace,
            backend_hint=backend_hint,
            base_adapter_path=base_adapter_path,
            min_confidence=min_confidence,
        )

    async def evaluate(
        self,
        *,
        method: str = "llm_judge",
        adapter_version: str | None = None,
        eval_type: str | None = None,
        workspace: str | None = None,
    ) -> Any:
        return self.pipeline.evaluate(
            method=method,
            adapter_version=adapter_version,
            eval_type=eval_type,
            workspace=workspace,
        )

    # Reliability/Health check methods

    async def get_health_status(self, workspace: str | None = None) -> dict[str, Any]:
        """Get comprehensive health status for daemon and runner."""
        return self.pipeline.get_health_status(workspace=workspace)

    async def get_heartbeat_status(self, workspace: str | None = None) -> dict[str, Any]:
        """Get detailed heartbeat status for daemon and runner."""
        return self.pipeline.get_heartbeat_status(workspace=workspace)

    async def get_lease_status(self, workspace: str | None = None) -> dict[str, Any]:
        """Get lease status for task execution."""
        return self.pipeline.get_lease_status(workspace=workspace)

    async def check_stale_status(self, takeover: bool = False, workspace: str | None = None) -> dict[str, Any]:
        """Check if daemon or runner is stale and optionally trigger takeover."""
        return self.pipeline.check_stale_status(takeover=takeover, workspace=workspace)

    async def force_recovery(self, workspace: str | None = None, reason: str | None = None) -> dict[str, Any]:
        """Force daemon recovery with reset restart policy."""
        return self.pipeline.force_recovery(workspace=workspace, reason=reason)

    async def get_reliability_alerts(
        self,
        level: str | None = None,
        scope: str | None = None,
        limit: int = 10,
        workspace: str | None = None,
    ) -> dict[str, Any]:
        """Get reliability alerts for monitoring."""
        return self.pipeline.get_reliability_alerts(
            level=level,
            scope=scope,
            limit=limit,
            workspace=workspace,
        )
