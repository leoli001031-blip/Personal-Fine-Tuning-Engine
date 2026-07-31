"""Phase 3 signal-driven finetuning loop primitives.

The module keeps the first Phase 3 cut intentionally small: a generic
persona/scenario schema, a persisted signal inbox, conservative routing rules,
and a candidate adapter plan that can be shown in Studio before real training
is wired in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal, Mapping
from uuid import uuid4

from .data_policy import (
    HIGH_RISK_PII_TYPES,
    audit_pii_exposure,
    route_signal_for_training,
    sanitize_for_training,
)
from .storage import resolve_home


SignalType = Literal["accept", "reject", "edit", "preference", "correction", "safety_block"]
RouteLane = Literal["memory", "profile", "training_candidate", "discard", "review"]


TRAINABLE_SIGNAL_TYPES = {"accept", "edit", "correction"}
PROFILE_FIRST_SIGNAL_TYPES = {"preference"}
BLOCKED_SIGNAL_TYPES = {"safety_block"}
HIGH_RISK_FLAGS = {
    "legal_advice",
    "medical_advice",
    "financial_advice",
    "diagnosis",
    "prescription",
    "treatment_plan",
    "binding_legal_opinion",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _signal_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


@dataclass(frozen=True)
class PersonaSpec:
    persona_id: str
    name: str
    identity: str
    goals: list[str]
    tone: str
    forbidden_areas: list[str]
    evaluation_criteria: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = {
            "persona_id": self.persona_id,
            "name": self.name,
            "identity": self.identity,
            "tone": self.tone,
        }
        missing = [key for key, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"persona missing required fields: {', '.join(missing)}")
        if not self.goals:
            raise ValueError("persona goals must not be empty")
        if not self.evaluation_criteria:
            raise ValueError("persona evaluation_criteria must not be empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PersonaSpec":
        persona = cls(
            persona_id=str(data.get("persona_id") or data.get("id") or "").strip(),
            name=str(data.get("name") or "").strip(),
            identity=str(data.get("identity") or "").strip(),
            goals=_string_list(data.get("goals")),
            tone=str(data.get("tone") or "").strip(),
            forbidden_areas=_string_list(data.get("forbidden_areas")),
            evaluation_criteria=_string_list(data.get("evaluation_criteria")),
            metadata=_dict(data.get("metadata")),
        )
        persona.validate()
        return persona


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    name: str
    task: str
    input_examples: list[str]
    expected_output: str
    risk_boundaries: list[str]
    human_review_required: bool = False
    high_risk_domains: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        required = {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "task": self.task,
            "expected_output": self.expected_output,
        }
        missing = [key for key, value in required.items() if not str(value).strip()]
        if missing:
            raise ValueError(f"scenario missing required fields: {', '.join(missing)}")
        if not self.input_examples:
            raise ValueError("scenario input_examples must not be empty")
        if not self.risk_boundaries:
            raise ValueError("scenario risk_boundaries must not be empty")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioSpec":
        scenario = cls(
            scenario_id=str(data.get("scenario_id") or data.get("id") or "").strip(),
            name=str(data.get("name") or "").strip(),
            task=str(data.get("task") or data.get("description") or "").strip(),
            input_examples=_string_list(data.get("input_examples") or data.get("examples")),
            expected_output=str(data.get("expected_output") or "").strip(),
            risk_boundaries=_string_list(data.get("risk_boundaries")),
            human_review_required=bool(data.get("human_review_required", False)),
            high_risk_domains=_string_list(data.get("high_risk_domains")),
            metadata=_dict(data.get("metadata")),
        )
        scenario.validate()
        return scenario


@dataclass(frozen=True)
class SignalRouteDecision:
    lanes: list[RouteLane]
    training_target: str
    eligible_for_training: bool
    requires_human_review: bool = False
    excluded_reason: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalInboxItem:
    signal_id: str
    signal_type: SignalType
    persona_id: str
    scenario_id: str
    user_input: str
    model_output: str = ""
    user_feedback: str = ""
    corrected_output: str = ""
    preference: str = ""
    source: str = "feedback"
    confidence: float = 0.7
    session_id: str = ""
    request_id: str = ""
    created_at: str = field(default_factory=_utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)
    route: SignalRouteDecision | None = None

    @property
    def eligible_for_training(self) -> bool:
        return bool(self.route and self.route.eligible_for_training)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["eligible_for_training"] = self.eligible_for_training
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SignalInboxItem":
        raw_route = _dict(data.get("route"))
        route = SignalRouteDecision(
            lanes=_string_list(raw_route.get("lanes")),  # type: ignore[arg-type]
            training_target=str(raw_route.get("training_target") or "none"),
            eligible_for_training=bool(raw_route.get("eligible_for_training", False)),
            requires_human_review=bool(raw_route.get("requires_human_review", False)),
            excluded_reason=str(raw_route.get("excluded_reason") or ""),
            reason=str(raw_route.get("reason") or ""),
        ) if raw_route else None
        signal_type = str(data.get("signal_type") or "accept").strip().lower()
        if signal_type == "copy":
            signal_type = "accept"
        if signal_type == "regenerate":
            signal_type = "reject"
        if signal_type == "delete":
            signal_type = "reject"
        if signal_type not in {"accept", "reject", "edit", "preference", "correction", "safety_block"}:
            signal_type = "reject"
        item = cls(
            signal_id=str(data.get("signal_id") or data.get("event_id") or _signal_id("sig")),
            signal_type=signal_type,  # type: ignore[arg-type]
            persona_id=str(data.get("persona_id") or DEFAULT_PERSONA.persona_id),
            scenario_id=str(data.get("scenario_id") or data.get("scenario") or DEFAULT_SCENARIO.scenario_id),
            user_input=str(data.get("user_input") or data.get("context") or ""),
            model_output=str(data.get("model_output") or ""),
            user_feedback=str(data.get("user_feedback") or ""),
            corrected_output=str(data.get("corrected_output") or data.get("edited_text") or ""),
            preference=str(data.get("preference") or ""),
            source=str(data.get("source") or "feedback"),
            confidence=float(data.get("confidence", 0.7) or 0.7),
            session_id=str(data.get("session_id") or ""),
            request_id=str(data.get("request_id") or ""),
            created_at=str(data.get("created_at") or _utcnow_iso()),
            metadata=_dict(data.get("metadata")),
            route=route,
        )
        if item.route is not None:
            return item
        return item.with_route(route_signal_item(item))

    def with_route(self, route: SignalRouteDecision) -> "SignalInboxItem":
        return SignalInboxItem(
            signal_id=self.signal_id,
            signal_type=self.signal_type,
            persona_id=self.persona_id,
            scenario_id=self.scenario_id,
            user_input=self.user_input,
            model_output=self.model_output,
            user_feedback=self.user_feedback,
            corrected_output=self.corrected_output,
            preference=self.preference,
            source=self.source,
            confidence=self.confidence,
            session_id=self.session_id,
            request_id=self.request_id,
            created_at=self.created_at,
            metadata=dict(self.metadata),
            route=route,
        )


DEFAULT_PERSONA = PersonaSpec(
    persona_id="ops-analyst",
    name="资料整理型业务分析员",
    identity="需要把交互反馈沉淀成可复用能力的业务使用者",
    goals=[
        "把反复确认过的表达偏好沉淀到 profile",
        "把事实性、可复用的工作习惯沉淀到 memory",
        "把安全、低风险、质量明确的修正样本转成训练候选",
    ],
    tone="克制、清楚、给出需要人工确认的边界",
    forbidden_areas=[
        "不记忆联系方式、证件、健康、财务账户等高风险 PII",
        "不把法律、医学、金融结论写入模型权重",
        "不把单次临时上下文当成长期偏好",
    ],
    evaluation_criteria=[
        "路由理由可解释",
        "训练候选可回溯到原始 signal",
        "高风险输出必须提示人工确认",
    ],
)


DEFAULT_SCENARIO = ScenarioSpec(
    scenario_id="contract-risk-summary",
    name="合同摘要与风险标注",
    task="整理合同片段，输出摘要、风险提示和需人工确认的问题，不提供法律结论。",
    input_examples=[
        "请整理这段合同条款：乙方需在 7 日内完成交付，逾期每日按合同总价 1% 承担违约金。",
        "帮我把竞业限制条款标出需要人工确认的风险点。",
    ],
    expected_output="条款摘要、风险提示、需人工确认项；不判断胜诉率、不替代律师意见。",
    risk_boundaries=[
        "只做资料整理和风险提示",
        "不得输出法律结论、诉讼策略或确定性合规判断",
        "涉及真实合同、个人信息或高金额争议时必须人工复核",
    ],
    human_review_required=True,
    high_risk_domains=["legal"],
    metadata={"demo_vertical": "contract_summary"},
)


def default_personas() -> list[PersonaSpec]:
    return [DEFAULT_PERSONA]


def default_scenarios() -> list[ScenarioSpec]:
    return [DEFAULT_SCENARIO]


def route_signal_item(item: SignalInboxItem) -> SignalRouteDecision:
    reasons: list[str] = []
    metadata = dict(item.metadata)
    risk_flags = {flag.lower() for flag in _string_list(metadata.get("risk_flags"))}
    pii_types = {pii.lower() for pii in _string_list(metadata.get("pii_types"))}
    if pii_types & HIGH_RISK_PII_TYPES:
        return SignalRouteDecision(
            lanes=["discard"],
            training_target="blocked",
            eligible_for_training=False,
            excluded_reason="high_risk_pii",
            reason="metadata includes high-risk PII",
        )

    pii_report = audit_pii_exposure(
        [
            {
                "sample_id": item.signal_id,
                "input": item.user_input,
                "output": item.corrected_output or item.model_output,
                "rejected": item.model_output if item.signal_type in {"edit", "correction", "reject"} else "",
            }
        ]
    )
    if pii_report.severity in {"high", "critical"}:
        return SignalRouteDecision(
            lanes=["discard"],
            training_target="blocked",
            eligible_for_training=False,
            excluded_reason="detected_high_risk_pii",
            reason="PII audit found high-risk text fields",
        )

    if item.signal_type in BLOCKED_SIGNAL_TYPES:
        return SignalRouteDecision(
            lanes=["discard", "review"],
            training_target="blocked",
            eligible_for_training=False,
            requires_human_review=True,
            excluded_reason="safety_block",
            reason="safety block signals are review evidence, not training data",
        )

    if risk_flags & HIGH_RISK_FLAGS:
        return SignalRouteDecision(
            lanes=["review"],
            training_target="blocked",
            eligible_for_training=False,
            requires_human_review=True,
            excluded_reason="high_risk_domain_decision",
            reason="high-risk domain conclusions must stay out of training candidates",
        )

    if item.signal_type in PROFILE_FIRST_SIGNAL_TYPES:
        reinforced = bool(metadata.get("repeated") or metadata.get("confirmed_by_feedback"))
        lanes: list[RouteLane] = ["profile"]
        if reinforced:
            lanes.append("training_candidate")
        return SignalRouteDecision(
            lanes=lanes,
            training_target="sft_candidate" if reinforced else "preference_only",
            eligible_for_training=reinforced,
            reason=(
                "reinforced preference can seed a style training candidate"
                if reinforced
                else "single preference updates profile before training"
            ),
        )

    if item.signal_type == "reject":
        return SignalRouteDecision(
            lanes=["review"],
            training_target="dpo_rejected_only",
            eligible_for_training=False,
            excluded_reason="requires_positive_pair",
            reason="reject is negative evidence until paired with a chosen output",
        )

    policy_payload = {
        "signal_type": "edit" if item.signal_type == "correction" else item.signal_type,
        "event_type": "edit" if item.signal_type == "correction" else item.signal_type,
        "confidence": item.confidence,
        "context": item.user_input,
        "model_output": item.model_output,
        "user_action": {
            "type": "edit" if item.signal_type == "correction" else item.signal_type,
            "edited_text": item.corrected_output,
        },
        "source_event_ids": metadata.get("source_event_ids") or [],
        "session_id": item.session_id,
        "request_id": item.request_id,
        "metadata": metadata,
    }
    decision = route_signal_for_training(policy_payload)
    if decision.eligible:
        lanes = ["training_candidate"]
        if item.signal_type == "correction":
            lanes.insert(0, "memory")
            reasons.append("correction can update reusable task memory")
        return SignalRouteDecision(
            lanes=lanes,  # type: ignore[arg-type]
            training_target=decision.primary_target,
            eligible_for_training=True,
            reason=decision.reason,
        )

    return SignalRouteDecision(
        lanes=["review"],
        training_target=decision.primary_target,
        eligible_for_training=False,
        excluded_reason=decision.reason or "not_trainable",
        reason=decision.reason or "signal requires more context before training",
    )


def signal_item_from_feedback(
    *,
    action: str,
    persona_id: str = DEFAULT_PERSONA.persona_id,
    scenario_id: str = DEFAULT_SCENARIO.scenario_id,
    user_input: str,
    model_output: str,
    edited_text: str | None = None,
    user_feedback: str = "",
    source: str = "feedback",
    confidence: float = 0.7,
    session_id: str = "",
    request_id: str = "",
    metadata: Mapping[str, Any] | None = None,
    signal_id: str | None = None,
) -> SignalInboxItem:
    normalized = action.strip().lower()
    if normalized == "copy":
        normalized = "accept"
    if normalized in {"delete", "regenerate"}:
        normalized = "reject"
    if normalized == "edit" and edited_text:
        signal_type: SignalType = "correction"
    elif normalized in {"accept", "reject", "edit", "preference", "correction", "safety_block"}:
        signal_type = normalized  # type: ignore[assignment]
    else:
        signal_type = "reject"
    item = SignalInboxItem(
        signal_id=signal_id or _signal_id("sig"),
        signal_type=signal_type,
        persona_id=persona_id,
        scenario_id=scenario_id,
        user_input=user_input,
        model_output=model_output,
        user_feedback=user_feedback,
        corrected_output=edited_text or "",
        preference=user_feedback if signal_type == "preference" else "",
        source=source,
        confidence=confidence,
        session_id=session_id,
        request_id=request_id,
        metadata=dict(metadata or {}),
    )
    return item.with_route(route_signal_item(item))


def _sample_output(item: SignalInboxItem) -> str:
    if item.signal_type in {"edit", "correction"} and item.corrected_output:
        return item.corrected_output
    if item.signal_type == "preference" and item.preference:
        return item.preference
    return item.model_output


def training_sample_from_signal(item: SignalInboxItem) -> dict[str, Any]:
    output = sanitize_for_training(_sample_output(item))
    instruction = (
        "Follow the selected persona and scenario. Preserve safety boundaries and mark items that need human confirmation."
    )
    sample: dict[str, Any] = {
        "sample_id": f"phase3-sample-{item.signal_id}",
        "source_signal_id": item.signal_id,
        "sample_type": "sft",
        "persona_id": item.persona_id,
        "scenario_id": item.scenario_id,
        "instruction": instruction,
        "input": sanitize_for_training(item.user_input),
        "output": output,
        "metadata": {
            "signal_type": item.signal_type,
            "route": item.route.to_dict() if item.route else {},
            "source": item.source,
        },
    }
    if item.model_output and item.signal_type in {"edit", "correction"}:
        sample["rejected"] = sanitize_for_training(item.model_output)
    return sample


class Phase3SignalLoopStore:
    def __init__(self, *, home: str | Path | None = None, workspace: str = "user_default") -> None:
        self.home = Path(home) if home is not None else resolve_home()
        self.workspace = workspace or "user_default"
        self.root = self.home / "phase3" / "workspaces" / self.workspace
        self.root.mkdir(parents=True, exist_ok=True)
        self.signals_path = self.root / "signals.json"
        self.plans_path = self.root / "candidate_plans.json"

    def personas(self) -> list[dict[str, Any]]:
        return [persona.to_dict() for persona in default_personas()]

    def scenarios(self) -> list[dict[str, Any]]:
        return [scenario.to_dict() for scenario in default_scenarios()]

    def _read_list(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [dict(item) for item in data if isinstance(item, dict)] if isinstance(data, list) else []

    def _write_list(self, path: Path, items: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def add_signal(self, item: SignalInboxItem) -> dict[str, Any]:
        item = item if item.route else item.with_route(route_signal_item(item))
        records = self._read_list(self.signals_path)
        records.append(item.to_dict())
        self._write_list(self.signals_path, records)
        return item.to_dict()

    def ingest_feedback(self, **kwargs: Any) -> dict[str, Any]:
        return self.add_signal(signal_item_from_feedback(**kwargs))

    def list_signals(
        self,
        *,
        signal_type: str | None = None,
        eligible_for_training: bool | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        items = [SignalInboxItem.from_dict(record).to_dict() for record in self._read_list(self.signals_path)]
        filtered: list[dict[str, Any]] = []
        for item in reversed(items):
            if signal_type and item.get("signal_type") != signal_type:
                continue
            if eligible_for_training is not None and bool(item.get("eligible_for_training")) is not eligible_for_training:
                continue
            filtered.append(item)
            if len(filtered) >= limit:
                break
        return filtered

    def candidate_signals(self, *, limit: int = 20) -> list[SignalInboxItem]:
        candidates = []
        for record in self.list_signals(eligible_for_training=True, limit=limit):
            candidates.append(SignalInboxItem.from_dict(record))
        return candidates

    def build_candidate_plan(
        self,
        *,
        persona_id: str = DEFAULT_PERSONA.persona_id,
        scenario_id: str = DEFAULT_SCENARIO.scenario_id,
        limit: int = 12,
    ) -> dict[str, Any]:
        candidates = [
            item
            for item in self.candidate_signals(limit=limit)
            if item.persona_id == persona_id and item.scenario_id == scenario_id
        ]
        samples = [training_sample_from_signal(item) for item in candidates]
        pii_report = audit_pii_exposure(samples).to_dict()
        blocked_by: list[str] = []
        if not samples:
            blocked_by.append("no_training_candidates")
        if pii_report.get("severity") in {"high", "critical"}:
            blocked_by.append("pii_audit_blocked")
            samples = []

        plan = {
            "kind": "phase3_candidate_training_plan",
            "plan_id": _signal_id("plan"),
            "workspace": self.workspace,
            "persona_id": persona_id,
            "scenario_id": scenario_id,
            "candidate_adapter": {
                "version": f"phase3-candidate-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
                "state": "planned" if samples else "blocked",
                "training_method": "sft",
                "sample_count": len(samples),
                "real_training_required": False,
            },
            "samples": samples,
            "sample_count": len(samples),
            "blocked_by": blocked_by,
            "pii_audit": pii_report,
            "eval_gate": {
                "required": True,
                "suites": ["memory_golden", "ordinary_chat", "refusal", "contract_risk_summary"],
                "promotion_requires": ["eval_passed", "manual_review_for_high_risk_scenario"],
                "current_state": "ready_for_eval" if samples else "blocked",
            },
            "handoff": {
                "training_endpoint": "/pfe/training/jobs",
                "eval_endpoint": "/pfe/eval",
                "promote_endpoint": "/pfe/candidate/promote",
                "archive_endpoint": "/pfe/candidate/archive",
            },
            "notes": [
                "Phase3 plan builds small SFT candidates from eligible inbox signals.",
                "The demo scenario only summarizes and flags risk; human confirmation is required.",
            ],
            "created_at": _utcnow_iso(),
        }
        plans = self._read_list(self.plans_path)
        plans.append(plan)
        self._write_list(self.plans_path, plans[-20:])
        return plan

    def summary(self) -> dict[str, Any]:
        signals = self.list_signals(limit=200)
        route_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        eligible_count = 0
        for item in signals:
            signal_type = str(item.get("signal_type") or "unknown")
            type_counts[signal_type] = type_counts.get(signal_type, 0) + 1
            if item.get("eligible_for_training"):
                eligible_count += 1
            route = _dict(item.get("route"))
            for lane in _string_list(route.get("lanes")):
                route_counts[lane] = route_counts.get(lane, 0) + 1
        latest_plan = self._read_list(self.plans_path)[-1:] or []
        return {
            "kind": "phase3_signal_loop",
            "workspace": self.workspace,
            "personas": self.personas(),
            "scenarios": self.scenarios(),
            "signals": signals[:20],
            "signal_count": len(signals),
            "eligible_training_count": eligible_count,
            "type_counts": type_counts,
            "route_counts": route_counts,
            "latest_plan": latest_plan[0] if latest_plan else None,
            "safety": {
                "high_risk_pii_excluded": True,
                "high_risk_domain_conclusions_excluded": True,
                "human_review_required_scenarios": [DEFAULT_SCENARIO.scenario_id],
            },
        }


__all__ = [
    "DEFAULT_PERSONA",
    "DEFAULT_SCENARIO",
    "Phase3SignalLoopStore",
    "PersonaSpec",
    "ScenarioSpec",
    "SignalInboxItem",
    "SignalRouteDecision",
    "default_personas",
    "default_scenarios",
    "route_signal_item",
    "signal_item_from_feedback",
    "training_sample_from_signal",
]
