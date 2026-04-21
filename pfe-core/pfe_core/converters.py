"""Conversion helpers between dataclasses and Pydantic schemas."""

from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from enum import Enum
from typing import Any, TypeVar, get_type_hints

from pydantic import BaseModel

from .config import (
    CuratorConfig,
    CuratorConfigSchema,
    DistillationConfig,
    DistillationConfigSchema,
    EvaluationConfig,
    EvaluationConfigSchema,
    JudgeConfig,
    JudgeConfigSchema,
    LoggingConfig,
    LoggingConfigSchema,
    ModelConfig,
    ModelConfigSchema,
    PFEConfig,
    PFEConfigSchema,
    PrivacyConfig,
    PrivacyConfigSchema,
    RouterConfig,
    RouterConfigSchema,
    SecurityConfig,
    SecurityConfigSchema,
    ServerConfig,
    ServerConfigSchema,
    StorageConfig,
    StorageConfigSchema,
    TrainerConfig,
    TrainerConfigSchema,
    TrainerTriggerConfig,
    TrainerTriggerConfigSchema,
)
from .models import (
    AdapterMeta,
    AdapterMetaSchema,
    EvalDetail,
    EvalDetailSchema,
    EvalReport,
    EvalReportSchema,
    InteractionEvent,
    InteractionEventSchema,
    RawSignal,
    RawSignalSchema,
    SignalQuality,
    SignalQualitySchema,
    TrainingSample,
    TrainingSampleSchema,
)

T = TypeVar("T")

_DATACLASS_TO_SCHEMA: dict[type[Any], type[BaseModel]] = {
    InteractionEvent: InteractionEventSchema,
    RawSignal: RawSignalSchema,
    SignalQuality: SignalQualitySchema,
    TrainingSample: TrainingSampleSchema,
    EvalDetail: EvalDetailSchema,
    EvalReport: EvalReportSchema,
    AdapterMeta: AdapterMetaSchema,
    ModelConfig: ModelConfigSchema,
    TrainerTriggerConfig: TrainerTriggerConfigSchema,
    TrainerConfig: TrainerConfigSchema,
    CuratorConfig: CuratorConfigSchema,
    DistillationConfig: DistillationConfigSchema,
    JudgeConfig: JudgeConfigSchema,
    EvaluationConfig: EvaluationConfigSchema,
    ServerConfig: ServerConfigSchema,
    RouterConfig: RouterConfigSchema,
    PrivacyConfig: PrivacyConfigSchema,
    SecurityConfig: SecurityConfigSchema,
    StorageConfig: StorageConfigSchema,
    LoggingConfig: LoggingConfigSchema,
    PFEConfig: PFEConfigSchema,
}

_SCHEMA_TO_DATACLASS: dict[type[Any], type[Any]] = {
    InteractionEventSchema: InteractionEvent,
    RawSignalSchema: RawSignal,
    SignalQualitySchema: SignalQuality,
    TrainingSampleSchema: TrainingSample,
    EvalDetailSchema: EvalDetail,
    EvalReportSchema: EvalReport,
    AdapterMetaSchema: AdapterMeta,
    ModelConfigSchema: ModelConfig,
    TrainerTriggerConfigSchema: TrainerTriggerConfig,
    TrainerConfigSchema: TrainerConfig,
    CuratorConfigSchema: CuratorConfig,
    DistillationConfigSchema: DistillationConfig,
    JudgeConfigSchema: JudgeConfig,
    EvaluationConfigSchema: EvaluationConfig,
    ServerConfigSchema: ServerConfig,
    RouterConfigSchema: RouterConfig,
    PrivacyConfigSchema: PrivacyConfig,
    SecurityConfigSchema: SecurityConfig,
    StorageConfigSchema: StorageConfig,
    LoggingConfigSchema: LoggingConfig,
    PFEConfigSchema: PFEConfig,
}


def to_pydantic(obj: Any, schema_cls: type[BaseModel] | None = None) -> BaseModel:
    """Convert a dataclass instance or mapping into its Pydantic schema."""

    if isinstance(obj, BaseModel):
        return obj

    if schema_cls is None:
        schema_cls = _DATACLASS_TO_SCHEMA.get(type(obj))
        if schema_cls is None:
            raise TypeError(f"No schema registered for {type(obj)!r}")

    payload = asdict(obj) if is_dataclass(obj) else obj
    return schema_cls.model_validate(payload)


def to_dataclass(obj: Any, dataclass_cls: type[T] | None = None) -> T:
    """Convert a Pydantic model or mapping into its dataclass counterpart."""

    if dataclass_cls is None:
        dataclass_cls = _SCHEMA_TO_DATACLASS.get(type(obj))
        if dataclass_cls is None:
            raise TypeError(f"No dataclass registered for {type(obj)!r}")

    if isinstance(obj, BaseModel):
        payload = obj.model_dump(mode="python")
    else:
        payload = obj

    return _build_dataclass(dataclass_cls, payload)


def _build_dataclass(cls: type[T], payload: Any) -> T:
    if not is_dataclass(cls):
        return payload  # type: ignore[return-value]
    if not isinstance(payload, dict):
        return payload  # type: ignore[return-value]

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_info in fields(cls):
        if field_info.name not in payload:
            continue
        kwargs[field_info.name] = _coerce_value(type_hints.get(field_info.name, field_info.type), payload[field_info.name])
    return cls(**kwargs)


def _coerce_value(annotation: Any, value: Any) -> Any:
    from dataclasses import is_dataclass
    from typing import get_args, get_origin

    origin = get_origin(annotation)
    args = get_args(annotation)

    if value is None:
        return None

    if origin is list:
        item_type = args[0] if args else Any
        return [_coerce_value(item_type, item) for item in value]
    if origin is dict:
        value_type = args[1] if len(args) > 1 else Any
        return {key: _coerce_value(value_type, item) for key, item in value.items()}
    if origin is tuple:
        item_type = args[0] if args else Any
        return tuple(_coerce_value(item_type, item) for item in value)
    if origin is not None and type(None) in args:
        non_none = next((arg for arg in args if arg is not type(None)), Any)
        return _coerce_value(non_none, value)

    if isinstance(annotation, type):
        if is_dataclass(annotation):
            return _build_dataclass(annotation, value)
        if issubclass(annotation, Enum):
            return annotation(value)
        if issubclass(annotation, BaseModel):
            return annotation.model_validate(value)

    return value
