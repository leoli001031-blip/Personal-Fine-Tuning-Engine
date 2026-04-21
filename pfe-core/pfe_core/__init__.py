"""PFE core package."""

from .config import (
    CuratorConfig,
    DistillationConfig,
    EvaluationConfig,
    JudgeConfig,
    LoggingConfig,
    ModelConfig,
    PFEConfig,
    PrivacyConfig,
    RouterConfig,
    SecurityConfig,
    ServerConfig,
    StorageConfig,
    TrainerConfig,
    TrainerTriggerConfig,
)
from .converters import to_dataclass, to_pydantic
from .models import (
    AdapterMeta,
    EvalDetail,
    EvalReport,
    InteractionEvent,
    RawSignal,
    TrainingSample,
)
from .adapter_store.lifecycle import (
    AdapterArtifactFormat,
    AdapterLifecycleState,
    can_promote,
    can_transition,
    validate_transition,
)
from .adapter_store.manifest import (
    ADAPTER_CONFIG_FILENAME,
    ADAPTER_MANIFEST_FILENAME,
    ADAPTER_MODEL_FILENAME,
    AdapterManifest,
)
from .scenarios import (
    ScenarioConfig,
    BUILTIN_SCENARIOS,
    get_builtin_scenario,
    list_builtin_scenarios,
    create_custom_scenario,
)
from .router import (
    SemanticClassifier,
    SemanticClassificationResult,
)
from .user_profile import (
    PreferenceScore,
    UserProfile,
    UserProfileStore,
    get_user_profile_store,
)
from .profile_extractor import (
    ProfileExtractor,
    SignalAnalysisResult,
    extract_profile_for_user,
    get_user_profile,
    analyze_user_conversation,
)
from .pii_detector import (
    PIIDetector,
    PIIDetectionResult,
    PIIFinding,
    PIIType,
    PIISeverity,
)
from .anonymizer import (
    Anonymizer,
    AnonymizationConfig,
    AnonymizationStrategy,
    create_anonymizer,
    anonymize_text,
)
from .pii_audit import (
    PIIAuditLog,
    PIIAuditEntry,
    PIIComplianceReport,
    PIIWhitelist,
)

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "AdapterArtifactFormat",
    "AdapterLifecycleState",
    "AdapterManifest",
    "AdapterMeta",
    "ADAPTER_CONFIG_FILENAME",
    "ADAPTER_MANIFEST_FILENAME",
    "ADAPTER_MODEL_FILENAME",
    "ConfidenceScorer",
    "create_custom_scenario",
    "CuratorConfig",
    "DistillationConfig",
    "EvaluationConfig",
    "EvalDetail",
    "EvalReport",
    "get_builtin_scenario",
    "InteractionEvent",
    "JudgeConfig",
    "list_builtin_scenarios",
    "LoggingConfig",
    "ModelConfig",
    "PFEConfig",
    "PrivacyConfig",
    "RawSignal",
    "ScenarioConfig",
    "BUILTIN_SCENARIOS",
    "SecurityConfig",
    "ServerConfig",
    "StorageConfig",
    "TrainingSample",
    "TrainerConfig",
    "TrainerTriggerConfig",
    "can_promote",
    "can_transition",
    "to_dataclass",
    "to_pydantic",
    "validate_transition",
    # User Profile exports
    "PreferenceScore",
    "UserProfile",
    "UserProfileStore",
    "get_user_profile_store",
    "ProfileExtractor",
    "SignalAnalysisResult",
    "extract_profile_for_user",
    "get_user_profile",
    "analyze_user_conversation",
    # PII Detection & Anonymization exports
    "PIIDetector",
    "PIIDetectionResult",
    "PIIFinding",
    "PIIType",
    "PIISeverity",
    "Anonymizer",
    "AnonymizationConfig",
    "AnonymizationStrategy",
    "create_anonymizer",
    "anonymize_text",
    "PIIAuditLog",
    "PIIAuditEntry",
    "PIIComplianceReport",
    "PIIWhitelist",
]

