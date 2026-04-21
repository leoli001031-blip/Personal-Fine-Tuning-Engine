"""Trainer service exports."""

from .service import TrainerService, TrainingRunResult
from .executors import (
    TrainerBackendRecipe,
    TrainerExecutionPlan,
    TrainerImportProbe,
    TrainerJobSpec,
    TrainerJobMaterialization,
    TrainerJobRunResult,
    build_training_execution_recipe,
    materialize_training_job_bundle,
    execute_mock_local_training,
    execute_mlx_training,
    execute_peft_training,
    execute_unsloth_training,
    probe_trainer_executor,
    run_materialized_training_job_bundle,
    summarize_real_training_execution,
    summarize_training_job_execution,
)
from .runtime import detect_trainer_runtime, plan_trainer_backend, summarize_trainer_backend_plan, trainer_runtime_summary
from .runtime_job import dispatch_training_job, run_training_job_file
from .dpo_dataset import DPODatasetBuilder, DPOPair, build_dpo_dataset_from_samples
from .dpo_executor import DPOTrainerExecutor, TrainingResult as DPOTrainingResult, execute_dpo_training, check_dpo_dependencies

# Phase 2.3: Training efficiency optimization backends
from .mlx_backend import (
    MLXTrainerBackend,
    MLXTrainingConfig,
    MLXTrainingResult,
    MLXBackendCapabilities,
    execute_mlx_training_real,
    get_mlx_capabilities,
)
from .unsloth_backend import (
    UnslothTrainerBackend,
    UnslothTrainingConfig,
    UnslothTrainingResult,
    UnslothBackendCapabilities,
    execute_unsloth_training_real,
    get_unsloth_capabilities,
)
from .backend_selector import (
    AutoBackendSelector,
    BackendSelectionResult,
    HardwareProfile,
    DependencyProfile,
    select_optimal_backend,
    get_backend_selection_summary,
)

# Phase 2.3: CLI
from .cli import main as cli_main, create_parser

# Phase 2.4: Forget Detection
from .forget_detector import (
    ForgetDetector,
    ForgetMetrics,
    ReplaySample,
    create_forget_detector,
    detect_forget_from_training_result,
)

# Phase 2-A: Dynamic Replay Strategy
from .replay_strategy import (
    AdaptiveReplayStrategy,
    AggressiveReplayStrategy,
    ConservativeReplayStrategy,
    ReplayPlan,
    ReplayStrategy,
    create_replay_strategy,
    get_strategy_summary,
)

# Phase 2-D: Incremental Training with Forget Control
from .adapter_lineage import (
    AdapterLineageTracker,
    LineageNode,
    LineageDecision,
    get_lineage_tracker,
)
from .auto_rollback import (
    AutoRollbackPolicy,
    RollbackDecision,
    get_auto_rollback_policy,
)

trainer = TrainerService()
service = trainer
train = trainer.train
train_result = trainer.train_result

__all__ = [
    # Core services
    "TrainerService",
    "TrainingRunResult",
    # Executors
    "TrainerBackendRecipe",
    "TrainerExecutionPlan",
    "TrainerImportProbe",
    "TrainerJobSpec",
    "TrainerJobMaterialization",
    "TrainerJobRunResult",
    "build_training_execution_recipe",
    "detect_trainer_runtime",
    "execute_mock_local_training",
    "execute_mlx_training",
    "execute_peft_training",
    "execute_unsloth_training",
    "plan_trainer_backend",
    "probe_trainer_executor",
    "materialize_training_job_bundle",
    "run_materialized_training_job_bundle",
    "summarize_real_training_execution",
    "summarize_training_job_execution",
    "dispatch_training_job",
    "run_training_job_file",
    "summarize_trainer_backend_plan",
    "trainer_runtime_summary",
    # DPO
    "DPODatasetBuilder",
    "DPOPair",
    "build_dpo_dataset_from_samples",
    "DPOTrainerExecutor",
    "DPOTrainingResult",
    "execute_dpo_training",
    "check_dpo_dependencies",
    # Phase 2.3: MLX Backend
    "MLXTrainerBackend",
    "MLXTrainingConfig",
    "MLXTrainingResult",
    "MLXBackendCapabilities",
    "execute_mlx_training_real",
    "get_mlx_capabilities",
    # Phase 2.3: Unsloth Backend
    "UnslothTrainerBackend",
    "UnslothTrainingConfig",
    "UnslothTrainingResult",
    "UnslothBackendCapabilities",
    "execute_unsloth_training_real",
    "get_unsloth_capabilities",
    # Phase 2.3: Backend Selector
    "AutoBackendSelector",
    "BackendSelectionResult",
    "HardwareProfile",
    "DependencyProfile",
    "select_optimal_backend",
    "get_backend_selection_summary",
    # Phase 2.3: CLI
    "cli_main",
    "create_parser",
    # Phase 2.4: Forget Detection
    "ForgetDetector",
    "ForgetMetrics",
    "ReplaySample",
    "create_forget_detector",
    "detect_forget_from_training_result",
    # Phase 2-A: Dynamic Replay Strategy
    "AdaptiveReplayStrategy",
    "AggressiveReplayStrategy",
    "ConservativeReplayStrategy",
    "ReplayPlan",
    "ReplayStrategy",
    "create_replay_strategy",
    "get_strategy_summary",
    # Phase 2-D: Incremental Training with Forget Control
    "AdapterLineageTracker",
    "LineageNode",
    "LineageDecision",
    "get_lineage_tracker",
    "AutoRollbackPolicy",
    "RollbackDecision",
    "get_auto_rollback_policy",
    # Convenience exports
    "service",
    "trainer",
    "train",
    "train_result",
]
