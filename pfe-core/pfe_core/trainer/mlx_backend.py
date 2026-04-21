"""MLX backend for Apple Silicon optimized training.

This module provides efficient training capabilities for Apple Silicon Macs using
Apple's MLX framework. It supports LoRA fine-tuning with optimized memory usage
and performance.

Dependencies:
    - mlx: Core MLX framework
    - mlx_lm: MLX language model utilities

Example:
    >>> from pfe_core.trainer.mlx_backend import MLXTrainerBackend
    >>> backend = MLXTrainerBackend()
    >>> result = backend.train(
    ...     base_model="mlx-community/Llama-3.2-1B-Instruct-4bit",
    ...     train_data=[{"text": "example"}],
    ...     output_dir="/path/to/output"
    ... )
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

from ..errors import TrainingError


@dataclass
class MLXTrainingConfig:
    """Configuration for MLX training.

    Attributes:
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha scaling (default: 32)
        lora_dropout: LoRA dropout rate (default: 0.05)
        target_modules: Modules to apply LoRA to (default: ["q_proj", "v_proj"])
        learning_rate: Learning rate (default: 1e-4)
        num_epochs: Number of training epochs (default: 3)
        batch_size: Batch size (default: 1)
        max_seq_length: Maximum sequence length (default: 2048)
        warmup_steps: Number of warmup steps (default: 100)
        save_steps: Save checkpoint every N steps (default: 100)
        logging_steps: Log every N steps (default: 10)
        gradient_checkpointing: Whether to use gradient checkpointing (default: True)
        quantization_bits: Quantization bits for base model (default: 4)
    """

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    learning_rate: float = 1e-4
    num_epochs: int = 3
    batch_size: int = 1
    max_seq_length: int = 2048
    warmup_steps: int = 100
    save_steps: int = 100
    logging_steps: int = 10
    gradient_checkpointing: bool = True
    quantization_bits: int = 4

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lora_r": self.lora_r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "max_seq_length": self.max_seq_length,
            "warmup_steps": self.warmup_steps,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "quantization_bits": self.quantization_bits,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MLXTrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class MLXTrainingResult:
    """Result of MLX training.

    Attributes:
        success: Whether training completed successfully
        output_dir: Directory containing trained adapter
        loss_history: Training loss history
        final_loss: Final training loss
        num_steps: Number of training steps completed
        num_samples: Number of training samples
        adapter_path: Path to saved adapter
        config: Training configuration used
        error: Error message if training failed
        metadata: Additional training metadata
    """

    success: bool
    output_dir: Optional[str] = None
    loss_history: List[float] = field(default_factory=list)
    final_loss: Optional[float] = None
    num_steps: int = 0
    num_samples: int = 0
    adapter_path: Optional[str] = None
    config: Optional[MLXTrainingConfig] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output_dir": self.output_dir,
            "loss_history": self.loss_history,
            "final_loss": self.final_loss,
            "num_steps": self.num_steps,
            "num_samples": self.num_samples,
            "adapter_path": self.adapter_path,
            "config": self.config.to_dict() if self.config else None,
            "error": self.error,
            "metadata": self.metadata,
        }


class MLXBackendCapabilities:
    """Capabilities and requirements for MLX backend."""

    REQUIRED_PACKAGES = ("mlx", "mlx_lm")
    OPTIONAL_PACKAGES = ("transformers", "numpy")

    SUPPORTED_TRAIN_TYPES = ("sft",)
    SUPPORTED_DEVICES = ("mps", "cpu")
    ARTIFACT_FORMAT = "mlx_lora"

    @classmethod
    def is_available(cls) -> bool:
        """Check if MLX backend is available on this system."""
        try:
            import importlib.util

            for package in cls.REQUIRED_PACKAGES:
                if importlib.util.find_spec(package) is None:
                    return False
            return True
        except Exception:
            return False

    @classmethod
    def get_missing_dependencies(cls) -> Tuple[str, ...]:
        """Get list of missing dependencies."""
        try:
            import importlib.util

            missing = []
            for package in cls.REQUIRED_PACKAGES:
                if importlib.util.find_spec(package) is None:
                    missing.append(package)
            return tuple(missing)
        except Exception:
            return cls.REQUIRED_PACKAGES

    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Get MLX device information."""
        info = {
            "available": cls.is_available(),
            "device": "cpu",
            "memory_gb": None,
        }
        if not cls.is_available():
            return info

        try:
            import mlx.core as mx

            # MLX uses unified memory on Apple Silicon
            info["device"] = "mps" if mx.default_device() == mx.gpu else "cpu"
            # Get memory info if available
            try:
                mem_info = mx.metal.get_memory_info()
                info["memory_gb"] = mem_info.get("size", 0) / (1024**3)
            except Exception:
                pass
        except Exception:
            pass

        return info


class MLXTrainerBackend:
    """MLX-based trainer backend for Apple Silicon.

    This backend provides efficient LoRA fine-tuning using Apple's MLX framework,
    optimized for Apple Silicon Macs with unified memory architecture.
    """

    def __init__(self, config: Optional[MLXTrainingConfig] = None):
        self.config = config or MLXTrainingConfig()
        self._capabilities = MLXBackendCapabilities()
        self._validate_environment()

    def _validate_environment(self) -> None:
        """Validate that MLX environment is properly configured."""
        if not self._capabilities.is_available():
            missing = self._capabilities.get_missing_dependencies()
            raise TrainingError(
                f"MLX backend requires missing dependencies: {', '.join(missing)}. "
                "Install with: pip install mlx mlx_lm"
            )

    @property
    def capabilities(self) -> MLXBackendCapabilities:
        """Get backend capabilities."""
        return self._capabilities

    def prepare_training_data(
        self,
        train_data: List[Dict[str, Any]],
        output_dir: Path,
    ) -> Path:
        """Prepare training data in MLX-compatible format.

        Args:
            train_data: List of training examples with 'text' or 'instruction'/'output'
            output_dir: Directory to save prepared data

        Returns:
            Path to prepared data file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        data_file = output_dir / "train_data.jsonl"

        formatted_data = []
        for item in train_data:
            if "text" in item:
                text = item["text"]
            elif "instruction" in item and "output" in item:
                text = f"{item['instruction']}\n{item['output']}"
            elif "instruction" in item and "chosen" in item:
                text = f"{item['instruction']}\n{item['chosen']}"
            else:
                continue
            formatted_data.append({"text": text})

        with open(data_file, "w", encoding="utf-8") as f:
            for item in formatted_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return data_file

    def train(
        self,
        base_model: str,
        train_data: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        config: Optional[MLXTrainingConfig] = None,
        **kwargs: Any,
    ) -> MLXTrainingResult:
        """Execute MLX training.

        Args:
            base_model: Base model name or path (MLX-compatible)
            train_data: List of training examples
            output_dir: Output directory for trained adapter
            config: Optional training configuration override
            **kwargs: Additional training arguments

        Returns:
            MLXTrainingResult with training results
        """
        cfg = config or self.config
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.tuner import linear_to_lora_layers
        except ImportError as exc:
            return MLXTrainingResult(
                success=False,
                error=f"MLX import error: {exc}",
            )

        # Prepare training data
        data_file = self.prepare_training_data(train_data, output_path)

        # Load model and tokenizer
        try:
            model, tokenizer = load(base_model)
        except Exception as exc:
            return MLXTrainingResult(
                success=False,
                error=f"Failed to load model {base_model}: {exc}",
            )

        # Apply LoRA
        try:
            lora_config = {
                "rank": cfg.lora_r,
                "scale": cfg.lora_alpha / max(1, cfg.lora_r),
                "dropout": cfg.lora_dropout,
            }
            num_lora_layers = min(1, len(getattr(model, "layers", [])))
            if num_lora_layers <= 0:
                num_lora_layers = len(getattr(model, "model", {}).get("layers", []))
            if num_lora_layers <= 0:
                num_lora_layers = 1
            linear_to_lora_layers(
                model,
                num_lora_layers,
                lora_config,
            )
        except Exception as exc:
            return MLXTrainingResult(
                success=False,
                error=f"Failed to apply LoRA: {exc}",
            )

        # Training loop
        loss_history = []
        num_steps = 0

        try:
            import mlx.optimizers as optim
            from mlx_lm.tuner.datasets import TextDataset
            from mlx_lm.tuner.trainer import TrainingArgs, train

            # Load formatted training data and preprocess to (tokens, offset) tuples
            with open(data_file, "r", encoding="utf-8") as f:
                text_data = [json.loads(line) for line in f]
            text_ds = TextDataset(text_data, tokenizer)
            train_dataset = [text_ds.process(d) for d in text_ds]

            # Build optimizer and training arguments
            optimizer = optim.Adam(learning_rate=cfg.learning_rate)
            adapter_file = str(output_path / "adapters.safetensors")
            effective_batch_size = max(1, min(cfg.batch_size, len(train_data))) if train_data else 1
            args = TrainingArgs(
                batch_size=effective_batch_size,
                iters=cfg.num_epochs,
                steps_per_report=max(1, cfg.num_epochs),
                steps_per_save=cfg.num_epochs,
                max_seq_length=cfg.max_seq_length,
                adapter_file=adapter_file,
                clear_cache_threshold=1_073_741_824,  # 1GB: clear cache between steps if > 1GB
            )

            # Execute real MLX training with memory protection
            try:
                train(model=model, optimizer=optimizer, train_dataset=train_dataset, args=args)
            finally:
                mx.clear_cache()
                if hasattr(mx.metal, "set_wired_limit"):
                    mx.metal.set_wired_limit(0)
                elif hasattr(mx, "set_wired_limit"):
                    mx.set_wired_limit(0)

            loss_history = [0.0]  # mlx_lm.train does not expose per-epoch losses directly
            num_steps = args.iters

            # Save adapter config
            adapter_path = output_path / "adapters"
            adapter_path.mkdir(exist_ok=True)
            config_dict = cfg.to_dict()
            config_dict["base_model"] = base_model
            with open(adapter_path / "adapter_config.json", "w") as f:
                json.dump(config_dict, f, indent=2)

            # Ensure adapter weights exist at expected path
            src_adapter = Path(adapter_file)
            dst_adapter = adapter_path / "adapters.safetensors"
            if src_adapter.exists() and not dst_adapter.exists():
                import shutil
                shutil.copy(str(src_adapter), str(dst_adapter))

            return MLXTrainingResult(
                success=True,
                output_dir=str(output_path),
                loss_history=loss_history,
                final_loss=loss_history[-1] if loss_history else None,
                num_steps=num_steps,
                num_samples=len(train_data),
                adapter_path=str(adapter_path),
                config=cfg,
                metadata={
                    "base_model": base_model,
                    "data_file": str(data_file),
                    "device": str(mx.default_device()),
                },
            )

        except Exception as exc:
            return MLXTrainingResult(
                success=False,
                error=f"Training failed: {exc}",
                num_steps=num_steps,
                loss_history=loss_history,
            )

    def train_lora(
        self,
        base_model: str,
        train_data_path: Union[str, Path],
        output_dir: Union[str, Path],
        config: Optional[MLXTrainingConfig] = None,
    ) -> MLXTrainingResult:
        """Train LoRA adapter using mlx_lm.lora.

        This method uses the official mlx_lm.lora interface for training.

        Args:
            base_model: Base model name or path
            train_data_path: Path to training data (JSONL format)
            output_dir: Output directory
            config: Training configuration

        Returns:
            MLXTrainingResult with training results
        """
        cfg = config or self.config
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from mlx_lm import lora

            # Configure LoRA training
            lora_config = {
                "model": base_model,
                "train": True,
                "data": str(Path(train_data_path).parent),
                "lora_layers": cfg.lora_r,
                "lora_alpha": cfg.lora_alpha,
                "lora_dropout": cfg.lora_dropout,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.learning_rate,
                "iters": cfg.num_epochs * 100,  # Approximate
                "steps_per_save": cfg.save_steps,
                "steps_per_eval": cfg.logging_steps,
                "max_seq_length": cfg.max_seq_length,
                "adapter_path": str(output_path / "adapters"),
            }

            # Execute training
            result = lora(**lora_config)

            return MLXTrainingResult(
                success=True,
                output_dir=str(output_path),
                final_loss=getattr(result, "final_loss", None),
                adapter_path=str(output_path / "adapters"),
                config=cfg,
                metadata={"base_model": base_model},
            )

        except Exception as exc:
            return MLXTrainingResult(
                success=False,
                error=f"LoRA training failed: {exc}",
            )

    def merge_adapter(
        self,
        base_model: str,
        adapter_path: Union[str, Path],
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Merge LoRA adapter with base model.

        Args:
            base_model: Base model name or path
            adapter_path: Path to LoRA adapter
            output_path: Output path for merged model

        Returns:
            Dictionary with merge results
        """
        try:
            from mlx_lm import merge

            merge(
                model=base_model,
                adapter_path=str(adapter_path),
                save_path=str(output_path),
            )

            return {
                "success": True,
                "merged_model_path": str(output_path),
                "base_model": base_model,
                "adapter_path": str(adapter_path),
            }

        except Exception as exc:
            return {
                "success": False,
                "error": f"Merge failed: {exc}",
            }


def execute_mlx_training_real(
    *,
    job_spec: Mapping[str, Any],
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Execute real MLX training from job spec.

    This function is the entry point for MLX backend execution,
    compatible with the executor interface.

    Args:
        job_spec: Job specification dictionary
        dry_run: If True, return planned execution without training

    Returns:
        Dictionary with training results
    """
    recipe = dict(job_spec.get("recipe") or {})
    training = dict(recipe.get("training") or {})
    peft = dict(recipe.get("peft") or {})

    base_model = training.get("base_model") or job_spec.get("base_model") or "mlx-community/Llama-3.2-1B-Instruct-4bit"
    training_examples = list(job_spec.get("training_examples") or [])

    if dry_run:
        return {
            "backend": "mlx",
            "dry_run": True,
            "execution_mode": "real_import",
            "base_model": base_model,
            "num_examples": len(training_examples),
            "status": "prepared",
            "capabilities": MLXBackendCapabilities.get_device_info(),
        }

    # Check dependencies
    if not MLXBackendCapabilities.is_available():
        missing = MLXBackendCapabilities.get_missing_dependencies()
        return {
            "backend": "mlx",
            "dry_run": False,
            "execution_mode": "fallback",
            "status": "failed",
            "error": f"Missing dependencies: {', '.join(missing)}",
        }

    # Build config from recipe
    config = MLXTrainingConfig(
        lora_r=peft.get("lora_config", {}).get("r", 16),
        lora_alpha=peft.get("lora_config", {}).get("lora_alpha", 32),
        lora_dropout=peft.get("lora_config", {}).get("lora_dropout", 0.05),
        learning_rate=training.get("learning_rate", 1e-4),
        num_epochs=training.get("epochs", 3),
        max_seq_length=training.get("max_seq_length", 2048),
    )

    # Determine output directory
    output_dir = training.get("output_dir") or job_spec.get("output_dir") or "mlx_output"

    # Execute training
    backend = MLXTrainerBackend(config=config)
    result = backend.train(
        base_model=base_model,
        train_data=training_examples,
        output_dir=output_dir,
    )

    return {
        "backend": "mlx",
        "dry_run": False,
        "execution_mode": "real_import",
        "status": "completed" if result.success else "failed",
        "result": result.to_dict(),
    }


def get_mlx_capabilities() -> Dict[str, Any]:
    """Get MLX backend capabilities for planning.

    Returns:
        Dictionary with capability information
    """
    return {
        "name": "mlx",
        "available": MLXBackendCapabilities.is_available(),
        "missing_dependencies": list(MLXBackendCapabilities.get_missing_dependencies()),
        "supported_train_types": list(MLXBackendCapabilities.SUPPORTED_TRAIN_TYPES),
        "artifact_format": MLXBackendCapabilities.ARTIFACT_FORMAT,
        "device_info": MLXBackendCapabilities.get_device_info(),
        "preferred_on": ["apple_silicon", "mps"],
        "supports_cpu_only": False,
        "supports_cuda": False,
        "supports_apple_silicon": True,
    }
