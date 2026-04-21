"""Unsloth backend for CUDA-optimized training.

This module provides highly efficient training capabilities using the Unsloth
library, which offers 2-5x faster training and 80% less memory usage for LoRA
fine-tuning on CUDA GPUs.

Dependencies:
    - unsloth: Core unsloth library
    - torch: PyTorch with CUDA support
    - transformers: Hugging Face transformers
    - trl: Transformers Reinforcement Learning

Example:
    >>> from pfe_core.trainer.unsloth_backend import UnslothTrainerBackend
    >>> backend = UnslothTrainerBackend()
    >>> result = backend.train(
    ...     base_model="unsloth/Llama-3.2-1B",
    ...     train_data=[{"instruction": "Hello", "output": "Hi"}],
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
class UnslothTrainingConfig:
    """Configuration for Unsloth training.

    Attributes:
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha scaling (default: 16)
        lora_dropout: LoRA dropout rate (default: 0)
        target_modules: Modules to apply LoRA to
        learning_rate: Learning rate (default: 2e-4)
        num_epochs: Number of training epochs (default: 3)
        batch_size: Batch size (default: 2)
        max_seq_length: Maximum sequence length (default: 2048)
        warmup_steps: Number of warmup steps (default: 5)
        save_steps: Save checkpoint every N steps (default: 100)
        logging_steps: Log every N steps (default: 1)
        gradient_accumulation_steps: Gradient accumulation steps (default: 4)
        max_grad_norm: Maximum gradient norm (default: 0.3)
        weight_decay: Weight decay (default: 0.01)
        use_rslora: Whether to use Rank-Stabilized LoRA (default: False)
        use_gradient_checkpointing: Enable gradient checkpointing (default: True)
        random_state: Random seed (default: 3407)
    """

    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2
    max_seq_length: int = 2048
    warmup_steps: int = 5
    save_steps: int = 100
    logging_steps: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.3
    weight_decay: float = 0.01
    use_rslora: bool = False
    use_gradient_checkpointing: bool = True
    random_state: int = 3407

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
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "weight_decay": self.weight_decay,
            "use_rslora": self.use_rslora,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "random_state": self.random_state,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "UnslothTrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class UnslothTrainingResult:
    """Result of Unsloth training.

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
        training_stats: Unsloth-specific training statistics
    """

    success: bool
    output_dir: Optional[str] = None
    loss_history: List[float] = field(default_factory=list)
    final_loss: Optional[float] = None
    num_steps: int = 0
    num_samples: int = 0
    adapter_path: Optional[str] = None
    config: Optional[UnslothTrainingConfig] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    training_stats: Dict[str, Any] = field(default_factory=dict)

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
            "training_stats": self.training_stats,
        }


class UnslothBackendCapabilities:
    """Capabilities and requirements for Unsloth backend."""

    REQUIRED_PACKAGES = ("unsloth", "torch", "transformers", "trl")
    OPTIONAL_PACKAGES = ("peft", "accelerate", "bitsandbytes")

    SUPPORTED_TRAIN_TYPES = ("sft", "dpo")
    SUPPORTED_DEVICES = ("cuda",)
    ARTIFACT_FORMAT = "peft_lora"

    @classmethod
    def is_available(cls) -> bool:
        """Check if Unsloth backend is available on this system."""
        try:
            import importlib.util

            for package in cls.REQUIRED_PACKAGES:
                if importlib.util.find_spec(package) is None:
                    return False

            # Also check for CUDA availability
            import torch

            return torch.cuda.is_available()
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

            # Check CUDA
            if not missing:
                import torch

                if not torch.cuda.is_available():
                    missing.append("cuda")

            return tuple(missing)
        except Exception:
            return cls.REQUIRED_PACKAGES

    @classmethod
    def get_device_info(cls) -> Dict[str, Any]:
        """Get CUDA device information."""
        info = {
            "available": cls.is_available(),
            "device": "cpu",
            "device_name": None,
            "memory_gb": None,
            "cuda_version": None,
        }
        if not cls.is_available():
            return info

        try:
            import torch

            if torch.cuda.is_available():
                info["device"] = "cuda"
                info["device_name"] = torch.cuda.get_device_name(0)
                info["memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (
                    1024**3
                )
                info["cuda_version"] = torch.version.cuda
        except Exception:
            pass

        return info

    @classmethod
    def get_unsloth_version(cls) -> Optional[str]:
        """Get installed Unsloth version."""
        try:
            import unsloth

            return getattr(unsloth, "__version__", None)
        except Exception:
            return None


class UnslothTrainerBackend:
    """Unsloth-based trainer backend for CUDA-optimized training.

    This backend provides highly efficient LoRA fine-tuning using the Unsloth
    library, offering 2-5x faster training and 80% less memory usage compared
    to standard PEFT training.
    """

    def __init__(self, config: Optional[UnslothTrainingConfig] = None):
        self.config = config or UnslothTrainingConfig()
        self._capabilities = UnslothBackendCapabilities()
        self._validate_environment()

    def _validate_environment(self) -> None:
        """Validate that Unsloth environment is properly configured."""
        if not self._capabilities.is_available():
            missing = self._capabilities.get_missing_dependencies()
            raise TrainingError(
                f"Unsloth backend requires missing dependencies: {', '.join(missing)}. "
                "Install with: pip install unsloth"
            )

    @property
    def capabilities(self) -> UnslothBackendCapabilities:
        """Get backend capabilities."""
        return self._capabilities

    def format_training_data(
        self,
        train_data: List[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        """Format training data for Unsloth.

        Args:
            train_data: List of training examples

        Returns:
            Formatted training data
        """
        formatted = []
        for item in train_data:
            if "text" in item:
                formatted.append({"text": item["text"]})
            elif "instruction" in item and "output" in item:
                # Alpaca format
                instruction = item["instruction"]
                input_text = item.get("input", "")
                output = item["output"]

                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                formatted.append({"text": text})
            elif "instruction" in item and "chosen" in item:
                # For DPO-style data in SFT
                text = f"{item['instruction']}\n{item['chosen']}"
                formatted.append({"text": text})
            elif "prompt" in item and "completion" in item:
                formatted.append({"text": item["prompt"] + item["completion"]})

        return formatted

    def train(
        self,
        base_model: str,
        train_data: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        config: Optional[UnslothTrainingConfig] = None,
        **kwargs: Any,
    ) -> UnslothTrainingResult:
        """Execute Unsloth training.

        Args:
            base_model: Base model name or path (Unsloth-compatible)
            train_data: List of training examples
            output_dir: Output directory for trained adapter
            config: Optional training configuration override
            **kwargs: Additional training arguments

        Returns:
            UnslothTrainingResult with training results
        """
        cfg = config or self.config
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer
            from transformers import TrainingArguments
            from datasets import Dataset
        except ImportError as exc:
            return UnslothTrainingResult(
                success=False,
                error=f"Unsloth import error: {exc}",
            )

        # Format training data
        formatted_data = self.format_training_data(train_data)
        if not formatted_data:
            return UnslothTrainingResult(
                success=False,
                error="No valid training data found",
            )

        # Create dataset
        dataset = Dataset.from_list(formatted_data)

        # Load model with Unsloth
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=cfg.max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=True,
            )
        except Exception as exc:
            return UnslothTrainingResult(
                success=False,
                error=f"Failed to load model {base_model}: {exc}",
            )

        # Apply LoRA
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=cfg.lora_r,
                target_modules=cfg.target_modules,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                use_gradient_checkpointing=cfg.use_gradient_checkpointing,
                random_state=cfg.random_state,
                use_rslora=cfg.use_rslora,
            )
        except Exception as exc:
            return UnslothTrainingResult(
                success=False,
                error=f"Failed to apply LoRA: {exc}",
            )

        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            warmup_steps=cfg.warmup_steps,
            learning_rate=cfg.learning_rate,
            max_grad_norm=cfg.max_grad_norm,
            weight_decay=cfg.weight_decay,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            save_strategy="steps",
            optim="adamw_8bit",
            seed=cfg.random_state,
            report_to="none",
        )

        # Create trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            args=training_args,
        )

        # Train
        try:
            train_result = trainer.train()

            # Save adapter
            adapter_path = output_path / "adapter"
            adapter_path.mkdir(exist_ok=True)
            model.save_pretrained(str(adapter_path))
            tokenizer.save_pretrained(str(adapter_path))

            # Get training stats
            training_stats = {
                "training_runtime": getattr(train_result, "training_runtime", None),
                "train_samples_per_second": getattr(
                    train_result, "train_samples_per_second", None
                ),
                "train_steps_per_second": getattr(
                    train_result, "train_steps_per_second", None
                ),
                "total_flos": getattr(train_result, "total_flos", None),
            }

            return UnslothTrainingResult(
                success=True,
                output_dir=str(output_path),
                final_loss=train_result.training_loss if hasattr(train_result, "training_loss") else None,
                num_steps=train_result.global_step if hasattr(train_result, "global_step") else 0,
                num_samples=len(train_data),
                adapter_path=str(adapter_path),
                config=cfg,
                metadata={
                    "base_model": base_model,
                    "model_type": "unsloth_fast",
                },
                training_stats=training_stats,
            )

        except Exception as exc:
            return UnslothTrainingResult(
                success=False,
                error=f"Training failed: {exc}",
            )

    def train_dpo(
        self,
        base_model: str,
        train_data: List[Dict[str, Any]],
        output_dir: Union[str, Path],
        config: Optional[UnslothTrainingConfig] = None,
        beta: float = 0.1,
        **kwargs: Any,
    ) -> UnslothTrainingResult:
        """Execute DPO training with Unsloth.

        Args:
            base_model: Base model name or path
            train_data: List of training examples with 'prompt', 'chosen', 'rejected'
            output_dir: Output directory
            config: Training configuration
            beta: DPO beta parameter
            **kwargs: Additional arguments

        Returns:
            UnslothTrainingResult with training results
        """
        cfg = config or self.config
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        try:
            from unsloth import FastLanguageModel
            from trl import DPOTrainer
            from transformers import TrainingArguments
            from datasets import Dataset
        except ImportError as exc:
            return UnslothTrainingResult(
                success=False,
                error=f"Import error: {exc}",
            )

        # Format DPO data
        dpo_data = {"prompt": [], "chosen": [], "rejected": []}
        for item in train_data:
            prompt = item.get("instruction") or item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")

            if prompt and chosen and rejected:
                dpo_data["prompt"].append(prompt)
                dpo_data["chosen"].append(chosen)
                dpo_data["rejected"].append(rejected)

        if not dpo_data["prompt"]:
            return UnslothTrainingResult(
                success=False,
                error="No valid DPO training data found",
            )

        dataset = Dataset.from_dict(dpo_data)

        # Load model
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=base_model,
                max_seq_length=cfg.max_seq_length,
                dtype=None,
                load_in_4bit=True,
            )
        except Exception as exc:
            return UnslothTrainingResult(
                success=False,
                error=f"Failed to load model: {exc}",
            )

        # Apply LoRA
        model = FastLanguageModel.get_peft_model(
            model,
            r=cfg.lora_r,
            target_modules=cfg.target_modules,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            use_gradient_checkpointing=cfg.use_gradient_checkpointing,
            random_state=cfg.random_state,
            use_rslora=cfg.use_rslora,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=cfg.num_epochs,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            learning_rate=cfg.learning_rate,
            max_grad_norm=cfg.max_grad_norm,
            warmup_steps=cfg.warmup_steps,
            logging_steps=cfg.logging_steps,
            save_steps=cfg.save_steps,
            optim="adamw_8bit",
            report_to="none",
        )

        # DPO Trainer
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # Will create from model
            tokenizer=tokenizer,
            train_dataset=dataset,
            beta=beta,
            args=training_args,
            max_length=cfg.max_seq_length,
            max_prompt_length=cfg.max_seq_length // 2,
        )

        try:
            train_result = trainer.train()

            # Save
            adapter_path = output_path / "adapter"
            adapter_path.mkdir(exist_ok=True)
            model.save_pretrained(str(adapter_path))
            tokenizer.save_pretrained(str(adapter_path))

            return UnslothTrainingResult(
                success=True,
                output_dir=str(output_path),
                final_loss=getattr(train_result, "training_loss", None),
                adapter_path=str(adapter_path),
                config=cfg,
                metadata={"base_model": base_model, "train_type": "dpo", "beta": beta},
            )

        except Exception as exc:
            return UnslothTrainingResult(
                success=False,
                error=f"DPO training failed: {exc}",
            )


def execute_unsloth_training_real(
    *,
    job_spec: Mapping[str, Any],
    dry_run: bool = True,
) -> Dict[str, Any]:
    """Execute real Unsloth training from job spec.

    This function is the entry point for Unsloth backend execution,
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

    base_model = (
        training.get("base_model")
        or job_spec.get("base_model")
        or "unsloth/Llama-3.2-1B"
    )
    training_examples = list(job_spec.get("training_examples") or [])
    train_type = training.get("train_type", "sft")

    if dry_run:
        return {
            "backend": "unsloth",
            "dry_run": True,
            "execution_mode": "real_import",
            "base_model": base_model,
            "num_examples": len(training_examples),
            "train_type": train_type,
            "status": "prepared",
            "capabilities": UnslothBackendCapabilities.get_device_info(),
        }

    # Check dependencies
    if not UnslothBackendCapabilities.is_available():
        missing = UnslothBackendCapabilities.get_missing_dependencies()
        return {
            "backend": "unsloth",
            "dry_run": False,
            "execution_mode": "fallback",
            "status": "failed",
            "error": f"Missing dependencies: {', '.join(missing)}",
        }

    # Build config from recipe
    config = UnslothTrainingConfig(
        lora_r=peft.get("lora_config", {}).get("r", 16),
        lora_alpha=peft.get("lora_config", {}).get("lora_alpha", 16),
        lora_dropout=peft.get("lora_config", {}).get("lora_dropout", 0.0),
        learning_rate=training.get("learning_rate", 2e-4),
        num_epochs=training.get("epochs", 3),
        max_seq_length=training.get("max_seq_length", 2048),
        batch_size=training.get("batch_size", 2),
    )

    # Determine output directory
    output_dir = (
        training.get("output_dir")
        or job_spec.get("output_dir")
        or "unsloth_output"
    )

    # Execute training
    backend = UnslothTrainerBackend(config=config)

    if train_type == "dpo":
        result = backend.train_dpo(
            base_model=base_model,
            train_data=training_examples,
            output_dir=output_dir,
            beta=training.get("dpo_beta", 0.1),
        )
    else:
        result = backend.train(
            base_model=base_model,
            train_data=training_examples,
            output_dir=output_dir,
        )

    return {
        "backend": "unsloth",
        "dry_run": False,
        "execution_mode": "real_import",
        "status": "completed" if result.success else "failed",
        "result": result.to_dict(),
    }


def get_unsloth_capabilities() -> Dict[str, Any]:
    """Get Unsloth backend capabilities for planning.

    Returns:
        Dictionary with capability information
    """
    return {
        "name": "unsloth",
        "available": UnslothBackendCapabilities.is_available(),
        "missing_dependencies": list(UnslothBackendCapabilities.get_missing_dependencies()),
        "supported_train_types": list(UnslothBackendCapabilities.SUPPORTED_TRAIN_TYPES),
        "artifact_format": UnslothBackendCapabilities.ARTIFACT_FORMAT,
        "device_info": UnslothBackendCapabilities.get_device_info(),
        "version": UnslothBackendCapabilities.get_unsloth_version(),
        "preferred_on": ["cuda"],
        "supports_cpu_only": False,
        "supports_cuda": True,
        "supports_apple_silicon": False,
    }
