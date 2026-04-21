"""DPO Trainer Executor for PFE.

Executes Direct Preference Optimization training using trl.DPOTrainer.
Supports QLoRA fine-tuning and incremental training from existing adapters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import Dataset

from ..config import TrainerConfig
from ..errors import TrainingError


@dataclass
class TrainingResult:
    """Result of DPO training execution.

    Attributes:
        success: Whether training completed successfully
        adapter_path: Path to the saved adapter
        metrics: Training metrics (loss, etc.)
        num_samples: Number of training samples used
        config: Training configuration used
        error: Error message if training failed
    """

    success: bool
    adapter_path: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    num_samples: int = 0
    config: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class DPOTrainerExecutor:
    """Execute DPO training using trl.DPOTrainer.

    This executor handles:
    - Loading base models with optional existing adapters
    - Configuring QLoRA/LoRA for efficient fine-tuning
    - Running DPO training with proper hyperparameters
    - Saving the resulting adapter
    """

    def __init__(self, config: TrainerConfig):
        """Initialize the DPO trainer executor.

        Args:
            config: Trainer configuration with DPO settings
        """
        self.config = config
        self._validate_dependencies()

    def _validate_dependencies(self) -> None:
        """Check that required dependencies are available."""
        try:
            import torch
            import transformers
            from trl import DPOTrainer
            from peft import LoraConfig, get_peft_model, PeftModel
        except ImportError as e:
            raise TrainingError(
                f"DPO training requires torch, transformers, trl, and peft: {e}"
            ) from e

    def train(
        self,
        base_model_path: str,
        adapter_output_path: str,
        dpo_dataset: "Dataset",
        base_adapter_path: Optional[str] = None,
        eval_dataset: Optional["Dataset"] = None,
    ) -> TrainingResult:
        """Execute DPO training.

        Args:
            base_model_path: Path or name of the base model
            adapter_output_path: Path to save the output adapter
            dpo_dataset: DPO dataset with prompt, chosen, rejected columns
            base_adapter_path: Optional path to existing adapter for incremental training
            eval_dataset: Optional evaluation dataset

        Returns:
            TrainingResult with success status and metrics
        """
        if len(dpo_dataset) == 0:
            return TrainingResult(
                success=False,
                error="DPO dataset is empty",
            )

        try:
            model, tokenizer = self._load_model_for_dpo(
                base_model_path, base_adapter_path
            )

            # Prepare datasets
            train_dataset = self._prepare_dataset(dpo_dataset, tokenizer)
            eval_dataset = (
                self._prepare_dataset(eval_dataset, tokenizer)
                if eval_dataset is not None
                else None
            )

            # Setup training arguments
            training_args = self._create_training_args(adapter_output_path)

            # Initialize DPOTrainer
            trainer = self._create_dpo_trainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                training_args=training_args,
            )

            # Train
            train_result = trainer.train()

            # Save adapter
            trainer.save_model(adapter_output_path)

            # Extract metrics
            metrics = {
                "train_loss": train_result.training_loss if hasattr(train_result, "training_loss") else None,
                "train_steps": train_result.global_step if hasattr(train_result, "global_step") else None,
                "num_train_samples": len(dpo_dataset),
            }

            if eval_dataset is not None:
                eval_metrics = trainer.evaluate()
                metrics["eval_loss"] = eval_metrics.get("eval_loss")

            return TrainingResult(
                success=True,
                adapter_path=adapter_output_path,
                metrics=metrics,
                num_samples=len(dpo_dataset),
                config={
                    "base_model": base_model_path,
                    "base_adapter": base_adapter_path,
                    "dpo_beta": self.config.dpo_beta,
                    "lora_r": self.config.lora_r,
                    "lora_alpha": self.config.lora_alpha,
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.epochs,
                },
            )

        except Exception as e:
            return TrainingResult(
                success=False,
                error=str(e),
                num_samples=len(dpo_dataset),
            )

    def _load_model_for_dpo(
        self,
        base_model_path: str,
        adapter_path: Optional[str] = None,
    ) -> tuple[Any, Any]:
        """Load model and tokenizer for DPO training.

        Args:
            base_model_path: Path or name of base model
            adapter_path: Optional path to existing adapter

        Returns:
            Tuple of (model, tokenizer)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        # Determine device and dtype
        device_map = self._get_device_map()
        torch_dtype = self._get_torch_dtype()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            trust_remote_code=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure quantization
        bnb_config = None
        if self.config.method == "qlora" and self.config.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif self.config.method == "qlora" and self.config.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load base model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch_dtype if bnb_config is None else None,
            trust_remote_code=True,
        )

        # Load existing adapter if provided (incremental training)
        if adapter_path and Path(adapter_path).exists():
            model = PeftModel.from_pretrained(model, adapter_path)
            model = model.merge_and_unload() if not self.config.method == "qlora" else model

        # Apply LoRA configuration
        if self.config.method in ("lora", "qlora"):
            from peft import LoraConfig, get_peft_model, TaskType

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                target_modules=self._get_target_modules(base_model_path),
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )

            if not isinstance(model, PeftModel):
                model = get_peft_model(model, lora_config)
            else:
                # Already has PEFT, add new adapter
                model.add_adapter("dpo", lora_config)
                model.set_adapter("dpo")

        return model, tokenizer

    def _get_device_map(self) -> Union[str, Dict[str, Any]]:
        """Get device map based on configuration."""
        import torch

        if self.config.device == "auto":
            if torch.cuda.is_available():
                return "auto"
            elif torch.backends.mps.is_available():
                return {"": "mps"}
            else:
                return {"": "cpu"}
        return {"": self.config.device}

    def _get_torch_dtype(self) -> Any:
        """Get torch dtype based on configuration."""
        import torch

        if self.config.device == "cuda" and torch.cuda.is_available():
            return torch.float16
        return torch.float32

    def _get_target_modules(self, model_name: str) -> List[str]:
        """Get LoRA target modules based on model architecture."""
        model_name_lower = model_name.lower()

        if "qwen" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "llama" in model_name_lower or "mistral" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gemma" in model_name_lower:
            return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "gpt2" in model_name_lower:
            return ["c_attn", "c_proj"]
        else:
            # Default target modules
            return ["q_proj", "v_proj"]

    def _prepare_dataset(self, dataset: Dataset, tokenizer: Any) -> Dataset:
        """Prepare dataset for DPO training.

        Args:
            dataset: Input dataset with prompt, chosen, rejected
            tokenizer: Tokenizer for encoding

        Returns:
            Prepared dataset
        """
        def format_example(example: Dict[str, Any]) -> Dict[str, Any]:
            """Format a single example for DPO."""
            # DPO expects prompt, chosen, rejected fields
            return {
                "prompt": example["prompt"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            }

        # Apply formatting
        formatted = dataset.map(format_example, remove_columns=dataset.column_names)
        return formatted

    def _create_training_args(self, output_dir: str) -> Any:
        """Create training arguments for DPO.

        Args:
            output_dir: Output directory for checkpoints

        Returns:
            TrainingArguments instance
        """
        try:
            from trl import DPOConfig
        except Exception:
            DPOConfig = None  # type: ignore[misc]

        num_epochs = self.config.epochs
        learning_rate = self.config.learning_rate

        kwargs = {
            "output_dir": output_dir,
            "num_train_epochs": num_epochs,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": learning_rate,
            "max_grad_norm": 0.3,
            "warmup_steps": 0,
            "lr_scheduler_type": "cosine",
            "logging_steps": 1,
            "save_strategy": "no",
            "bf16": False,
            "fp16": self.config.device == "cuda",
            "remove_unused_columns": False,
            "run_name": "pfe-dpo-training",
            "report_to": "none",
        }

        if DPOConfig is not None:
            return DPOConfig(
                **kwargs,
                beta=self.config.dpo_beta,
                label_smoothing=0.0,
                max_length=128,
            )

        from transformers import TrainingArguments

        return TrainingArguments(
            **kwargs,
            warmup_ratio=0.03,
            save_strategy="epoch",
            evaluation_strategy="epoch" if self.config.epochs > 1 else "no",
        )

    def _create_dpo_trainer(
        self,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset],
        training_args: Any,
    ) -> Any:
        """Create DPOTrainer instance.

        Args:
            model: The model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            training_args: Training arguments

        Returns:
            DPOTrainer instance
        """
        from trl import DPOTrainer
        import inspect

        sig = inspect.signature(DPOTrainer.__init__)
        trainer_kwargs: Dict[str, Any] = {
            "model": model,
            "train_dataset": train_dataset,
            "eval_dataset": eval_dataset,
            "args": training_args,
        }
        if "processing_class" in sig.parameters:
            trainer_kwargs["processing_class"] = tokenizer
        else:
            trainer_kwargs["tokenizer"] = tokenizer

        return DPOTrainer(**trainer_kwargs)


def execute_dpo_training(
    config: TrainerConfig,
    base_model_path: str,
    adapter_output_path: str,
    dpo_dataset: Dataset,
    base_adapter_path: Optional[str] = None,
    eval_dataset: Optional[Dataset] = None,
) -> TrainingResult:
    """Convenience function to execute DPO training.

    Args:
        config: Trainer configuration
        base_model_path: Path or name of base model
        adapter_output_path: Path to save output adapter
        dpo_dataset: DPO dataset
        base_adapter_path: Optional existing adapter for incremental training
        eval_dataset: Optional evaluation dataset

    Returns:
        TrainingResult
    """
    executor = DPOTrainerExecutor(config)
    return executor.train(
        base_model_path=base_model_path,
        adapter_output_path=adapter_output_path,
        dpo_dataset=dpo_dataset,
        base_adapter_path=base_adapter_path,
        eval_dataset=eval_dataset,
    )


def check_dpo_dependencies() -> Dict[str, bool]:
    """Check if all DPO dependencies are available.

    Returns:
        Dictionary with dependency availability
    """
    result = {
        "torch": False,
        "transformers": False,
        "trl": False,
        "peft": False,
        "datasets": False,
        "accelerate": False,
        "bitsandbytes": False,
    }

    for module in result:
        try:
            __import__(module)
            result[module] = True
        except ImportError:
            pass

    result["all_available"] = all(result.values())
    return result
