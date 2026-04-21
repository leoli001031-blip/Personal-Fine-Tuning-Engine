"""Stable runtime entrypoint for materialized trainer jobs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .executors import (
    execute_dpo_training,
    execute_mock_local_training,
    execute_mlx_training,
    execute_peft_training,
    execute_unsloth_training,
)
from .mlx_backend import execute_mlx_training_real, MLXBackendCapabilities
from .unsloth_backend import execute_unsloth_training_real, UnslothBackendCapabilities
from .dpo_dataset import DPODatasetBuilder


def dispatch_training_job(job_spec: Mapping[str, Any], *, dry_run: bool) -> dict[str, Any]:
    """Dispatch training job to appropriate backend.

    This function routes training jobs to the most appropriate backend
    based on the job specification and available hardware.

    Phase 2.3: Enhanced with real MLX and Unsloth backends.
    """
    backend = str(job_spec.get("execution_executor") or job_spec.get("backend") or "mock_local")

    # Phase 2.3: Use real MLX backend when available
    if backend == "mlx":
        # Check if real MLX backend is available
        if MLXBackendCapabilities.is_available() and not dry_run:
            return execute_mlx_training_real(job_spec=job_spec, dry_run=dry_run)
        # Fall back to basic executor
        return execute_mlx_training(job_spec=job_spec, dry_run=dry_run)

    # Phase 2.3: Use real Unsloth backend when available
    if backend == "unsloth":
        # Check if real Unsloth backend is available
        if UnslothBackendCapabilities.is_available() and not dry_run:
            return execute_unsloth_training_real(job_spec=job_spec, dry_run=dry_run)
        # Fall back to basic executor
        return execute_unsloth_training(job_spec=job_spec, dry_run=dry_run)

    if backend == "dpo":
        # Build DPO dataset from signals if not already provided
        training_examples = list(job_spec.get("training_examples") or [])
        if not training_examples and not dry_run:
            # Extract configuration from job_spec
            workspace = job_spec.get("workspace")
            min_confidence = job_spec.get("min_confidence", 0.7)
            limit = job_spec.get("max_samples")

            # Build dataset from signals
            dataset_builder = DPODatasetBuilder(workspace=workspace)
            dpo_dataset = dataset_builder.build_from_signals(
                min_confidence=min_confidence,
                limit=limit,
            )

            # Convert dataset to training examples
            if len(dpo_dataset) > 0:
                training_examples = [
                    {
                        "instruction": dpo_dataset["prompt"][i],
                        "chosen": dpo_dataset["chosen"][i],
                        "rejected": dpo_dataset["rejected"][i],
                        "sample_type": "dpo",
                    }
                    for i in range(len(dpo_dataset))
                ]
                # Update job_spec with training examples
                job_spec = dict(job_spec)
                job_spec["training_examples"] = training_examples

        return execute_dpo_training(job_spec=job_spec, dry_run=dry_run)
    if backend == "peft":
        return execute_peft_training(job_spec=job_spec, dry_run=dry_run)

    return execute_mock_local_training(job_spec=job_spec, dry_run=dry_run)


def run_training_job_file(
    job_json_path: str | Path,
    *,
    result_json_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    job_path = Path(job_json_path).expanduser()
    execution_plan = json.loads(job_path.read_text(encoding="utf-8"))
    job_spec = execution_plan.get("job_spec") or execution_plan
    result = dispatch_training_job(job_spec, dry_run=dry_run)
    output_path = Path(result_json_path or execution_plan.get("result_json_path") or job_path.with_name("training_job_result.json")).expanduser()
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a materialized PFE trainer job")
    parser.add_argument("--job-json", required=True)
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = run_training_job_file(args.job_json, result_json_path=args.result_json, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
