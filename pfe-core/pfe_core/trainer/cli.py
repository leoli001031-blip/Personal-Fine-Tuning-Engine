"""CLI for training backend management and execution.

This module provides command-line interface for:
- Backend selection and recommendations
- Training job execution
- Backend capability checking
- Hardware and dependency profiling

Examples:
    # Check available backends and get recommendations
    $ python -m pfe_core.trainer.cli check

    # Get detailed backend recommendation
    $ python -m pfe_core.trainer.cli recommend --train-type sft

    # Run training with auto-selected backend
    $ python -m pfe_core.trainer.cli train --train-type sft --data data.jsonl

    # Run training with specific backend
    $ python -m pfe_core.trainer.cli train --backend unsloth --data data.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from .backend_selector import (
    AutoBackendSelector,
    HardwareProfile,
    DependencyProfile,
    select_optimal_backend,
    get_backend_selection_summary,
)
from .mlx_backend import (
    MLXBackendCapabilities,
    MLXTrainerBackend,
    MLXTrainingConfig,
)
from .unsloth_backend import (
    UnslothBackendCapabilities,
    UnslothTrainerBackend,
    UnslothTrainingConfig,
)
from .runtime import detect_trainer_runtime, trainer_runtime_summary


def cmd_check(args: argparse.Namespace) -> int:
    """Check backend availability and capabilities."""
    print("=" * 60)
    print("PFE Training Backend Check")
    print("=" * 60)

    # Hardware detection
    hardware = HardwareProfile.detect()
    print("\n📊 Hardware Profile:")
    print(f"  Platform: {hardware.platform}")
    print(f"  Machine: {hardware.machine}")
    print(f"  Apple Silicon: {hardware.is_apple_silicon}")
    print(f"  CUDA Available: {hardware.cuda_available}")
    if hardware.cuda_version:
        print(f"  CUDA Version: {hardware.cuda_version}")
    print(f"  MPS Available: {hardware.mps_available}")
    print(f"  CPU Count: {hardware.cpu_count}")
    if hardware.memory_gb:
        print(f"  Memory: {hardware.memory_gb:.1f} GB")

    # Dependencies
    deps = DependencyProfile.detect()
    print("\n📦 Dependencies:")
    for name, available in deps.to_dict().items():
        status = "✅" if available else "❌"
        print(f"  {status} {name}")

    # Backend availability
    print("\n🔧 Backend Availability:")
    backends = {
        "MLX (Apple Silicon)": MLXBackendCapabilities.is_available(),
        "Unsloth (CUDA)": UnslothBackendCapabilities.is_available(),
        "PEFT (General)": deps.peft and deps.torch and deps.transformers,
        "Mock (Fallback)": True,
    }
    for name, available in backends.items():
        status = "✅" if available else "❌"
        print(f"  {status} {name}")

    # Recommendations
    print("\n💡 Recommendations:")
    selector = AutoBackendSelector(hardware=hardware, dependencies=deps)

    for train_type in ["sft", "dpo"]:
        result = selector.select_backend(train_type=train_type)
        print(f"  {train_type.upper()}: {result.selected_backend}")
        print(f"     Reason: {result.reason}")
        if result.warnings:
            for warning in result.warnings:
                print(f"     ⚠️  {warning}")

    print("\n" + "=" * 60)
    return 0


def cmd_recommend(args: argparse.Namespace) -> int:
    """Get detailed backend recommendation."""
    selector = AutoBackendSelector()

    result = selector.select_backend(
        train_type=args.train_type,
        base_model=args.base_model,
        backend_hint=args.backend,
    )

    print("=" * 60)
    print("Backend Recommendation")
    print("=" * 60)
    print(f"\nSelected Backend: {result.selected_backend}")
    print(f"Confidence: {result.confidence * 100:.0f}%")
    print(f"Reason: {result.reason}")
    print(f"Estimated Performance: {result.estimated_performance}")

    if result.requirements:
        print(f"\nRequirements:")
        for req in result.requirements:
            print(f"  - {req}")

    if result.alternatives:
        print(f"\nAlternative Backends:")
        for alt in result.alternatives:
            print(f"  - {alt}")

    if result.warnings:
        print(f"\nWarnings:")
        for warning in result.warnings:
            print(f"  ⚠️  {warning}")

    # Full report
    if args.verbose:
        print("\n" + "=" * 60)
        print("Full Report")
        print("=" * 60)
        report = selector.get_recommendation_report(args.train_type)
        print(json.dumps(report, indent=2, default=str))

    return 0


def cmd_mlx_info(args: argparse.Namespace) -> int:
    """Show MLX backend information."""
    print("=" * 60)
    print("MLX Backend Information")
    print("=" * 60)

    caps = MLXBackendCapabilities()
    info = caps.get_device_info()

    print(f"\nAvailable: {info['available']}")
    print(f"Device: {info['device']}")
    if info['memory_gb']:
        print(f"Memory: {info['memory_gb']:.2f} GB")

    if not info['available']:
        missing = caps.get_missing_dependencies()
        print(f"\nMissing Dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install mlx mlx_lm")

    return 0


def cmd_unsloth_info(args: argparse.Namespace) -> int:
    """Show Unsloth backend information."""
    print("=" * 60)
    print("Unsloth Backend Information")
    print("=" * 60)

    caps = UnslothBackendCapabilities()
    info = caps.get_device_info()
    version = caps.get_unsloth_version()

    print(f"\nAvailable: {info['available']}")
    if version:
        print(f"Version: {version}")
    print(f"Device: {info['device']}")
    if info['device_name']:
        print(f"Device Name: {info['device_name']}")
    if info['memory_gb']:
        print(f"Memory: {info['memory_gb']:.2f} GB")
    if info['cuda_version']:
        print(f"CUDA Version: {info['cuda_version']}")

    if not info['available']:
        missing = caps.get_missing_dependencies()
        print(f"\nMissing Dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall with: pip install unsloth")

    return 0


def cmd_runtime(args: argparse.Namespace) -> int:
    """Show runtime summary."""
    summary = trainer_runtime_summary()

    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("=" * 60)
        print("Trainer Runtime Summary")
        print("=" * 60)
        print(f"\nPlatform: {summary.get('platform_name')}")
        print(f"Machine: {summary.get('machine')}")
        print(f"Python: {summary.get('python_version')}")
        print(f"Runtime Device: {summary.get('runtime_device')}")
        print(f"Apple Silicon: {summary.get('apple_silicon')}")
        print(f"CUDA Available: {summary.get('cuda_available')}")
        print(f"MPS Available: {summary.get('mps_available')}")

        if 'capabilities' in summary:
            print("\nCapabilities:")
            for name, caps in summary['capabilities'].items():
                print(f"  {name}: {caps.get('artifact_format', 'N/A')}")

        if summary.get('notes'):
            print("\nNotes:")
            for note in summary['notes']:
                print(f"  - {note}")

    return 0


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        prog="pfe-trainer",
        description="PFE Training Backend CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s check                    # Check all backends
  %(prog)s recommend                # Get backend recommendation
  %(prog)s recommend --train-type dpo
  %(prog)s mlx-info                 # Show MLX backend info
  %(prog)s unsloth-info             # Show Unsloth backend info
  %(prog)s runtime                  # Show runtime summary
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check command
    check_parser = subparsers.add_parser(
        "check",
        help="Check backend availability and capabilities",
    )
    check_parser.set_defaults(func=cmd_check)

    # recommend command
    recommend_parser = subparsers.add_parser(
        "recommend",
        help="Get detailed backend recommendation",
    )
    recommend_parser.add_argument(
        "--train-type",
        choices=["sft", "dpo"],
        default="sft",
        help="Training type (default: sft)",
    )
    recommend_parser.add_argument(
        "--base-model",
        help="Base model name for compatibility checking",
    )
    recommend_parser.add_argument(
        "--backend",
        help="Preferred backend hint",
    )
    recommend_parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show full report",
    )
    recommend_parser.set_defaults(func=cmd_recommend)

    # mlx-info command
    mlx_parser = subparsers.add_parser(
        "mlx-info",
        help="Show MLX backend information",
    )
    mlx_parser.set_defaults(func=cmd_mlx_info)

    # unsloth-info command
    unsloth_parser = subparsers.add_parser(
        "unsloth-info",
        help="Show Unsloth backend information",
    )
    unsloth_parser.set_defaults(func=cmd_unsloth_info)

    # runtime command
    runtime_parser = subparsers.add_parser(
        "runtime",
        help="Show runtime summary",
    )
    runtime_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    runtime_parser.set_defaults(func=cmd_runtime)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    try:
        return args.func(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if hasattr(args, "verbose") and args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
