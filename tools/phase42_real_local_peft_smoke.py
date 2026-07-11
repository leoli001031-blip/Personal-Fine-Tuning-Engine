#!/usr/bin/env python3
"""Run and evidence a real 12-step local Qwen PEFT training probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.adapter_store.quality import validate_adapter_artifact
from pfe_core.trainer.executors import _run_real_local_peft_training


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _examples() -> list[dict[str, Any]]:
    rows = [
        ("如何确认服务正在运行？", "先检查 PID、监听端口和最近日志，再报告真实状态。"),
        ("提交代码前做什么？", "先检查 git status 和 diff，运行 focused tests，再提交相关文件。"),
        ("训练失败应该怎么记录？", "保存命令、退出码、错误日志和产物目录，并将状态标为 blocked。"),
        ("如何区分模拟反馈？", "明确标记 simulated_usage，不把它写成 actual_user_feedback。"),
        ("什么时候可以 promote？", "只有独立 holdout 通过且人工复核后才提出 promote 建议。"),
        ("如何保护私密资料？", "只保存脱敏结构、hash 和计数，不提交原始私密正文。"),
        ("如何判断 adapter 有效？", "对 base 和 adapter 使用同一 holdout，比较原始输出与预设指标。"),
        ("用户纠正方向后怎么办？", "立即以最新目标为准，停止无关展开，并重新核对验收条件。"),
        ("测试通过等于产品可用吗？", "不等于；还需要真实运行、协议兼容和用户场景证据。"),
        ("上下文太长如何处理？", "计算真实 token 预算，保留最近上下文，并显式报告是否截断。"),
        ("流式接口如何结束？", "发送最终 finish_reason，然后发送 data: [DONE]。"),
        ("没有合格 adapter 时怎么服务？", "回退到 base，并把 adapter 状态明确标成 blocked 或 archived。"),
    ]
    return [
        {
            "sample_id": f"phase42-real-sft-{index:03d}",
            "instruction": instruction,
            "chosen": chosen,
            "rejected": None,
            "sample_type": "sft",
            "source": "phase42_runtime_integrity_probe",
            "actual_product_benefit_claim_allowed": False,
        }
        for index, (instruction, chosen) in enumerate(rows, start=1)
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        type=Path,
        default=REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "trainer_job_outputs" / "phase42-real-local-qwen25-0_5b-12step",
    )
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "demo"
            / "phase42-trustworthy-training-runtime-hardening"
            / "evidence-real-training"
        ),
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    base_model = args.base_model.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    evidence_dir = args.evidence_dir.expanduser().resolve()
    if not (base_model / "config.json").exists():
        raise SystemExit(f"local base model is unavailable: {base_model}")
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    if args.clean and evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    job_spec = {
        "backend": "peft",
        "execution_backend": "peft",
        "execution_executor": "peft",
        "executor_mode": "real_local",
        "ready": True,
        "dry_run": False,
        "recipe": {
            "training": {
                "method": "lora",
                "base_model_path": str(base_model),
                "base_model": str(base_model),
                "local_only": True,
                "epochs": 1,
                "max_steps": max(1, args.steps),
                "learning_rate": 0.0002,
                "seed": 42,
                "output_dir": str(output_dir),
            }
        },
        "audit": {"import_probe": {"ready": True, "missing_modules": []}},
        "training_examples": _examples(),
    }
    _write_json(evidence_dir / "training_manifest.json", job_spec)

    result = _run_real_local_peft_training(job_spec)
    real_execution = dict(result.get("real_execution") or {})
    artifact_dir = Path(str(real_execution.get("artifact_dir")))
    adapter_path = artifact_dir / "adapter_model.safetensors"
    validation = validate_adapter_artifact(
        artifact_dir,
        {"artifact_name": adapter_path.name, "artifact_format": "peft_lora"},
    )
    validation["sha256"] = _sha256(adapter_path) if adapter_path.exists() else None
    validation["parameters_updated"] = real_execution.get("parameters_updated")
    validation["steps"] = real_execution.get("steps")

    completed = (
        result.get("status") == "completed"
        and real_execution.get("success") is True
        and real_execution.get("parameters_updated") is True
        and int(real_execution.get("steps") or 0) >= max(1, args.steps)
        and validation.get("valid") is True
    )
    attempt = {
        "kind": "phase42_real_local_peft_training_attempt",
        "status": "completed" if completed else "failed",
        "real_training": completed,
        "base_model": str(base_model),
        "requested_steps": max(1, args.steps),
        "execution": real_execution,
        "adapter_validation": validation,
        "actual_product_benefit_claim_allowed": False,
        "auto_promotion_allowed": False,
    }
    _write_json(evidence_dir / "training_attempt.json", attempt)
    _write_json(evidence_dir / "adapter_validation.json", validation)
    _write_json(evidence_dir / "train_log.json", {"loss_history": real_execution.get("loss_history") or []})

    if not completed:
        raise SystemExit("Phase42 real local PEFT smoke failed integrity checks")
    print("PHASE42 REAL LOCAL PEFT SMOKE PASSED")
    print(f"base_model: {base_model}")
    print(f"steps: {real_execution.get('steps')}")
    print(f"initial_loss: {real_execution.get('initial_loss')}")
    print(f"final_loss: {real_execution.get('final_loss')}")
    print(f"parameters_updated: {real_execution.get('parameters_updated')}")
    print(f"adapter_path: {adapter_path}")
    print(f"adapter_sha256: {validation.get('sha256')}")
    print(f"tensor_count: {validation.get('tensor_count')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
