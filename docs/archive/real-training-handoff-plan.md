# Real Training Handoff Plan

更新时间：2026-04-29

这份计划用于后续模型继续开发真实训练能力，重点是 Apple Silicon / MLX 路径的稳定性。当前策略不是移除真实训练，而是把真实训练变成显式开启、可预检、可隔离、可诊断的能力。

## 当前事实

- 默认状态下真实训练必须被阻断，除非 `PFE_REAL_TRAINING=1`。
- Apple Silicon 上的 MLX / Metal 失败可能直接触发 Python 进程级退出，父进程不能直接跑 GPU smoke 或真实训练。
- `mlx`、`peft`、`unsloth` 的真实训练入口应通过子进程隔离。
- 测试和普通 agent 执行不得触发真实 MLX / PEFT / Unsloth 训练。
- 所有命令优先使用项目虚拟环境：`.venv/bin/python` 和 `.venv/bin/pfe`。

## 总体目标

把训练链路整理成四层：

1. Planning：选择 backend、生成训练计划，不导入重型训练框架。
2. Preflight：检查环境、数据、模型路径和输出目录，不在父进程触发 GPU 执行。
3. Isolated execution：真实训练只在 materialized 子进程里执行。
4. Reporting：无论成功、失败、abort、timeout，都能落盘日志和结构化 diagnostics。

## 开发约束

- 不要在父进程中直接 import 后立即执行 MLX GPU 操作。
- 不要绕过 `PFE_REAL_TRAINING`。
- 不要让生成的 `trainer_job.py` 再次 materialize 自己。
- 不要把 `mlx_output/`、`subprocess_training_jobs/`、`trainer_job.py`、`trainer_job.json` 这类运行产物提交进仓库。
- 不要并发运行多个 pytest；先跑窄范围测试，再逐步扩大。

## 可执行任务

### RT-1：稳定真实训练边界

Owner：训练 runtime

文件范围：

- `pfe-core/pfe_core/trainer/real_execution.py`
- `pfe-core/pfe_core/trainer/runtime_job.py`
- `tests/test_training_real_execution_gate.py`

验收：

- `PFE_REAL_TRAINING=0` 时 `mlx` / `peft` / `unsloth` / `dpo` 都返回 blocked。
- `PFE_REAL_TRAINING=1` 时 `mlx` / `peft` / `unsloth` 先 preflight，再进入子进程。
- 子进程内看到 `PFE_TRAINING_SUBPROCESS=1` 后直接执行 backend，不再递归 materialize。

### RT-2：补齐 DPO 隔离策略

状态：已完成第一版。DPO 会先在父进程构建 preference pairs，再走 preflight 和子进程隔离；后续仍需做真实 DPO 训练样例验证。

Owner：DPO trainer

文件范围：

- `pfe-core/pfe_core/trainer/runtime_job.py`
- `pfe-core/pfe_core/trainer/executors.py`
- `tests/test_dpo_executor_unit.py`
- `tests/test_training_real_execution_gate.py`

验收：

- DPO 在父进程中可以先从 signals 构建 preference pairs。
- 构建完有效样本后，真实 DPO 训练进入子进程。
- 无样本、依赖缺失、模型路径缺失时返回 blocked 或 failed diagnostics，而不是崩主进程。

### RT-3：Apple Silicon MLX 最小真实训练

状态：已完成成功闭环。当前机器 16GB 内存无法直接训练本地 `models/Qwen3-4B`，preflight 会以 `insufficient_memory` 阻断；换用 `mlx-community/Qwen2.5-0.5B-Instruct-4bit` 下载到 `models/Qwen2.5-0.5B-Instruct-4bit` 后，1 条样本 / 1 step 的 MLX 子进程训练已成功产出 adapter。

Owner：MLX backend

文件范围：

- `pfe-core/pfe_core/trainer/mlx_backend.py`
- `pfe-core/pfe_core/trainer/preflight.py`
- `tests/test_trainer_real_execution.py`

验收：

- 使用本地小样本和明确 base model 路径跑一次最小 MLX 训练；本地相对模型路径必须在进入子进程前解析为绝对路径。
- 失败时 `diagnostics.json` 至少包含 `returncode`、`signal_name`、`failure_category`、`stdout_log`、`stderr_log`。
- 如果出现 `SIGABRT`，父进程仍然正常返回 failed 状态；Metal insufficient memory 应归类为 `killed_oom`，并在后续 preflight 中尽量提前 blocked。

已验证命令：

```bash
PFE_REAL_TRAINING=1 .venv/bin/python - <<'PY'
from pathlib import Path
from pfe_core.trainer.real_execution import run_backend_in_subprocess

model = "models/Qwen2.5-0.5B-Instruct-4bit"
output_dir = Path("trainer_job_outputs/rt3-mlx-qwen05-success").resolve()
job = {
    "backend": "mlx",
    "execution_executor": "mlx",
    "base_model": model,
    "output_dir": str(output_dir),
    "timeout_seconds": 90,
    "training_examples": [{"instruction": "Say ping.", "output": "pong"}],
    "recipe": {
        "training": {
            "base_model": model,
            "epochs": 1,
            "max_seq_length": 64,
            "learning_rate": 1e-5,
            "output_dir": str(output_dir / "mlx_output"),
        },
        "peft": {"lora_config": {"r": 2, "lora_alpha": 4, "lora_dropout": 0.0}},
    },
}
print(run_backend_in_subprocess(job, backend="mlx", dry_run=False))
PY
```

结果：`status=completed`、`runner_status=completed`，产物包含 `mlx_output/adapters/adapters.safetensors`、`training_job_result.json`、`diagnostics.json`、stdout/stderr log。CLI 临时 workspace 也已通过 `pfe train --real-local --backend mlx --base-model models/Qwen2.5-0.5B-Instruct-4bit --epochs 1`，产出 `adapter_model.safetensors` 和 `adapter_manifest.json`。

### RT-4：CLI 和文档收口

状态：已完成第一版。`pfe train` / `pfe dpo` 已提供 `--backend`、`--dry-run`、`--real-local` 语义；后续可继续打磨文案和更完整的端到端测试。

Owner：CLI / docs

文件范围：

- `pfe-cli/pfe_cli/main.py`
- `docs/guides/dpo-training.md`
- `docs/05-tech-and-risk.md`

验收：

- `pfe train --dry-run` 永远不触发真实训练。
- `pfe train --real-local` 或环境变量显式开启真实训练。
- 帮助文案解释 Apple Silicon 自动 MLX 是规划选择，不等于默认执行真实训练。

## 推荐验证命令

```bash
.venv/bin/python -m ruff check \
  pfe-core/pfe_core/trainer/backends.py \
  pfe-core/pfe_core/trainer/executors.py \
  pfe-core/pfe_core/trainer/mlx_backend.py \
  pfe-core/pfe_core/trainer/runtime_job.py \
  pfe-core/pfe_core/trainer/service.py \
  pfe-core/pfe_core/trainer/preflight.py \
  pfe-core/pfe_core/trainer/real_execution.py \
  tests/conftest.py \
  tests/test_trainer_executor_recipe.py \
  tests/test_trainer_runtime.py \
  tests/test_training_real_execution_gate.py
.venv/bin/python -m pytest -q tests/test_training_real_execution_gate.py --tb=short
.venv/bin/python -m pytest -q tests/test_trainer_runtime.py tests/test_trainer_executor_recipe.py tests/test_dpo_executor_unit.py --tb=short
PFE_REAL_TRAINING=0 .venv/bin/pfe train --dry-run --backend mlx --base-model models/Qwen3-4B
```

全仓库 `ruff check pfe-core pfe-cli pfe-server tests` 仍有历史 lint 债，先不要把这件事混进真实训练隔离提交；后续按 `software-refactor-plan.md` 的 R4 单独处理。

## 当前优先级

1. 先稳定真实训练边界和测试安全。
2. 再补 DPO 子进程隔离。
3. 最后跑 Apple Silicon MLX 的最小真实训练闭环。
