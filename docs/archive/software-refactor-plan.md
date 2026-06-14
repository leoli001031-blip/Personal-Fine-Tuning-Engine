# Software Refactor Plan

更新时间：2026-05-02

这份计划用于把 PFE 从“功能已经堆出来”整理成“模块边界清晰、真实训练可控、CLI/API 行为一致、测试可渐进验证”的结构。重构按切片推进，每一阶段都要保持可运行、可回滚、可验证。

## 重构原则

- 不做一次性全仓库大改；每个阶段只改一个主要边界。
- 先修运行时风险，再修风格债。
- 真实训练永远显式开启，默认不触发。
- 每一阶段至少有定向测试通过，再进入下一阶段。
- 大量历史 lint 债单独排期，不混进训练功能重构。

## R1：训练执行边界

状态：基本完成，继续随真实训练最小闭环补验证。

目标：

- 统一 `PFE_REAL_TRAINING`、`PFE_TRAINING_SUBPROCESS`、`real_local` 的语义。
- `mlx`、`peft`、`unsloth`、`dpo` 真实训练都走 preflight + materialized 子进程。
- Service、CLI、子进程三条入口行为一致。

主要文件：

- `pfe-core/pfe_core/trainer/real_execution.py`
- `pfe-core/pfe_core/trainer/preflight.py`
- `pfe-core/pfe_core/trainer/runtime_job.py`
- `pfe-core/pfe_core/trainer/executors.py`
- `pfe-core/pfe_core/trainer/service.py`

验收：

- `PFE_REAL_TRAINING=0` 时真实 backend 全 blocked。
- `--real-local` / `real_local=True` 会把真实训练意图传到 materialized 子进程。
- 子进程不会递归 materialize 自己。
- DPO 无样本、缺依赖、缺模型路径时安全 blocked/failed。

## R2：CLI 命令边界

状态：进行中。

目标：

- 把 `pfe-cli/pfe_cli/main.py` 中训练、DPO、serve、console、status 的实现逐步拆成小模块。
- 命令参数解析、环境变量临时覆盖、handler 调用、输出格式化分离。
- `--dry-run`、`--real-local`、`--backend` 在 train / dpo 中语义一致。

建议拆分：

- 已完成第一段：`pfe_cli/training_commands.py` 承接 `pfe train` / `pfe dpo` 注册与 handler 编排。
- 已完成第二段：`pfe_cli/runtime_commands.py` 承接 `pfe serve` / `pfe console` / `pfe status` 注册与 handler 编排。
- 已完成第三段：`pfe_cli/operations_commands.py` 承接 `trigger` / `daemon` / `candidate` / `eval-trigger` / `collect` 命令组。
- 已完成第四段：`pfe_cli/utility_commands.py` 承接 `doctor` / `dashboard` / `boot` / `profile` / `scenario` / `route` / `data` 命令组。
- 已完成第五段：`pfe_cli/doctor_formatting.py` 承接 doctor readiness formatter/helper，`main.py` 保留兼容 wrapper。
- 已完成第六段：`pfe_cli/console_routing.py` 承接 console slash-command routing 和 compact summary helper，`main.py` 保留兼容 wrapper。
- 已完成第七段：`pfe_cli/status_formatting.py` 承接 status matrix 前处理和 cached training snapshot 合并，`main.py` 保留兼容 wrapper。
- 已完成第八段：`pfe_cli/operations_formatting.py` 承接 operations dashboard / alert / event stream / console digest / ops attention formatter，`main.py` 保留兼容 wrapper。
- 已完成第九段：`pfe_cli/serve_formatting.py` 承接 serve result / serve preview matrix formatter，`main.py` 保留兼容 wrapper。
- 已完成第十段：`pfe_cli/result_formatting.py` 承接 train / eval result matrix formatter，legacy formatter 暂留 `main.py`。
- 已完成第十一段：`pfe_cli/operations_history_formatting.py` 承接 candidate / queue / worker / daemon history 与 timeline formatter，`main.py` 保留兼容 wrapper。
- 已完成第十二段：`pfe_cli/daemon_formatting.py` 承接 daemon health / heartbeat / lease / stale / alerts formatter，`main.py` 保留兼容 wrapper。
- 已完成第十三段：`pfe_cli/console_io.py` 承接 console chat text / transcript append / snapshot payload / line editing / history navigation / raw input helper，`main.py` 保留兼容 wrapper。
- 已完成第十四段：`pfe_cli/console_surface.py` 承接 console help / settings / compact status surface helper，`main.py` 保留兼容 wrapper。
- 已完成第十五段：`pfe_cli/console_actions.py` 承接 console focus action mapping / shortcut hint helper，`main.py` 保留兼容 wrapper。
- 已完成第十六段：`pfe_cli/workflow_commands.py` 承接 `generate` / `distill` / `eval` 命令注册与 handler 编排。
- `pfe_cli/runtime_env.py`

下一步：

- 将 train / dpo legacy 输出格式化逐步迁出 `main.py`，但要保留现有格式化测试。
- 继续拆分 `main.py` 中剩余 legacy train / eval 输出格式化与状态 legacy formatter。
- 再做高风险 lint 切片，不把 400+ 历史 lint 一次性混进 CLI 拆分。

## R3：Pipeline / Operations 拆分

状态：进行中。candidate / queue / runner helper 三个纯逻辑切片已完成，`PipelineService` 保留兼容方法。

目标：

- 将 `pipeline.py` 中 operations、candidate、queue、eval/promote、signal summary 分离。
- 优先拆出纯函数和状态 summary，减少 7000+ 行单文件的交叉依赖。
- 先修运行时 bug 类 lint：`F821`、`F822`、`F601`。

已完成：

- `pfe_core/pipeline_candidate.py`：candidate history entry、history summary、timeline stage、timeline payload。
- `tests/test_pipeline_candidate_helpers.py`：锁定拆出 helper 的纯函数行为。
- `pfe_core/pipeline_queue.py`：train queue item history、review summary、review policy summary、queue history payload。
- `tests/test_pipeline_queue_helpers.py`：锁定 queue helper 的纯函数行为。
- `pfe_core/pipeline_runner.py`：worker runner / daemon history、timeline、状态摘要 helper。
- `tests/test_pipeline_runner_helpers.py`：锁定 runner / daemon helper 的纯函数行为。
- `pfe_core/pipeline_operations.py`：operations event 分类、generic monitor summary、ordered actions helper。
- `tests/test_pipeline_operations_helpers.py`：锁定 operations helper 的纯函数行为。

下一步：

- 继续拆 operations dashboard / console summary 中的 payload 组装函数。
- 最后处理 `Sequence` / `logger` / `snapshot` 等未定义变量类 lint。

## R4：Lint Stabilization

状态：已完成。

目标：

- 不追求一次修完所有 ruff。
- 按风险顺序修：
  1. `F821` / `F822` / `F601`
  2. `F841`
  3. `F401`
  4. tests 的 `E402`

验收：

- 每个模块批次有对应测试。
- 不把纯格式化与行为重构混在一个提交。

已完成：

- `pipeline.py` 的 `F601` 重复 key 已清理。
- `pipeline.py` 的 `F821/F822` 高风险未定义引用已清理：补齐 `Sequence`、模块 logger，并移除不可达 `return snapshot`。
- `pipeline.py` 的 `F401` 未用 import 已清理。
- `pipeline.py` 的 `F841/F402` 已清理；该文件默认 `ruff check` 通过。
- 全仓库 `F821/F822/F601` 已清理；剩余主要是测试 `E402`、未用 import、未用局部变量等低风险 lint 债。
- 全仓库 `F811/F402/F541/F841` 已清理。
- 全仓库 `F401` 已清理；`pfe_core.__init__` 的 router 公开导出已补齐，DPO/MLX 依赖探测改为轻量 `find_spec`，降低仅探测依赖时触发重库初始化的风险。
- tests 的 `E402` 已清理：重复的 per-file `sys.path` 注入块已删除，统一保留 `tests/conftest.py` 作为测试路径入口。
- 默认 `ruff check pfe-core pfe-cli pfe-server tests` 已通过。
- 完整测试 `pytest tests` 已通过：1027 passed。

## R5：真实训练最小闭环

状态：已完成 Apple Silicon / MLX 最小闭环；DPO 隔离闭环已由现有真实 DPO 测试覆盖；Linux/CUDA 的 PEFT/Unsloth 已补可执行验证入口，后续剩实机回填结果。

目标：

- Apple Silicon 上跑一次最小 MLX 真实训练。
- Linux/CUDA 路径后续跑 PEFT/Unsloth 最小训练。
- DPO 使用小 preference pair 数据跑一次 dry-run + preflight + 子进程失败诊断。

验收：

- 成功时 adapter 产物可被 manifest 识别。
- 失败时 diagnostics 包含 `returncode`、`signal_name`、`failure_category`、stdout/stderr log。

已验证：

- Apple Silicon runtime：`Darwin arm64`，`runtime_device=mps`，`mlx=0.31.1`，`mlx_lm=0.31.2`。
- 原始 MLX 隔离子进程：使用本地 `models/Qwen2.5-0.5B-Instruct-4bit`、1 条样本、1 step，产出 `mlx_output/adapters/adapters.safetensors`、`training_job_result.json`、`diagnostics.json`、stdout/stderr log。
- CLI 临时 workspace：`pfe generate --num 8` 后执行 `pfe train --backend mlx --real-local --base-model /Users/zcc/Desktop/PFE/models/Qwen2.5-0.5B-Instruct-4bit --epochs 1 --train-type sft`，产出 `adapter_model.safetensors`、`adapter_manifest.json`、`training_meta.json`、`diagnostics.json`。
- CLI 摘要第一版已收口：训练预览区分 `execution_intent=planned/dry_run/real_local`，结果摘要会显示真实 `execution_backend` 和 runner 状态，避免 MLX 成功训练后显示 `backend=unknown`。
- 完整测试已覆盖 DPO 真实最小路径：`tests/test_trainer_dpo_real.py`。
- Linux/CUDA 验证入口已补：`tools/verify_cuda_real_training.py` 支持 `--backend peft|unsloth|all`，默认 dry-run，显式 `--run` 后走真实训练 gate、preflight、materialized 子进程和 diagnostics；使用说明见 `docs/guides/cuda-real-training-validation.md`。

后续：

- 在 Linux/CUDA 环境执行 `tools/verify_cuda_real_training.py --backend all --run`，回填 PEFT/Unsloth 最小训练结果。
- 根据 Linux/CUDA 实机结果补齐 PEFT/Unsloth 诊断样例和文档记录。
