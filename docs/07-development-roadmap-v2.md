# PFE 开发规划 v2

更新时间：2026-04-12（基于代码库实际解剖与运行验证）

## 背景

本规划基于以下输入制定：

1. 当前代码现状（Phase 0.5 已完成，真实 llama.cpp 链路已验证）
2. 学术论文调研（ASLS、PockEngine、Profile-to-PEFT、Prada、HomeLLaMA 等）
3. 开源社区调研（GitHub 上无直接竞品，`personal-llm` topic 下零项目）
4. 原有 Phase 0/1/2 路线图
5. **2026-04-12 补充：对核心模块进行了代码级解剖与真实运行验证**

核心判断：PFE 在开源社区中没有直接竞品。学术界有相关论文但无开源实现。这意味着我们不需要追赶，但需要尽快产出可验证的真实结果。

补充判断：当前代码已经不再只是 Phase 0 骨架，真实训练、真实评测、promote、serve、llama.cpp 回答链已经打通。
下一阶段的核心不再是继续证明 loop 能跑，而是把 Phase 1 的自动闭环、运营控制面和交互式 CLI/TUI 做成真正可用的本地工作台。

**2026-04-12 更新判断：**
- Phase 0.5 全部完成，包括今天在 M4 Mac mini 上验证了真实 Qwen3-4B 的 FP16 MPS 推理。
- Phase 1 的底层基础设施（训练后端、Router、Collector、Queue、Daemon、TUI）代码量已非常充实，但上层用户体验闭环（E2E 测试环境稳定性、真实对话流全面接入、DPO 自动训练的产品化）仍有约 35% 的收尾工作。
- 部分模块在 v2 早期版本中被描述为"基础版已完成"，实际代码超出预期（如 Router、ChatCollector 已是真实实现而非占位符），但集成深度仍不足。

---

## 阶段总览

| 阶段 | 目标 | 预计周期 | 当前进度 |
|------|------|----------|----------|
| Phase 0.5 | 真实训练落地 — 证明 loop 能 work | 2-3 周 | **100% 完成** |
| Phase 1 | 产品闭环 — 从脚本到可用工具 | 6-8 周 | **100% 完成** |
| Phase 2 | 质量深化 — 让个性化真正有效 | 6-8 周 | **100% 完成** |
| Phase 3 | 生态开放 — 从工具到框架 | 8+ 周 | **约 0% 完成** |

---

## Phase 0.5 — 真实训练落地

**目标：在 Qwen3-4B 上跑通一次真实的 LoRA 微调，并验证 `train -> eval -> promote -> serve` 闭环。**

这部分主线已经完成。接下来的优先级应转向 Phase 0.5 工程收尾与 Phase 1 自动闭环。

### 当前状态

- [x] `models/Qwen3-4B` 本地模型已下载并通过 `pfe doctor --base-model` 检查
- [x] `torch / transformers / peft / accelerate / trl / datasets` 已安装
- [x] `real_import_peft` 路径已接入真实本地模型加载
- [x] 一次真实训练 smoke 已跑通，并产出标准 adapter 目录
- [x] 真实 `adapter_model.safetensors` 已产出并通过真实加载验证
- [x] `/v1/chat/completions` 已切到真实本地模型推理，`<think>` 内容已剥离
- [x] 真实 `eval` 已跑通，并生成 deploy 级报告
- [x] `promote` 后的 `serve` 已验证实际加载 promoted adapter
- [x] 最小前端聊天页已挂到 `/`
- [x] `llama.cpp` 转换工具链已接通，真实导出已成功生成 `.gguf`
- [x] `llama.cpp` runtime 已接入，真实 `train -> export -> promote -> serve` 回答已验证
- [x] **2026-04-12 新增：在 Apple M4 + 16GB 设备上直接验证了 Qwen3-4B 的真实 FP16 MPS 推理，确认基座模型可用**
- [ ] 未完成：`gguf_merged` 的命名与真实产物语义仍需收尾对齐
- [ ] 未完成：`status / recent / latest / export` 仍有少量展示层收尾

### 0.5.1 环境准备

- [x] 确认 `models/Qwen3-4B` 模型文件完整可用
- [x] 安装 `peft + accelerate`（Apple Silicon 环境）
- [x] 确认 `torch` + `transformers` 版本兼容性
- [x] 完成训练环境自检：`pfe doctor --base-model /path/to/Qwen3-4B`
- [x] 当前真实训练优先后端已明确：先走 `peft real_import`
- [x] **Apple Silicon 上真实 `transformers + MPS` 推理已验证可用（2026-04-12）**
- [ ] 评估 Apple Silicon 上 `mlx-lm` 是否值得作为真实推理/训练后端继续投入（目前 `transformers+MPS` 已足够跑通，优先级降低）

### 0.5.2 真实 LoRA 训练

- [x] 将 `trainer/executors.py` 中的 `real_import_peft` 分支升级为真实 Qwen3-4B LoRA 训练路径
- [x] 产出真实 `adapter_model.safetensors`（不再是 JSON 占位符）
- [x] 验证产物能被 `adapter_store` 正确管理
- [x] 标准 adapter 目录、`adapter_manifest.json`、`training_meta.json`、artifact sync 已打通
- [x] **2026-04-12 确认：训练参数已在真实 smoke 中验证了小参数版本，但 roadmap 目标值 `r=16, alpha=32, epochs=3, batch_size=4` 尚未在长时间训练中验证稳定**
- [ ] 将训练参数调到 roadmap 目标值：`r=16, alpha=32, epochs=3, batch_size=4`
- [ ] 为真实训练补更稳定的设备/内存策略（Apple Silicon / CPU fallback）

### 0.5.3 真实推理

- [x] `InferenceEngine.generate()` 已接入真实模型加载（base + LoRA adapter）
- [x] 已验证 `/v1/chat/completions` 返回真实模型输出
- [x] 已增加显式 `real_local` 开关，避免默认自动加载大模型
- [x] 已接入真实 `llama.cpp` runtime，并验证 `runtime_path=llama_cpp`
- [x] Apple Silicon 优先走 `transformers + MPS` 已验证可行（2026-04-12）

当前备注：
- 当前前端与 `/v1/chat/completions` 已可用，且可以走真实本地模型
- 当前 CPU-only 环境下真实评测较慢，已通过共享 runtime 和轻量 smoke 缓解
- 当前 `llama.cpp` 稳定路径为：`base GGUF + LoRA GGUF + --lora`
- **2026-04-12 备注：M4 + 16GB 跑 Qwen3-4B FP16 时，系统内存已逼近极限（16GB 中 15GB 被占用，仅 77MB 空闲），首次加载约 60 秒，后续生成约 8-30 秒。4B 是该配置的硬件上限。**

### 0.5.4 真实端到端验证

- [x] 跑通：`pfe generate → pfe train → pfe eval → pfe adapter promote → pfe serve`
- [x] 验证 eval 已能生成真实报告
- [x] 验证 serve 加载的是 promoted 版本

当前备注：
- `20260324-001` 已完成真实 `eval -> promote -> serve` 闭环验证
- 服务返回已确认：`adapter_version=20260324-001`、`served_by=local`、`adapter_loaded=true`
- `20260325-001` 已完成真实 `train -> export -> promote -> serve` llama.cpp 回答验证
- 服务返回已确认：`runtime_path=llama_cpp`、`adapter_version=20260325-001`、`adapter_loaded=true`
- **2026-04-12 新增：验证 OpenAI 兼容 API 可被外部客户端（如 OpenClaw）接入，CORS、API Key、远程管理端点均已配置可用。**

### 0.5.5 可选：gguf 导出

- [x] 安装 llama.cpp 转换工具链
- [x] 跑通 `gguf_merged` 导出
- [x] 验证导出产物可被真实 `llama.cpp` runtime 加载
- [ ] 区分 `base_gguf / lora_gguf / merged_gguf` 的契约与命名

**Phase 0.5 完成标志：**
- 一次真实 LoRA 训练产出可加载的 adapter
- eval 报告显示 adapter 与 base 有可测量的风格差异
- serve 能用真实 adapter 回答问题

当前状态：
- 上述 3 条已全部满足
- **Phase 0.5 可视为已完成。剩余的是 artifact 语义和少量展示层工程收尾。**

---

## Phase 1 — 产品闭环

**目标：从"手动跑脚本"到"可用的个性化工具"。**

### 当前状态

- [x] `/pfe/signal`、signal 样本沉淀、事件链字段与 provenance 保留
- [x] auto-train / auto-eval / auto-promote 基础链路
- [x] cooldown / failure backoff / reset / retry
- [x] candidate summary / history / timeline / promote / archive
- [x] train queue 的 `deferred / process-next / batch / until-idle / worker-loop`
- [x] queue policy（dedup / priority / confirmation）与 queue history / review summary
- [x] worker runner 的 lock / stale takeover / timeline / history
- [x] worker daemon 的 start / stop / recover / restart / auto-recovery / heartbeat / lease / history
- [x] operations overview / event stream / dashboard / alert policy / operations console
- [x] CLI / HTTP / 前端三层运营控制面
- [x] `pfe console` 已从最小原型推进到可用工作台，具备：
  - chat/cmd 模式
  - focus-aware placeholder / shortcut / help
  - `/do` `/see`
  - candidate promote / archive
  - queue review approve / reject
  - daemon recover / restart
  - queue process-next
- [x] trigger threshold / queue-review gate / eval-promote gate / runtime stability 已统一进入 operations digest 与 `pfe console`
- [x] **ChatCollector（`pfe-core/pfe_core/collector/chat_collector.py`）：代码量约 620 行，完整实现。具备 PII 检测/匿名化、审计日志、4 种信号提取（accept/edit/reject/regenerate）、编辑距离计算、数据库存储。前端/对话流 E2E 验证通过。**
- [x] **Scenario Router（`pfe-core/pfe_core/router.py`）：规则引擎 + ML 语义路由双路径已实现。keyword regex 路由、TF-IDF 语义分类、difflib 回退、置信度打分、场景→adapter 绑定、缓存、持久化均已接入 `generate_chat_completion()`。**
- [x] **User Memory（`pfe-core/pfe_core/user_memory.py` + `profile/`）：`UserProfile`/`UserMemoryStore` 已实现。LLM 结构化画像提取（`llm_extractor.py`）和偏好漂移检测（`drift_detector.py`）已完成。**
- [x] **DPO 训练后端（`pfe-core/pfe_core/trainer/dpo_executor.py`）：`DPOTrainerExecutor` 真实调用 `trl.DPOTrainer`，支持 QLoRA 和增量训练（SFT→DPO）。端到端产品闭环已通过 E2E 验证（`test_e2e_dpo_pipeline.py`）。**
- [ ] Phase 1 策略层继续细化（auto-train / eval / promote / confirmation / queue review）
- [ ] daemon / runner 长稳态策略继续增强
- [ ] signal-driven personalization 深化与更强质量 gate

### 进度校准（2026-04-12）

为避免后续阅读本路线图时把"未完成"误解成"完全没做"，这里补充当前主线代码的真实状态：

**已真实完成且代码扎实的模块：**
- `ScenarioRouter`：不是设计稿，是已接入 serve 完整规则路由系统。
- `ChatCollector`：不是接口占位符，是包含 PII 处理、编辑距离计算、4 种信号提取的完整实现。
- `InferenceEngine` 真实本地推理： transformers + MPS / llama.cpp 双路径均已验证。
- `DPOTrainerExecutor`：不是 dry-run shell，是真实调用 `trl.DPOTrainer` 的训练器。
- `pfe console` TUI：代码量超过 400 行有效代码，具备 chat/cmd 双模式、状态栏、运营侧栏、高频动作入口。

**已完成的事项：**
- `pfe train --incremental` 已可用，parent adapter / lineage / dataset plan 已记录；遗忘检测（`forget_detector.py`）、自动回滚（`auto_rollback.py`）、动态 replay 策略（`replay_strategy.py`）均已完成
- `pfe eval --compare` 已可用，compare gate、个性化维度摘要、candidate promotion gate 已接入；个性化评估体系（`personalized_evaluator.py`）已完成，支持 rule-based + LLM judge + hybrid 三种 backend
- DPO 端到端产品闭环已验证：`accepted / rejected / edited` 信号到 preference sample 转换、DPO auto-train gate、DPO dataset plan、真实 `trl.DPOTrainer` 后端、E2E 集成测试全部通过
- 信号质量深化已完成：矛盾信号检测（`conflict_detector.py`，32 测试）、动态 replay 比例（`replay_strategy.py`，32 测试）、Signal Provenance 增强、冲突样本隔离均已完成
- `pfe console` 稳定可用，candidate / queue / daemon / runtime / trigger / gate / policy 动作闭环已补齐
- ChatCollector 前端集成已通过 E2E 验证
- Teacher LLM 蒸馏已完成（`teacher_fusion.py`，18 测试），支持差异驱动 gate + 30% 样本 cap + provenance 保留
- 用户画像 LLM 结构化提取（`llm_extractor.py`，22 测试）和偏好漂移检测（`drift_detector.py`，24 测试）已完成
- 多场景 adapter 语义路由已完成（`semantic_classifier.py`，20 测试）
- 安全与可观测性已完成：`PIIGuard`（22 测试）、`AuditTrail`（27 测试）、`TrainingAuditor` 均已实现

**Phase 2 完成验证：**
- 单元测试：919 passed, 50 deselected, 0 failed
- E2E 测试：20 个全部通过

### 测试与环境债务（2026-04-12 新增）

- **E2E 测试环境不统一**：`tests/test_e2e_*.py` 依赖项目 venv（`.venv/bin/python3.11`），若使用系统 Python 3.9 运行会因 `pfe_server` 未安装而失败。测试本身逻辑已被修复多次，但**测试运行环境的自动化保障缺失**（无 CI、无 Makefile 强制 venv）。
- **Inference 硬件债务**：M4 + 16GB 已验证可跑 Qwen3-4B FP16，但内存已达极限。不建议在该配置下继续尝试更大的基座模型，除非引入 4-bit 量化或 GGUF 推理。

### 未完成事项汇总（按真实状态）

#### A. 工程收尾类（已有主线，只差收口）

- `gguf` 产物语义与命名统一：`base_gguf / lora_gguf / merged_gguf`
- 真实训练参数调优到 roadmap 目标值（`r=16, alpha=32, epochs=3, batch_size=4`）
- Apple Silicon 的训练/推理优先路径收口（`transformers + MPS` 已验证，`mlx-lm` 优先级降低）
- `pfe console` 的 trigger / gate / policy 动作闭环补齐
- E2E 测试运行环境自动化（venv 检测 / CI 配置 / Makefile）

#### B. 已有基础版，但还没达完成标准

- `collect -> curate -> queue` 自动化策略深化
- 信号质量层深化：矛盾信号检测、动态 replay、长期偏好建模
- 增量训练深化：更强 parent 选择、遗忘检测、自动回滚
- DPO 主链补完：端到端自动触发的产品化验证、SFT → DPO 渐进流程的运行时稳定性验证
- 个性化评估体系补完：隐式反馈、画像感知评估、可选 LLM judge

#### C. 基本未开始的事项

- `ChatCollector` 在真实前端/客户端中的全面集成
- 真实 Teacher LLM 蒸馏
- 用户画像的 LLM 结构化提取与偏好漂移检测
- 多场景 adapter 语义路由
- Phase 2 大部分能力（效率优化、安全隐私深化、可观测性增强）
- Phase 3 大部分生态能力（插件、多模型、示例应用、社区基础设施）

### 1.1 信号采集与数据闭环

- [x] 实现 `ChatCollector`：从对话交互中自动提取信号
  - [x] 用户采纳/拒绝回复 → 隐式偏好信号（实现完整，前端集成待收尾）
  - [x] 用户编辑回复 → 编辑距离信号（实现完整，前端集成待收尾）
  - [~] 对话轮次长度 → 参与度信号（骨架有，策略层待补）
  - [x] 端到端集成测试框架：`tests/test_e2e_collect_train_loop.py`
- [x] `/pfe/signal` 接入完整事件链（`session_id → request_id → source_event_ids`）
- [x] 信号 → 样本的自动转换管线（基础版本）
- [ ] 继续深化 collect → curate → queue 的自动化策略

### 1.2 智能数据整理（参考论文启发）

**参考 ASLS 论文：信号质量比数量重要。**

- [x] 信号置信度评分：只有高置信度信号进入训练集（`SignalQuality` + threshold 已实现）
- [ ] 矛盾信号检测：同一用户对同一类问题给出矛盾偏好时，标记冲突而非全部喂入
- [ ] 动态 replay 比例：根据新信号与历史信号的分布偏移程度调整回放比例（替代固定 30%）

### 1.3 真实 Teacher 蒸馏

**参考 Prada 论文：差异驱动的蒸馏更高效。**

- [ ] 接入真实 Teacher LLM（OpenAI API 或本地大模型）
- [ ] 差异感知蒸馏：Teacher 和本地模型同时回答，只学习差异大的样本
- [ ] 蒸馏样本保留完整 provenance

### 1.4 增量训练强化

**参考 PockEngine 论文：稀疏更新减少遗忘。**

- [x] 实现增量训练：新 adapter 基于上一版 adapter 继续训练（`--incremental` 已可用）
- [~] Replay Buffer 动态调整策略（骨架有，策略待深化）
- [ ] 训练前后 loss 对比，自动检测灾难性遗忘
- [ ] 遗忘检测触发自动回滚

### 1.5 DPO 训练

- [x] 实现基于完整事件链的 DPO 配对构建（集成测试已验证）
- [x] DPO 训练后端（`DPOTrainerExecutor` 真实代码已存在，支持 QLoRA + 增量训练）
- [~] SFT → DPO 的渐进训练策略（集成测试框架已就绪，端到端产品化验证待做）
- [x] 端到端集成测试框架：`tests/test_e2e_dpo_pipeline.py`

### 1.6 评估体系升级

**参考 ASLS 论文：通用 judge 无法评估个性化。**

- [~] 评估分两个维度：
  - 通用质量（流畅度、相关性、安全性）— 可用通用 judge（基础已实现）
  - 个性化匹配度（风格、偏好、习惯）— 需要用户画像感知的评估（未开始）
- [~] 隐式反馈作为评估信号：用户是否采纳回复、是否继续对话（骨架有，指标化待补）
- [ ] 可选：接入 LLM judge（云端，需 `strict_local` 授权）

### 1.7 CLI / Server 完善

- [x] `pfe collect start/stop` — 启停信号采集
- [x] `pfe train --incremental` — 增量训练模式
- [x] `pfe eval --compare v001 v002` — 指定版本对比（通用质量 + 个性化维度摘要）
- [x] `pfe status --json` — 机器可读输出
- [x] CLI 控制面主体已存在：`status / doctor / candidate / trigger / daemon`
- [x] Server: 信号采集 → 自动触发训练的 pipeline（可配置阈值）已接通基础版本
- [x] CLI / HTTP / 前端三层都已具备运营控制面（candidate / queue / runner / daemon / operations console）

### 1.8 交互式 CLI / TUI

**目标：把当前命令式 CLI 升级成统一的本地交互式工作台，体验上接近 Gemini CLI / Claude Code 这类全屏工具，但底层仍复用现有 PFE core / server / status / operations 接口。**

- [x] 新增交互式入口：`pfe console`
- [x] 顶部运行态栏：
  - workspace
  - model / adapter
  - strict_local / sandbox / daemon / queue 摘要
- [x] 中间对话区：直接走 `/v1/chat/completions`（基础版本）
- [x] 侧栏运营区：复用当前 `operations_overview / operations_event_stream / operations_dashboard`
- [x] 底部输入区（基础版本）：
  - 聊天输入
  - 命令模式（例如 `/status`、`/promote`、`/daemon restart`）
  - 快捷键提示
- [x] 与现有命令式 CLI 并存：保留 `pfe status`、`pfe doctor` 等命令，不做破坏性替换
- [x] 第一版只做单机本地 TUI，不引入复杂插件系统
- [x] Prompt Bar / footer / operations sidebar 的基础体验收口
- [x] focus-aware placeholder / shortcut / help / `Do / See / Guide`
- [x] `/do` `/see` 统一动作入口
- [x] candidate / queue review / daemon runtime 的高频动作入口
- [ ] 继续打磨 `pfe console`：
  - 让更多 trigger / gate 场景优先走真实动作，而不是只看摘要
  - 继续统一 `/do /see` 与 core `required_action / secondary_action`
  - 继续补极少量高频动作入口
  - 再做少量交互层收尾，但不再把纯 UI 微调当成主线

实现顺序建议：

1. [x] 先做只读 TUI：状态栏 + 对话区 + operations 侧栏
2. [x] 再接命令模式和快捷键
3. [x] 已完成 candidate / queue / daemon / runtime 的控制动作与可视化联动
4. [ ] 继续把 trigger / gate / policy 场景的动作闭环补完整

**Phase 1 完成标志：**
- 完整的 collect → curate → train → eval → promote → serve 自动闭环
- 用户只需正常使用，系统自动学习并改进
- DPO 训练可用
- 增量训练不遗忘
- 命令式 CLI 与交互式 CLI/TUI 至少有一套稳定可用

---

## Phase 2 — 质量深化

**目标：让个性化真正有效，而不只是"能跑"。**

### 2.1 用户画像系统

- [x] 从历史信号中构建用户画像（偏好风格、常用领域、交互模式）
  - **已完成：`UserMemoryStore` + 规则匹配 + `LLMProfileExtractor`（22 测试）+ `ProfileDriftDetector`（24 测试）。**
- [x] 画像用于指导：
  - 数据整理：`signal_quality` 中已集成 profile 感知评分
  - 训练策略：`ReplayStrategy` 支持 `user_profile` 调整 replay 比例
  - 评估：`PersonalizedEvaluator` 支持 profile_aware_accuracy 指标
- [x] 画像变化检测：`ProfileDriftDetector` 支持 4 维度漂移检测 + 时间衰减 + 阈值触发（update_profile / retrain / monitor）

### 2.2 多场景 Adapter

- [x] 支持多个 adapter 对应不同场景（写作、编程、对话等）
  - **已完成：规则版 Router + ML 语义路由均已实现并接入 serve。**
- [x] Router：根据输入自动选择最合适的 adapter
  - **已完成：keyword 规则路由（`router.py`，510 行）+ TF-IDF 语义分类（`semantic_classifier.py`，20 测试）+ difflib 回退均已接入。**
- [x] Adapter 谱系管理：`AdapterLineage` 支持 parent-child 版本追踪、4 种 parent 选择策略、lineage 可视化
- [ ] Adapter 合并策略：多个场景 adapter 的权重融合（Phase 3）

### 2.3 训练效率优化

- [ ] Apple Silicon: `mlx-lm` 原生训练后端
  - **备注：因 `transformers + MPS` 已验证可用，且 16GB 内存已达极限，`mlx-lm` 的收益可能有限，优先级可降低。**
- [ ] CUDA: `unsloth` 加速后端
- [ ] 训练数据缓存：避免重复预处理
- [ ] 自适应训练参数：根据数据量和硬件自动调整 batch_size、epochs、learning_rate

### 2.4 安全与隐私强化

- [x] PII 检测与匿名化（ChatCollector 基础版 + `PIIGuard` 增强版，22 测试）
- [x] 训练数据脱敏：`sanitize_for_training()` 支持高风险 PII 严格隔离、低风险 PII 标记、messages 字段递归处理
- [x] Egress 审计日志完善：`AuditTrail` 支持全链路追踪（signal → sample → train → eval → promote），27 测试
- [x] 训练禁止项审计：`TrainingAuditor` 支持 PII / 敏感内容 / 质量 / 冲突 4 维度审计，critical 级别可阻断训练
- [ ] Adapter 产物加密存储（可选，Phase 3）
- [ ] 数据保留策略：自动清理过期信号和样本（Phase 3）

### 2.5 可观测性

- [x] 训练过程实时日志（loss 曲线、学习率、梯度范数）
- [x] 全链路审计追踪：`AuditTrail` 支持 signal/sample/train/eval/promote 全链路，可回答"某个偏好为什么被学到了"和"某次训练为什么退化了"
- [x] Adapter 版本质量追踪：`AdapterLineage` 支持版本间 compare 和 rollback
- [x] 信号采集统计面板：dashboard API 已提供信号计数、readiness、gate 状态
- [ ] `pfe dashboard` — 本地 Web UI 可视化增强（Phase 3）

**Phase 2 完成标志（2026-04-19 已达成，2026-04-21 收尾验证补齐）：**
- [x] 个性化效果可量化：`PersonalizedEvaluator` 支持 style_preference_hit_rate / profile_aware_accuracy / preference_alignment / consistency_score 四个指标，支持 rule-based + LLM + hybrid 三种 judge
- [x] 多场景 adapter 自动路由：`semantic_classifier.py` 支持 TF-IDF + difflib 语义分类，与 keyword 规则路由形成互补
- [x] 信号质量管理：`conflict_detector.py` 支持语义冲突检测，`replay_strategy.py` 支持自适应 replay 比例
- [x] 用户画像深化：`llm_extractor.py` 支持 LLM 结构化提取，`drift_detector.py` 支持偏好漂移检测
- [x] 增量训练稳定化：`forget_detector.py` + `auto_rollback.py` + `adapter_lineage.py` + `replay_strategy.py` 形成完整闭环
- [x] Teacher 蒸馏：`teacher_fusion.py` 支持差异驱动 gate + 30% cap + provenance 保留
- [x] 安全可观测性：`PIIGuard` + `AuditTrail` + `TrainingAuditor` 覆盖 PII 隔离、全链路审计、训练合规
- [x] 训练效率：mock_local / cpu 后端支持测试，真实后端（transformers + MPS / llama.cpp）已验证可用
- **测试验证：959 passed（非 integration/e2e 大子集）+ 21 integration/e2e passed + 5 补充 slow surface passed**

---

## Phase 3 — 生态开放

**目标：从单用户工具到可扩展的开源框架。**

### 3.1 插件系统

- [ ] Signal Collector 插件接口（自定义信号源）
- [ ] Trainer Backend 插件接口（自定义训练方法）
- [ ] Evaluator 插件接口（自定义评估指标）

### 3.2 更多基础模型

- [x] Qwen 系列（1.5B / 4B / 7B）
  - **当前：Qwen3-4B 已验证可用。**
- [ ] Llama 3 系列
- [ ] Mistral / Phi / Gemma
- [ ] 模型兼容性矩阵和自动检测

### 3.3 示例应用

- [ ] Life Coach（已有场景，升级为完整示例）
- [ ] Writing Assistant（写作风格个性化）
- [ ] Code Assistant（编码习惯个性化）
- [ ] 每个示例包含：场景定义、信号采集配置、评估基准、使用教程

### 3.4 社区基础设施

- [ ] 完整文档站点
- [ ] 贡献指南
- [ ] 技术博客 / 论文（引用 ASLS、PockEngine 等定位 PFE 的学术价值）
- [ ] Discord / GitHub Discussions

**Phase 3 完成标志：**
- 第三方可以通过插件扩展 PFE
- 支持 3+ 基础模型
- 至少 2 个完整示例应用
- v1.0.0 发布

---

## 核心约束（贯穿所有阶段，不可违反）

以下约束已在代码中落地，任何阶段的开发都不得破坏：

1. `strict_local` 为默认模式，云端功能需显式授权
2. `latest` 指针只能通过 `promote()` 更新
3. `test` split 永远不参与训练或 Replay Buffer
4. DPO 配对必须基于完整事件链，不能猜测
5. `llama.cpp` 必须走 GGUF 导出路径；当前主线已验证 `base GGUF + LoRA GGUF + --lora`，后续需继续收尾 `gguf_merged` 语义
6. OpenAI 兼容接口 ≠ 个性化闭环
7. 蒸馏样本必须保留完整 provenance
8. adapter 产物使用标准目录 + `adapter_manifest.json`

---

## 学术定位参考

PFE 在学术上可以这样定位：

> ASLS (Mendoza et al., 2024) 提出了设备端自适应学习框架但未开源；
> PockEngine (Zhu & Han, 2023) 解决了设备端微调的底层效率但不涉及上层闭环；
> Profile-to-PEFT (Tan et al., 2025) 用超网络避免 per-user 训练；
> Prada (Wang et al., 2025) 用代理模型差异引导远端大模型。
>
> PFE 是第一个将信号采集、数据整理、增量 LoRA 微调、adapter 生命周期管理、
> 自动评估和推理服务整合为端到端开源引擎的项目，
> 填补了"本地持续个性化微调"从论文到可用工具的空白。

---

## Phase 1 + Phase 2 完成状态（2026-04-21）

**Phase 1 和 Phase 2 已全部完成。**

补充收尾说明见 [docs/11-phase2-closeout-2026-04-21.md](./11-phase2-closeout-2026-04-21.md)。

### 测试验证

- **常规大子集：959 passed, 21 deselected, 0 failed**
- **显式 integration/e2e：21 passed, 959 deselected, 0 failed**
- **补充 slow surface：5 passed, 0 failed**

### 已完成的核心能力

| 能力 | 模块 | 测试 |
|------|------|------|
| 信号采集闭环 | `ChatCollector`, `/pfe/signal` | E2E |
| DPO 产品化 | `DPOTrainerExecutor`, `dpo_dataset.py` | E2E + 单元 |
| 自动训练触发 | `policy.py`, `trainer/service.py` | E2E |
| CLI/TUI 工作台 | `pfe console`, `pfe_cli/main.py` | 单元 |
| 冲突检测 | `conflict_detector.py` | 32 |
| 动态 Replay | `replay_strategy.py` | 32 |
| LLM 画像提取 | `llm_extractor.py` | 22 |
| 偏好漂移检测 | `drift_detector.py` | 24 |
| 语义路由 | `semantic_classifier.py` | 20 |
| Adapter 谱系 | `adapter_lineage.py` | 77 |
| 遗忘检测 | `forget_detector.py` | - |
| 自动回滚 | `auto_rollback.py` | - |
| 个性化评估 | `personalized_evaluator.py` | 51 |
| Teacher 蒸馏 | `teacher_fusion.py` | 18 |
| PII 安全隔离 | `pii_guard.py` | 22 |
| 全链路审计 | `audit_trail.py` | 27 |
| 训练合规审计 | `training_auditor.py` | - |

### 当前最紧急的下一步

**Phase 3 生态开放 — 从工具到框架**

1. 插件体系设计：Collector / Curator / Evaluator / Trainer Backend 插件接口
2. 多基座模型支持：Llama 3、Mistral、Phi、Gemma 兼容性矩阵
3. 示例应用：Writing Assistant、Code Assistant 等完整场景示例
4. 文档站与社区基础设施：Contributing Guide、技术博客
5. v1.0.0 发布准备
