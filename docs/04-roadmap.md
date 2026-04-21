# PFE 开发路线图

> **提示：本文档为早期原始规划，最新开发状态请参阅 [`docs/07-development-roadmap-v2.md`](07-development-roadmap-v2.md)。**

## 阶段总览

```
Phase 0 (4周)          Phase 1 (8周)           Phase 2 (8周)
技术验证               产品闭环                 生态化
─────────────────────────────────────────────────────────────
generate→train→eval    采集→微调→使用循环       插件体系+多模型+社区
```

---

## Phase 0：技术验证（4 周）

**目标：** 证明"本地 LoRA 微调能产生可感知的风格变化"

| 周 | 任务 | 产出 | 负责模块 |
|----|------|------|---------|
| W1 | 搭建项目骨架 + 实现 `TeacherCurator` | 能用 GPT-4/Claude 批量生成场景数据 | Curator |
| W2 | 实现 `Trainer`（QLoRA on 3B） | 能在本地跑通微调，产出 adapter | Trainer |
| W3 | 实现 `AutoEvaluator` | LLM-as-a-Judge 能量化微调前后差异 | Evaluator |
| W4 | 端到端串联 + 效果调优 | 一条命令跑通 `generate → train → eval` | 全模块 |

### Phase 0 验收标准

- [ ] Life Coach 场景下，Judge 评测风格匹配度提升 > 20%（基线：基座模型 + 空 system prompt 的零样本输出）
- [ ] 基础回答质量不退化（quality_preservation > 0.8，基线同上）
- [ ] 3B 模型在 8GB 显存 / 16GB 统一内存的 Mac 上能跑通
- [ ] **同时测试 7B 模型**，确认 3B 的能力下限是否可接受（回应 review 第 2 点）
- [ ] 蒸馏样本保留完整 provenance（teacher model / prompt version / split / generation config）

### Phase 0 关键决策点

> **Gate：** 如果 3B 模型微调后在非暴力沟通场景的基本质量不达标，需要调整技术路线起点（上调到 7B 或更换基座模型）。先验证天花板，再规划产品。

### W1 详细任务

```
□ 初始化项目结构（pyproject.toml, pfe-core/, pfe-cli/）
□ 定义核心数据模型（RawSignal, TrainingSample, AdapterMeta, EvalReport）
  □ 为事件链路保留 `request_id / session_id / adapter_version`
□ 实现 TeacherCurator
  □ 场景模板设计（Life Coach 场景）
  □ 调用云端 API 批量生成对话数据（显式 opt-in）
  □ 生成结果质量筛选
  □ 记录 teacher model / prompt version / generation config
  □ 输出 SFT + DPO 两种格式
□ 实现 pfe generate CLI 命令
```

### W2 详细任务

```
□ 集成 unsloth / peft
□ 实现 TrainerConfig 配置体系
□ 实现 QLoRA 训练流程
  □ 数据加载 + tokenization
  □ LoRA 配置 + 训练循环
  □ 训练过程 loss 输出
  □ Adapter 保存（标准目录 + manifest）
□ Apple Silicon 兼容性测试（mlx-lm 优先，MPS + peft 作为 fallback）
□ 实现 pfe train CLI 命令
```

### W3 详细任务

```
□ 设计评测 prompt 模板（LLM-as-a-Judge）
□ 实现 AutoEvaluator
  □ 四维度评分：style_match, preference_alignment, quality_preservation, personality_consistency
  □ 本地 Judge 优先，云端 Judge 仅作为可选模式
  □ 基于 holdout / test split 做评测，禁止训练集直接评测
  □ 评测报告生成
□ 实现 EvalReport 输出格式（JSON + 终端可读）
□ 实现 pfe eval CLI 命令
```

### W4 详细任务

```
□ 端到端 pipeline 串联
□ 实现 AdapterStore 基础版（save/load/list/promote）
□ 效果调优（调整 LoRA 参数、训练轮数、学习率）
□ 3B vs 7B 对比实验
□ 撰写 Phase 0 技术验证报告
□ 决策：确认基座模型选型
```

---

## Phase 1：产品闭环（8 周）

**目标：** 从"手动跑脚本"变成"可用的工具"

| 周 | 任务 | 产出 | 负责模块 |
|----|------|------|---------|
| W1 | 实现 `ChatCollector` | 对话中自动采集行为信号 | Collector |
| W2 | 实现 `SignalScorer` + 信号提纯 | 信号评分与噪声过滤 | Curator |
| W3 | 实现 Data Curator 完整流程 | 信号 → 样本自动转换 + 去重/平衡 | Curator |
| W4 | 实现 Adapter Store + 增量微调 | 版本管理 + 在已有 adapter 上继续训练 | Store + Trainer |
| W5 | 实现 DPO 训练流程 | 偏好对齐训练支持 | Trainer |
| W6 | 实现 Inference Engine + `pfe serve` | 推理引擎 + OpenAI 兼容 API 服务 | Inference + Server |
| W7 | 实现 CLI 完整命令 + 端到端测试 | `pfe init/train/eval/serve` 全部可用 | CLI |
| W8 | 集成测试 + Bug 修复 + 文档 | 稳定可发布版本 | 全模块 |

### W1 详细任务

```
□ 实现 InteractionEvent 数据模型
□ 实现 ChatCollector
  □ 对话接受/拒绝/编辑/重新生成事件采集
  □ 保存 `request_id / session_id / adapter_version`
  □ 信号写入 SQLite（signals 表）
□ 实现 pfe signal 相关 CLI 命令（查看、导出）
□ 发布最小版 Signal SDK / 接入协议说明
□ 单元测试
```

### W2 详细任务

```
□ 实现 SignalScorer（评分策略）
□ 实现信号提纯逻辑
  □ 多信号交叉验证
  □ 时间窗口过滤（连续操作合并）
  □ 噪声标记（模糊信号标记为"待确认"）
□ 实现 pfe data review 命令（人工抽检）
□ 单元测试
```

### W3 详细任务

```
□ 实现完整的 Signal → TrainingSample 转换流程
□ 基于事件链路做 DPO 配对校验
□ 实现样本去重（基于 embedding 相似度）
□ 实现样本平衡（按场景/主题分桶）
□ 实现质量门槛过滤（score < 0.3 丢弃）
□ 实现蒸馏数据切分（train / val / test）
□ 输出 SFT + DPO 两种格式
□ 单元测试
```

### W4 详细任务

```
□ 实现 AdapterStore 完整版（save/load/list/rollback/prune）
□ 实现增量微调（Replay Buffer 混入历史样本）
□ 实现 TrainTrigger（自动触发微调）
□ 实现 AutoRollback（训练后自动评测 + 回滚）
□ 单元测试
```

### W5 详细任务

```
□ 集成 trl.DPOTrainer
□ 实现 DPO 训练配置（beta 参数等）
□ 实现 SFT → DPO 的推荐训练流程
□ DPO 训练效果验证
□ 单元测试
```

### W6 详细任务

```
□ 实现 InferenceEngine 核心接口
  □ transformers 后端（CUDA）
  □ mlx-lm 后端（Apple Silicon）
  □ llama.cpp 后端（CPU fallback）
  □ 后端自动选择逻辑
  □ 定义统一 artifact manifest 与导出流程
□ 实现 adapter 热切换（swap_adapter）
□ 实现 pfe-server（FastAPI）
  □ OpenAI Chat Completions API 兼容
  □ PFE 扩展端点（/pfe/adapters, /pfe/signal, /pfe/distill/run, /pfe/status 等）
  □ 流式响应支持
  □ 本机/公网访问的鉴权与安全策略
□ 集成测试
```

### W7 详细任务

```
□ 实现 pfe init 命令（项目初始化 + 模型下载）
□ 实现 pfe distill 命令（Teacher / Distillation 入口）
□ 完善 pfe train / eval / serve 命令
□ 实现 pfe adapter list / rollback 命令
□ 实现 `pfe adapter promote` 命令
□ 端到端测试：generate → train → eval → serve 完整流程
□ CLI 帮助文档和错误提示优化
```

### W8 详细任务

```
□ 全模块集成测试
□ Mac (Apple Silicon) + Linux (CUDA) 双平台验证
□ Bug 修复和性能优化
□ 编写 Quick Start 教程
□ 准备 v0.2.0-beta 发布
```

### Phase 1 验收标准

- [ ] 用户通过 CLI 能完成完整的"对话 → 采集 → 微调 → 使用"循环
- [ ] 推理 API 兼容 OpenAI 格式，任意支持 OpenAI API 的前端都能接入本地推理
- [ ] 接入 Signal SDK 或 `/pfe/signal` 后，现有应用可以跑通个性化闭环
- [ ] 增量微调不出现明显的灾难性遗忘（Replay Buffer 验证）
- [ ] 信号评分准确率 > 70%（人工标注 100 条样本作为 ground truth，对比 SignalScorer 的二分类结果：正样本/负样本是否与人工标注一致）
- [ ] `pfe serve` 在 Apple Silicon Mac（≥ 16GB）和 CUDA GPU（≥ 8GB）上均能正常运行
- [ ] 非本机访问管理接口时必须开启 API Key 鉴权
- [ ] Judge 评测默认使用 holdout / test split，且与训练集无重叠

### Phase 1 重点：信号提纯设计

> **回应 review 第 3 点：** "无感埋点"的信号提纯逻辑在 Phase 1 就开始设计，而非留到 Phase 2。

信号提纯策略：
1. **多信号交叉验证：** 单一信号不直接作为训练样本，需要多个信号互相印证
2. **时间窗口过滤：** 短时间内的连续操作合并为一个意图（如快速编辑多次 → 一次大幅编辑）
3. **噪声标记：** 对模糊信号（如复制）标记为"待确认"，不直接进入训练集
4. **人工抽检机制：** `pfe data review` 命令，让用户审核自动标注的样本

### Phase 1 冷启动方案

> **回应 review 第 4 点：** Day 1 价值方案。详细设计见 [03-module-design.md → Teacher 模式](03-module-design.md)。
>
> 核心思路：有预训练场景 adapter → 直接加载；无 → Teacher 模式生成 200 条数据快速微调。后续交互数据持续积累，增量微调。

---

## Phase 2：生态化（8 周）

**目标：** 从"单人工具"变成"可扩展的开源框架"

| 周 | 任务 | 产出 |
|----|------|------|
| W1-2 | 插件体系实现 | 自定义 Collector / Curator / Evaluator |
| W3-4 | 更多基座模型支持 | Llama、Mistral、Phi、Gemma |
| W5 | Router 实现 | 规则路由 + 简单置信度路由 |
| W6 | A/B Test Evaluator + 长期追踪 Dashboard | 用户偏好收集 + 趋势可视化 |
| W7 | 示例应用：Writing Assistant | 第二个官方示例 |
| W8 | 文档、教程、社区基础设施 | 完整文档站 + Contributing Guide |

### Phase 2 验收标准

- [ ] 第三方开发者能通过插件机制扩展所有核心模块
- [ ] 支持 ≥ 4 个基座模型家族
- [ ] Router 在测试集上的路由准确率 > 80%
- [ ] 有 ≥ 2 个完整的示例应用
- [ ] 文档覆盖所有公开 API

---

## 里程碑与发布计划

| 里程碑 | 时间点 | 发布内容 |
|--------|--------|---------|
| **v0.1.0-alpha** | Phase 0 结束（第 4 周） | 技术验证通过，发布技术博客验证社区关注度 |
| **v0.2.0-beta** | Phase 1 结束（第 12 周） | CLI 可用，完整微调闭环，首个可用版本 |
| **v1.0.0** | Phase 2 结束（第 20 周） | 插件体系、多模型支持、完整文档 |

### 开源发布策略

1. **Phase 0 结束：** 先发技术博客，验证社区关注度，再决定后续投入力度
2. **Phase 1 结束：** 正式开源，附带完整的 Life Coach 微调教程作为 showcase
3. **Phase 2 结束：** 1.0 正式版，完整生态

### README 核心卖点

> "3 条命令让你的本地模型学会你的风格"
> ```
> pfe init → pfe train → pfe serve
> ```
