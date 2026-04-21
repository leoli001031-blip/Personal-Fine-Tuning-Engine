# Personal Finetune Engine (PFE) — 项目概览

> 版本：v0.0.1-draft | 更新日期：2026-04-10
>
> 说明：当前为文档草稿版本。Phase 0 结束后发布 v0.1.0-alpha，Phase 1 结束后发布 v0.2.0-beta。

## 一、项目定位

**一句话定义：** 一个让任何人都能基于自己的使用数据，在本地持续微调小模型的开源框架。

**核心价值主张：** 把"个性化"从 prompt 拼接升级到权重内化——用户的偏好、风格、习惯不再是塞进 system prompt 的文本，而是真正写入模型权重的能力。

**产品形态：** PFE 的核心交付物是一个本地个性化微调引擎，而不是单独的聊天 App。它提供 CLI、训练/评测管线、本地推理服务，以及一套行为信号接入协议；`Life Coach`、写作助手等是基于引擎构建的官方示例应用。

**与现有项目的差异化：**

| 项目 | 定位 | PFE 的差异 |
|------|------|-----------|
| Ollama | 本地模型推理 | PFE 管个性化微调，Ollama 管推理 |
| LM Studio | 模型管理 + 推理 GUI | PFE 聚焦"数据→微调→评测"管线 |
| Jan | 本地 AI 助手 | PFE 是引擎层，不是应用层 |
| Open Interpreter | 代码执行 Agent | 完全不同的方向 |

**开源策略：开源引擎，闭源产品。**
- 引擎（PFE）：MIT，简洁宽松，便于社区采用和商业使用
- 产品（Life Coach、桌面宠物等）：作为引擎的"官方示例应用"

## 二、核心能力概览

当前阶段，PFE 已验证的主链能力是 `collect -> curate -> train -> eval -> promote -> serve`。其中 `train / eval / serve` 属于核心闭环；`generate / distill / profile / route` 已有实现，但仍应视为 bootstrap、启发式或分析型能力，而不是完全成熟的默认路径。

PFE 提供七大核心模块，形成完整的个人微调闭环：

```
用户交互 → [Signal Collector] → [Data Curator] → [Trainer] → [Evaluator]
                                                       ↓
                                              [Adapter Store] ← 版本管理
                                                       ↓
                                        [Router] → [Inference Engine] → 推理服务
```

| 模块 | 职责 | 一句话描述 |
|------|------|-----------|
| Signal Collector | 信号采集 | 从用户交互中记录原始行为信号 |
| Data Curator | 样本构建 | 把原始信号转化为训练样本；冷启动生成/蒸馏当前主要承担 bootstrap 辅助作用 |
| Trainer | 微调训练 | 在本地执行 LoRA/QLoRA 微调（SFT + DPO） |
| Evaluator | 效果评测 | 支持 LLM-as-a-Judge 与 A/B 测试，量化微调效果 |
| Adapter Store | 版本管理 | 管理 adapter 的版本、切换、回滚 |
| Router | 推理路由 | 当前以关键词/规则做辅助路由，不应视为权威决策层 |
| Inference Engine | 推理引擎 | 模型加载、adapter 挂载、推理执行 |

## 三、用户体验目标

**3 条命令完成个性化微调：**

```bash
pfe init --base-model Qwen/Qwen2.5-3B-Instruct
pfe train --method qlora --epochs 3
pfe serve --port 8921
```

**完整工作流：**

```bash
# 1. 初始化项目
pfe init --base-model Qwen/Qwen2.5-3B-Instruct

# 2. 可选冷启动：生成 bootstrap 训练数据（当前以模板/启发式能力为主）
pfe generate --scenario life-coach --style "温和、非暴力沟通风格" --num 200

# 2b. 或显式走 Teacher / Distillation 管线
# 注意：若未配置真实 teacher path，当前工作流仍可能回退为模板/合成样本输出
pfe distill --teacher-model gpt-4o --scenario life-coach --style "温和、非暴力沟通风格" --num 200

# 3. 或手动添加 / 从历史导入
pfe add-sample --input "我和同事吵架了" --output "听起来你现在很沮丧..."
pfe import --format chatgpt-export --file conversations.json

# 4. 微调
pfe train --method qlora --epochs 3

# 5. 评测
pfe eval --base-model base --adapter latest --num-samples 20

# 6. 启动推理服务（兼容 OpenAI API 格式）
pfe serve --port 8921

# 7. 版本管理
pfe adapter list
pfe adapter rollback -1
```

## 四、目标用户

| 用户类型 | 使用场景 | 关注点 |
|---------|---------|--------|
| 独立开发者 | 为自己的应用集成个性化能力 | API 易用性、集成成本 |
| AI 爱好者 | 训练"懂自己"的本地模型 | 门槛低、效果可感知 |
| 研究者 | 探索个性化微调方法 | 可扩展性、评测框架 |
| 产品团队 | 在产品中嵌入个性化微调 | 稳定性、隐私合规 |

## 五、核心设计原则

1. **本地优先（Local-first）：** 用户数据默认只存本地；所有云功能默认关闭，只有在用户显式开启后才允许上传脱敏后的文本片段
2. **渐进式复杂度：** 3 条命令能跑通，深度定制也支持
3. **可插拔架构：** 每个模块都可以被替换或扩展
4. **推理兼容优先：** `pfe serve` 兼容 OpenAI Chat Completions，现有前端可零改动接入推理；若要开启个性化闭环，还需额外接入 PFE 的信号上报协议 / SDK。最小接入方式见 `docs/10-openai-closed-loop-integration.md`
5. **增量而非全量：** 每次微调基于上一版 adapter 继续训练，而非从头开始

## 六、隐私与云功能边界

PFE 支持 Teacher 数据生成、LLM-as-a-Judge 评测、云端路由等可选云能力，但这些能力**不属于默认路径**：

- 默认模式是 `strict_local`：只允许本地采集、本地训练、本地推理
- Teacher 模式、云端 Judge、云端 Router 默认关闭，需要用户显式开启
- 开启云能力时，优先对文本做脱敏处理，并记录到出境审计日志
- 如果上层应用没有接入 PFE 信号协议，PFE 只能提供“本地推理服务”，不能自动形成个性化训练闭环

## 七、LLM 蒸馏与 Judge 评测

PFE 在架构上支持把更强的 LLM 用作两类角色，但这些能力不属于默认本地闭环：

- **Teacher / Distiller：** 生成冷启动数据、重写弱样本、构造 DPO 的 `chosen/rejected`
- **Judge：** 对比基座模型与微调模型的输出，评估训练是否真正提升

设计原则：

- Teacher 和 Judge 是两个独立角色，默认不要复用同一套 prompt 与评测集
- 蒸馏数据必须保留来源信息（teacher model、prompt version、采样参数）
- Judge 评测必须优先使用独立 holdout 集，避免“训练过什么就评什么”

## 八、术语表

| 中文 | 英文 | 说明 |
|------|------|------|
| 适配器 | Adapter (LoRA Adapter) | 微调产出的轻量级权重文件，叠加在基座模型上 |
| 基座模型 | Base Model | 未经个性化微调的原始预训练模型 |
| 场景 | Scenario | 特定的使用场景模板（如 life-coach、writing-assistant） |
| 信号 | Signal (RawSignal) | 从用户交互中采集的原始行为数据 |
| 样本 | Sample (TrainingSample) | 经过处理、可用于训练的数据条目 |
| 监督微调 | SFT (Supervised Fine-Tuning) | 基于 instruction-output 对的标准微调方式 |
| 偏好对齐 | DPO (Direct Preference Optimization) | 基于 chosen-rejected 对的偏好学习方式 |
| 增量微调 | Incremental Fine-Tuning | 在已有 adapter 基础上继续训练，而非从头开始 |
| 回放缓冲 | Replay Buffer | 增量微调时混入的历史样本，防止灾难性遗忘 |
| 推理路由 | Routing | 根据请求复杂度决定走本地模型还是云端 API |
| 数据蒸馏 | Distillation | 用更强的 LLM 生成、重写或筛选数据，供较小模型训练 |
| 教师模型 | Teacher Model | 用于蒸馏数据的强模型 |
| 裁判模型 | Judge Model | 用于评估训练结果的模型 |
