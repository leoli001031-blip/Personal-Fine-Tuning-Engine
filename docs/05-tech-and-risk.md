# PFE 技术选型与风险评估

## 一、技术选型

| 组件 | 推荐方案 | 备选方案 | 选型理由 |
|------|---------|---------|---------|
| 微调框架 | `unsloth` (CUDA) / `mlx-lm` (Mac) | `peft` + `trl` | unsloth 速度快 2x；Mac 走 mlx-lm 原生路径 |
| 推理引擎 | `transformers` (CUDA) / `mlx-lm` (Mac) / `llama.cpp` (CPU) | `vLLM` | 按平台自动选择最优后端 |
| 基座模型 | `Qwen2.5-3B-Instruct` | `Phi-3-mini`, `Llama-3.2-3B` | 中文能力强，社区支持好，License 友好 |
| 数据存储 | SQLite | — | 零依赖，本地优先，够用 |
| API 格式 | 推理层 OpenAI 兼容 + PFE Signal 扩展 | — | 推理接入成本低，同时保留个性化闭环所需事件协议 |
| 包管理 | `uv` + `pyproject.toml` | `poetry` | 现代 Python 项目标准，速度快 |
| CLI 框架 | `typer` | `click` | 简洁，自动生成帮助文档 |
| API 框架 | `FastAPI` | `Flask` | 异步支持好，自动生成 OpenAPI 文档 |
| 量化 | `bitsandbytes` (CUDA) / `mlx` (Mac，无需额外量化) | `GPTQ`, `AWQ` | CUDA 走 4-bit QLoRA；Mac 走 mlx 原生精度（统一内存无需激进量化） |

### 基座模型对比（Phase 0 需验证）

| 模型 | 参数量 | 中文能力 | 推理能力 | 显存需求(4bit) | License |
|------|--------|---------|---------|---------------|---------|
| Qwen2.5-3B-Instruct | 3B | 强 | 中 | ~4GB | Apache 2.0 |
| Qwen2.5-7B-Instruct | 7B | 强 | 中上 | ~6GB | Apache 2.0 |
| Phi-3-mini-4k | 3.8B | 中 | 中上 | ~4GB | MIT |
| Llama-3.2-3B-Instruct | 3B | 弱 | 中 | ~4GB | Llama 3.2 |

> **Phase 0 决策：** 同时测试 3B 和 7B，确认能力下限后再锁定。

### Python 版本与依赖

```toml
[project]
name = "personal-finetune-engine"
requires-python = ">=3.10"

dependencies = [
    "typer>=0.9.0",
    "rich>=13.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "pydantic>=2.0",
    "torch>=2.1.0",
    "transformers>=4.36.0",
    "peft>=0.7.0",
    "trl>=0.7.0",
    "datasets>=2.16.0",
    "bitsandbytes>=0.41.0",
    "safetensors>=0.4.0",
    "openai>=1.0.0",          # 用于显式开启后的 Teacher / Judge 云调用
    "portalocker>=2.0.0",     # 跨平台文件锁
    "watchdog>=3.0.0",        # 文件系统监听（adapter 热切换）
]

[project.optional-dependencies]
unsloth = ["unsloth>=2024.1"]
mlx = ["mlx>=0.5.0", "mlx-lm>=0.5.0"]           # Apple Silicon 训练 + 推理
llama-cpp = ["llama-cpp-python>=0.2.0"]           # CPU 推理后端
eval = ["sentence-transformers>=2.2.0"]           # 用于样本去重
```

---

## 二、风险评估与应对

### 风险矩阵

| 风险 | 概率 | 影响 | 应对策略 |
|------|------|------|---------|
| 3B 模型微调后质量不够 | 中 | 高 | Phase 0 同时测 3B 和 7B，确认能力下限 |
| 行为信号太嘈杂，样本质量差 | 高 | 高 | Curator 层保守筛选，宁可少不可脏；Phase 1 设计信号提纯 |
| 增量微调导致灾难性遗忘 | 中 | 高 | Replay Buffer + 每版评测，退化自动回滚 |
| 云功能引发的数据出境/隐私顾虑 | 中 | 高 | 默认 `strict_local`，Teacher/Judge/Router 云能力显式 opt-in，调用前脱敏并写审计日志 |
| 社区不买账 | 中 | 中 | Phase 0 结束先发技术博客验证关注度，再决定投入力度 |
| unsloth 不支持某些模型/平台 | 低 | 中 | peft + trl 作为 fallback 方案 |
| Mac MPS 训练不稳定 | 中 | 中 | Mac 走 mlx-lm 原生路径（非 MPS），避免 MPS 后端已知问题 |
| 路由层判断不准确 | 中 | 中 | Phase 0-1 不做路由，Phase 2 从规则路由开始渐进 |
| 多后端 adapter 产物不兼容 | 中 | 高 | 定义 canonical artifact + manifest，serve 前做兼容性检查与必要转换 |
| Teacher / Judge 污染评测 | 中 | 高 | 强制 holdout/test split，记录 provenance，默认避免 teacher 数据直接作为 judge 测试集 |

### 关键风险详解

#### 风险 1：小模型能力下限

**问题：** 非暴力沟通场景需要情绪识别、共情表达、语境理解——这些恰恰是小模型最弱的地方。LoRA 微调能注入风格，但很难弥补基座模型在推理和共情上的能力缺口。

**应对：**
1. Phase 0 W4 做 3B vs 7B 对比实验
2. 设计分级评测：基础能力（情绪识别准确率）+ 风格能力（风格匹配度）
3. 如果 3B 基础能力不达标，调整为 7B 作为默认推荐模型
4. 长期方向：随着小模型能力提升（Qwen3、Llama4 等），持续下探参数量

#### 风险 2：信号噪声

**问题：** 用户复制一段话可能是觉得好，也可能是发给别人吐槽。打断可能是不满意，也可能是网络卡了。

**应对：**
1. 信号评分采用保守策略，模糊信号不进入训练集
2. 多信号交叉验证：单一信号不直接作为训练样本
3. 提供 `pfe data review` 命令，让用户审核自动标注
4. 持续迭代评分策略，基于用户反馈调整权重

#### 风险 3：数据出境与合规边界

**问题：** Teacher 模式、云端 Judge、Router fallback 都会让“默认本地”的叙事变得模糊；如果不开启前没有明确说明，容易形成产品承诺与真实行为不一致。

**应对：**
1. 默认模式固定为 `strict_local`
2. 所有云功能都通过独立开关显式启用，不允许隐式回退到云端
3. 云调用前先脱敏，并写入 `egress_audit_log`
4. 文档和 CLI 帮助明确区分“推理兼容”与“个性化闭环接入”

#### 风险 4：冷启动

**问题：** 在第一次微调完成之前，用户体验和普通 chatbot 没有区别。

**应对：**
1. 提供预训练场景 adapter 作为默认起点（Day 1 即有个性化体验）
2. Teacher 模式快速生成 200 条数据 → 首次微调可在 10 分钟内完成
3. 社区贡献场景 adapter 模板，降低冷启动门槛

#### 风险 5：蒸馏评测失真

**问题：** 如果 Teacher 生成的数据既参与训练又直接作为 Judge 测试集，或者 Judge 与 Teacher 使用同一套模板，评测结果会偏乐观，无法反映真实泛化能力。

**应对：**
1. 蒸馏样本强制打 `train / val / test` 切分标记
2. `test` 切分不得参与训练，也不得进入 Replay Buffer
3. 评测默认从 holdout / test 集选样
4. 记录 `teacher_model / judge_model / prompt_version / run_id`
5. 长期追踪中单独统计 teacher 数据占比，监控是否“越训越像 teacher”

---

## 三、开源运营策略

### 命名

`personal-finetune-engine` 太长，推荐短名字：
- **pfe** — 简洁直接，CLI 命令友好
- 备选：`meld`、`imprint`

### License

MIT — 简洁宽松，对商业使用和社区采用都更友好。

### 社区基础设施

| 设施 | 工具 | 时间点 |
|------|------|--------|
| 代码托管 | GitHub | Phase 0 开始 |
| CI/CD | GitHub Actions | Phase 0 W1 |
| 文档站 | MkDocs + Material | Phase 2 W8 |
| 讨论区 | GitHub Discussions | Phase 1 发布时 |
| Issue 模板 | Bug Report / Feature Request / Model Support | Phase 1 |

### 社区贡献方向

1. **信号采集插件：** 更多平台的交互信号采集（VSCode、浏览器、Obsidian 等）
2. **场景 adapter 模板：** 预训练好的场景起点（写作、编程、客服等）
3. **评测方法：** 更多维度的评测指标和 Judge prompt
4. **基座模型适配：** 新模型的训练配置和兼容性验证

---

## 四、商业模式思考

> **回应 review 第 6 点：** 即使是开源项目，也需要明确可持续发展的路径。

| 模式 | 描述 | 可行性 |
|------|------|--------|
| 云端 Teacher API | 提供高质量的数据生成服务（按量付费） | 中，需要有差异化 |
| 托管微调服务 | 为没有 GPU 的用户提供云端微调（按次付费） | 高，真实需求 |
| 企业版 | 多用户管理、审计日志、合规功能 | 中长期 |
| 场景 Adapter 市场 | 社区贡献的高质量场景 adapter（付费/免费） | 长期 |

**短期策略：** 先不考虑商业化，专注引擎质量和社区增长。当 GitHub Stars > 1K 时再认真规划。

---

## 五、质量保障

### 测试策略

| 层级 | 覆盖范围 | 工具 |
|------|---------|------|
| 单元测试 | 各模块核心逻辑 | pytest |
| 集成测试 | 端到端 pipeline | pytest + fixtures |
| 效果测试 | 微调前后质量对比 | AutoEvaluator |
| 兼容性测试 | 多模型 + 多平台 | GitHub Actions matrix |

### CI/CD Pipeline

```yaml
# 每次 PR
- lint (ruff)
- type check (mypy)
- unit tests
- integration tests (mock model)

# 每周 / Release 前
- 真实模型微调测试（3B, 7B）
- 多平台测试（Linux CUDA, Mac mlx-lm, CPU-only）
- 效果回归测试
```

---

## 六、已知限制与未来考虑

以下场景当前版本（v1.0 前）不支持，记录在此供后续规划：

| 场景 | 当前状态 | 计划 |
|------|---------|------|
| 多用户/多租户 | 不支持，单用户设计 | v2.0+ 企业版考虑 |
| 数据备份/迁移 | 手动复制 `~/.pfe/` 目录 | Phase 2 提供 `pfe backup` / `pfe restore` 命令 |
| GPU 资源竞争 | 不检测其他 GPU 进程 | Phase 1 训练前检查 GPU 显存可用量，不足时提示用户 |
| 本地数据加密 | 不加密，依赖 OS 文件权限 | Phase 2 可选 `pfe init --encrypt`，使用 SQLCipher 加密 SQLite |
| 离线模式 | Teacher 模式需要网络 | 提供预下载的场景 adapter，离线可用 |
| 完整闭环接入 | 仅有 OpenAI API 兼容还不够 | 通过 Signal SDK / `/pfe/signal` 补齐事件上报 |

---

## 七、附录：Review 问题回应追踪

| Review 问题 | 文档回应位置 | 状态 |
|-------------|-------------|------|
| 1. 验证标准缺失 | 03-module-design.md → Evaluator → 验证标准 | ✅ 已补充 |
| 2. 1.5B-3B 能力下限 | 04-roadmap.md → Phase 0 → 同时测 3B 和 7B | ✅ 已补充 |
| 3. 无感埋点信号提纯 | 04-roadmap.md → Phase 1 → 信号提纯设计 | ✅ 已补充 |
| 4. 冷启动解法 | 03-module-design.md → Curator → Day 1 方案；04-roadmap.md → Phase 1 冷启动 | ✅ 已补充 |
| 5. 路由层判断逻辑 | 03-module-design.md → Router → 分阶段实现 | ✅ 已补充 |
| 6. 商业模式 | 05-tech-and-risk.md → 商业模式思考 | ✅ 已补充 |
