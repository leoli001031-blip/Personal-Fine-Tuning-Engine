# PFE 架构设计

## 一、项目目录结构

```
personal-finetune-engine/
├── pfe-core/                  # 核心引擎
│   ├── collector/             # 信号采集层
│   ├── curator/               # 样本构建层
│   ├── trainer/               # 微调训练层
│   ├── evaluator/             # 效果评测层
│   ├── router/                # 推理路由层
│   ├── adapter_store/         # Adapter 管理层
│   └── inference/             # 推理引擎层
├── plugins/                   # 插件体系
│   ├── collectors/            # 信号采集插件
│   ├── curators/              # 样本策略插件
│   └── evaluators/            # 评测方法插件
├── examples/                  # 官方示例应用
│   ├── life-coach/            # 非暴力沟通 Life Coach
│   ├── writing-assistant/     # 写作助手
│   └── task-dispatcher/       # 任务分发
├── pfe-cli/                   # CLI 工具
├── pfe-server/                # 本地 API 服务（OpenAI 兼容，依赖 pfe-core/inference）
├── tests/                     # 测试
├── pyproject.toml
└── README.md
```

> **命名约定：** 目录名使用 kebab-case（`pfe-core/`），Python 包名使用 snake_case（`pfe_core`）。安装后通过 `import pfe_core` 引用。映射关系在 `pyproject.toml` 中通过 `[tool.setuptools.package-dir]` 配置：`pfe_core = "pfe-core"`，`pfe_cli = "pfe-cli"`，`pfe_server = "pfe-server"`。

## 二、数据流架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        用户交互层                                │
│  (Chat UI / API Client / 第三方应用)                             │
└──────────────┬──────────────────────────────────┬───────────────┘
               │ 交互事件(InteractionEvent)        │ 推理请求
               ▼                                   ▼
┌──────────────────────┐                ┌─────────────────────┐
│   Signal Collector   │                │      Router         │
│   ─────────────────  │                │   ───────────────   │
│   • ChatCollector    │                │   • 规则匹配        │
│   • EditCollector    │                │   • 复杂度估算      │
│   • CopyCollector    │                │   • 置信度检测      │
│   • InterruptCollector│               └────┬──────────┬─────┘
└──────────┬───────────┘                     │          │
           │ RawSignal                  本地推理     云端路由
           ▼                                │          │
┌──────────────────────┐              ┌─────▼──────────▼─────┐
│    Data Curator      │              │  Inference Engine     │
│   ─────────────────  │              │  ──────────────────   │
│   • SignalScorer     │              │  • transformers 后端  │
│   • 样本筛选/格式化   │              │  • mlx-lm 后端       │
│   • 去重/平衡         │              │  • llama.cpp 后端    │
│   • Teacher 模式      │              │  • Adapter 热切换    │
└──────────┬───────────┘              └──────────────────────┘
           │ TrainingDataset                     ▲
           ▼                                     │ 加载 adapter
┌──────────────────────┐                         │
│      Trainer         │                         │
│   ─────────────────  │                         │
│   • SFT 微调         │                         │
│   • DPO 偏好对齐     │                         │
│   • 增量训练         │                         │
│   • Replay Buffer    │                         │
└──────────┬───────────┘                         │
           │ AdapterResult                       │
           ▼                                     │
┌──────────────────────┐     ┌──────────────────────┐
│     Evaluator        │────▶│   Adapter Store      │──┘
│   ─────────────────  │     │   ───────────────    │
│   • AutoEvaluator    │     │   • 版本管理         │
│   • ABTestEvaluator  │     │   • 回滚             │
│   • LongTermTracker  │     │   • 清理             │
└──────────────────────┘     └──────────────────────┘
```

### 2.1 接入边界：推理兼容 vs 个性化闭环

PFE 对外提供两层能力，文档中必须明确区分：

1. **推理兼容层：** `pfe-server` 兼容 OpenAI Chat Completions API，现有前端可以把它当作本地模型服务使用
2. **个性化闭环层：** 若希望 PFE 自动采集接受/拒绝/编辑/重生成等行为，并持续完成 `signal -> sample -> train -> eval -> deploy` 闭环，上层应用还需要接入 PFE Signal SDK 或主动调用 `/pfe/signal`

> **结论：** “OpenAI API 兼容”只覆盖推理接入，不等于“零改动接入完整个性化闭环”。

### 2.2 LLM 蒸馏与评测闭环

PFE 允许引入更强的 LLM 参与数据蒸馏和训练结果评测，推荐的闭环如下：

```
用户历史 / 场景模板 / 原始信号
          ↓
   Teacher / Distiller
          ↓
   Filter / Dedup / Split
          ↓
     SFT / DPO Dataset
          ↓
        Trainer
          ↓
 Base vs Adapted Outputs
          ↓
       Judge LLM
          ↓
  EvalReport + Deploy Decision
```

约束：

- Teacher 和 Judge 逻辑上是两个角色，默认不共享评测集
- 蒸馏集、验证集、测试集要显式切分；测试集不得回流训练
- 所有蒸馏/评测调用都要记录模型、prompt 版本、采样参数和时间

## 三、核心数据模型

所有数据模型使用 Python `@dataclass` 定义，同时提供 Pydantic v2 版本用于 API 校验：

```python
# 内部使用（dataclass）
from pfe_core.models import TrainingSample

# API 层使用（Pydantic，自动校验）
from pfe_server.models import TrainingSampleSchema
```

两个版本的字段完全一致，通过工具函数相互转换：
```python
from pfe_core.converters import to_pydantic, to_dataclass

sample = TrainingSample(...)  # dataclass
schema = to_pydantic(sample)   # 转为 Pydantic model
sample2 = to_dataclass(schema) # 转回 dataclass
```

### 3.0 交互事件（InteractionEvent）

```python
@dataclass
class InteractionEvent:
    event_id: str            # 唯一事件 ID
    request_id: str          # 同一次生成请求的唯一 ID
    parent_event_id: str | None   # 同一会话内的前序事件（如 regenerate 前的被拒绝回复）
    event_type: str          # chat | edit | copy | interrupt
    timestamp: datetime
    session_id: str          # 所属会话 ID
    adapter_version: str | None   # 触发这次回复时所使用的 adapter 版本
    scenario: str | None     # 场景标签（life-coach / writing 等）
    user_input: str          # 用户输入内容
    model_output: str        # 模型输出内容
    user_action: str         # accept | reject | edit | copy | interrupt | regenerate
    action_detail: dict      # 具体操作详情（编辑 diff、停止位置等）
    metadata: dict           # 场景标签、设备信息等
```

> `InteractionEvent` 是 Signal Collector 的输入，由上层应用（Chat UI / API Client）产生。`RawSignal` 是 Signal Collector 的输出，是对 `InteractionEvent` 的标准化提取。

### 3.1 原始信号（RawSignal）

```python
@dataclass
class RawSignal:
    signal_id: str
    source_event_id: str     # 对应的 InteractionEvent.event_id
    request_id: str
    session_id: str
    adapter_version: str | None    # 该信号发生时，回复来自哪个 adapter 版本
    event_type: str          # accept | reject | edit | copy | interrupt | regenerate
    timestamp: datetime
    context: str             # 当时的对话上下文
    model_output: str        # 模型原始输出
    user_action: dict        # 用户的具体操作（编辑内容、停止位置等）
    metadata: dict           # 场景、设备等元信息
```

### 3.2 训练样本（TrainingSample）

```python
@dataclass
class TrainingSample:
    sample_id: str
    sample_type: str         # "sft" | "dpo"
    instruction: str         # 用户输入/上下文
    chosen: str              # 期望输出（SFT 时即为 output，DPO 时为偏好输出）
    rejected: str | None     # 拒绝输出（仅 DPO 使用，SFT 时为 None）
    score: float             # 样本质量分
    source: str              # signal | teacher | import | manual
    source_event_ids: list[str]   # 参与构造该样本的事件链路
    source_adapter_version: str | None   # 产生原始回复的 adapter 版本
    metadata: dict           # 蒸馏来源、teacher/judge 版本、场景标签、切分信息等
```

> **格式约定：** `sample_type="sft"` 时 `rejected` 为 None，训练时使用 `instruction` + `chosen` 作为 SFT 对；`sample_type="dpo"` 时 `rejected` 不为空，训练时使用 `instruction` + `chosen` + `rejected` 作为 DPO 三元组。
>
> **链路要求：** 所有由隐式行为自动生成的 DPO 样本，必须能追溯到 `session_id + request_id + source_event_ids`。如果事件链不完整，则降级为 SFT 样本或直接丢弃，避免错误配对。
>
> **蒸馏要求：** `source="teacher"` 时，`metadata` 至少包含 `teacher_model`、`teacher_prompt_version`、`generation_config`、`dataset_split`。

### 3.3 Adapter 元信息（AdapterMeta）

```python
@dataclass
class AdapterMeta:
    version: str            # e.g. "20260401-001"
    base_model: str
    created_at: datetime
    num_samples: int        # 训练样本数
    state: str              # training | pending_eval | promoted | archived | failed_eval
    artifact_format: str    # peft_lora | mlx_lora | gguf_merged
    training_config: dict   # 训练参数
    eval_report: EvalReport | None
```

### 3.4 评测报告（EvalReport）

```python
@dataclass
class EvalReport:
    adapter_version: str
    base_model: str
    num_test_samples: int
    scores: dict[str, float]     # style_match, preference_alignment, quality_preservation, personality_consistency
    comparison: str              # "improved" | "neutral" | "degraded"
    recommendation: str          # "deploy" | "keep_previous" | "needs_more_data"
    details: list[EvalDetail]    # 逐条对比详情
```

## 四、存储设计

### 4.1 信号与样本存储

使用本地 SQLite，零依赖，按时间分区：

```
~/.pfe/data/
├── signals.db               # 原始信号
├── samples.db               # 训练样本
└── exports/                 # 导出目录
```

**signals 表结构：**

```sql
CREATE TABLE signals (
    id               TEXT PRIMARY KEY,
    source_event_id  TEXT NOT NULL,
    request_id       TEXT NOT NULL,
    session_id       TEXT NOT NULL,
    adapter_version  TEXT,
    event_type       TEXT NOT NULL,
    timestamp        DATETIME NOT NULL,
    context          TEXT,
    model_output     TEXT,
    user_action      TEXT,          -- JSON
    metadata         TEXT,          -- JSON
    processed        BOOLEAN DEFAULT FALSE
);

CREATE INDEX idx_signals_time ON signals(timestamp);
CREATE INDEX idx_signals_type ON signals(event_type);
CREATE INDEX idx_signals_session ON signals(session_id);
CREATE INDEX idx_signals_request ON signals(request_id);
CREATE INDEX idx_signals_adapter ON signals(adapter_version);
```

**samples 表结构：**

```sql
CREATE TABLE samples (
    id                    TEXT PRIMARY KEY,
    sample_type           TEXT NOT NULL,       -- sft | dpo
    instruction           TEXT NOT NULL,
    chosen                TEXT NOT NULL,
    rejected              TEXT,                -- NULL for SFT samples
    score                 REAL NOT NULL,
    source                TEXT NOT NULL,       -- signal | teacher | import | manual
    source_event_ids      TEXT,                -- JSON array
    source_adapter_version TEXT,
    created_at            DATETIME NOT NULL,
    used_in_version       TEXT,                -- 首次用于训练的 adapter 版本（NULL=未使用）
    metadata              TEXT                 -- JSON
);

CREATE INDEX idx_samples_type ON samples(sample_type);
CREATE INDEX idx_samples_score ON samples(score);
CREATE INDEX idx_samples_used ON samples(used_in_version);
CREATE INDEX idx_samples_source_adapter ON samples(source_adapter_version);
```

> **关于 `used_in_version`：** 该字段记录样本**首次**被用于训练的 adapter 版本。增量训练时，即使样本被多次用作 Replay Buffer，该字段保持不变。Replay Buffer 的选取基于 `used_in_version` 和训练时间窗口（最近 5 个版本），而非该字段的更新。

### 4.2 Adapter 存储

```
~/.pfe/adapters/
├── user_default/
│   ├── 20260401-001/                    # 格式：YYYYMMDD-NNN（日期+当日序号）
│   │   ├── adapter_model.safetensors
│   │   ├── adapter_config.json
│   │   ├── adapter_manifest.json
│   │   ├── training_meta.json
│   │   └── eval_report.json
│   ├── 20260408-001/
│   └── latest -> 20260408-001/          # 软链接指向最新版本
└── scenarios/
    ├── life_coach/
    └── writing/
```

> **版本号格式：** 使用 `YYYYMMDD-NNN` 格式（如 `20260401-001`），比 `v001` 更具可读性，能直接看出训练时间。CLI 中支持简写：`pfe adapter rollback 20260401-001` 或 `pfe adapter rollback -1`（回退到上一版本）。

**Canonical Artifact 契约：**

- 默认标准产物是 `peft_lora`：`adapter_model.safetensors + adapter_config.json + adapter_manifest.json`
- `adapter_manifest.json` 记录 `artifact_format`、训练后端、目标推理后端、是否需要额外导出步骤
- `unsloth` / `peft` 训练后必须导出为上述标准目录
- `mlx-lm` 训练后若原生产物格式不同，必须额外导出到标准目录，或在 manifest 中声明 `artifact_format = "mlx_lora"`
- `llama.cpp` 不直接消费 LoRA adapter；当 serve 端选择 `llama.cpp` 时，由 `pfe export --target gguf-merged` 在缓存目录生成可加载的合并 GGUF，并在 manifest 中登记来源版本

**训练/推理兼容矩阵：**

| 训练后端 | 标准保存格式 | transformers | mlx-lm | llama.cpp |
|---------|-------------|-------------|--------|-----------|
| unsloth / peft | `peft_lora` | 直接加载 | 需要转换 | 需要合并导出 GGUF |
| mlx-lm | `mlx_lora` 或导出成 `peft_lora` | 需要转换 | 直接加载 | 需要合并导出 GGUF |
| 任意后端导出的 GGUF | `gguf_merged` | 不适用 | 不适用 | 直接加载 |

### 4.3 配置存储

```
~/.pfe/
├── config.toml              # 全局配置
├── data/
├── adapters/
└── cache/                   # 临时缓存
```

**config.toml 完整配置项：**

```toml
[model]
base_model = "Qwen/Qwen2.5-3B-Instruct"   # 基座模型 HF ID 或本地路径
device = "auto"                              # auto | cpu | cuda | mps

[trainer]
method = "qlora"              # lora | qlora
train_type = "sft"            # sft | dpo
quantization = "4bit"         # 4bit | 8bit | none
lora_r = 16
lora_alpha = 32
epochs = 3
learning_rate = 2e-4
max_samples = 500
dpo_beta = 0.1
replay_ratio = 0.3            # Replay Buffer 中历史样本占比

[trainer.trigger]
min_new_samples = 50           # 触发增量微调的最小新样本数
max_interval_days = 7          # 最长间隔天数（有新样本时）

[curator]
score_threshold = 0.3          # 低于此分数的样本丢弃
dedup_similarity = 0.92        # embedding 去重相似度阈值
dedup_model = "BAAI/bge-small-zh-v1.5"  # 去重用的 embedding 模型

[distillation]
enabled = false
teacher_model = ""
teacher_prompt_version = "v1"
teacher_temperature = 0.7
teacher_max_samples = 200
rewrite_weak_samples = true
generate_dpo_pairs = true
train_split = 0.8
val_split = 0.1
test_split = 0.1

[evaluation.judge]
mode = "local_first"           # local_first | cloud_only
judge_model = ""
judge_prompt_version = "v1"
require_holdout_split = true
forbid_teacher_test_overlap = true

[server]
port = 8921                    # 默认端口（避开常见的 8000/8080）
host = "127.0.0.1"             # 如需公网访问，改为 "0.0.0.0" 且必须启用 API Key 鉴权

[router]
strategy = "local_only"        # local_only | keyword | confidence | hybrid
confidence_threshold = 0.7
cloud_model = ""               # 云端模型 API（如 gpt-4）
cloud_api_key_env = "OPENAI_API_KEY"  # 从环境变量读取，不存配置文件

[privacy]
mode = "strict_local"          # strict_local | cloud_assisted
allow_teacher_cloud = false    # 是否允许 Teacher 模式调用云端 API
allow_judge_cloud = false      # 是否允许 LLM-as-a-Judge 使用云端
allow_router_cloud = false     # 是否允许路由到云端模型
redact_pii = true              # 云调用前先做脱敏
require_explicit_consent = true
egress_audit_log = "~/.pfe/logs/egress_audit.jsonl"

[security]
allow_remote_access = false    # false 时仅允许 127.0.0.1
auth_mode = "local_optional"   # local_optional | api_key_required
api_key_env = "PFE_API_KEY"
allowed_origins = ["http://127.0.0.1", "http://localhost"]

[storage]
signal_retention_days = 90     # 原始信号保留天数
adapter_keep_versions = 10     # adapter 保留版本数

[logging]
level = "INFO"                 # DEBUG | INFO | WARNING | ERROR
file = "~/.pfe/logs/pfe.log"  # 日志文件路径
max_size_mb = 50               # 单文件最大大小
backup_count = 3               # 保留的历史日志文件数
format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
```

### 4.4 隐私、出境与安全策略

PFE 的默认模式是 `strict_local`，因此涉及云端的能力必须经过统一的出境控制：

| 能力 | 默认状态 | 开启条件 | 出境内容 | 额外要求 |
|------|---------|---------|---------|---------|
| Teacher 模式 | 关闭 | 用户显式开启 `allow_teacher_cloud=true` | 场景模板、风格描述、脱敏后的用户文本片段 | 写入出境审计日志 |
| LLM-as-a-Judge | 关闭 | 用户显式开启 `allow_judge_cloud=true` | 测试 prompt、对比回复、可选参考回复 | 评测报告记录所用 Judge 模型 |
| Router 云端兜底 | 关闭 | 用户显式开启 `allow_router_cloud=true` 且配置云模型 | 当前请求上下文 | 返回结果标记 `served_by=cloud` |

安全约束：

- `host = 127.0.0.1` 是默认值；若改为 `0.0.0.0`，必须同时开启 `auth_mode = "api_key_required"`
- `/pfe/train/trigger`、`/pfe/signal`、`/pfe/adapters` 属于管理接口；非本机访问时必须鉴权
- 所有云调用都要先经过脱敏器；脱敏失败时请求直接中止

### 4.5 数据保留策略

| 数据类型 | 默认保留时间 | 清理方式 |
|---------|------------|---------|
| 原始信号（signals.db） | 90 天 | 每次启动时自动清理过期记录 |
| 训练样本（samples.db） | 永久保留（已用于训练的样本是 Replay Buffer 的数据源） | 手动 `pfe data prune --older-than 1y`（清理未使用的旧样本） |
| 蒸馏元数据 | 跟随样本永久保留 | 写入 `samples.metadata` |
| Adapter 版本 | 保留最近 10 个版本 | `AdapterStore.prune(keep_last_n=10)` |
| 评测报告 | 跟随 adapter 版本，adapter 被清理时一并删除 | 同上 |

> **Adapter 保留数量说明：** 默认从 5 调整为 10。按默认触发条件（50 条新样本触发一次），10 个版本可覆盖约 500 条样本的训练历史，足够回滚到较早的稳定版本。

### 4.6 并发与锁机制

```
pfe serve（推理服务）与 pfe train（训练）可能同时运行，需要处理以下并发场景：
```

| 场景 | 冲突点 | 解决方案 |
|------|--------|---------|
| serve 运行中执行 train | adapter 文件被替换 | 训练写入新版本目录（如 `20260408-002/`），完成后原子更新 latest 软链接，serve 通过 `watchdog` 库监听 `~/.pfe/adapters/*/latest` 软链接变化（`FileSystemEventHandler`，轮询间隔 1s），检测到变化后调用 `swap_adapter()` |
| 信号采集与样本构建同时写 SQLite | SQLite 写锁 | 连接时设置 WAL 模式：`PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA wal_autocheckpoint=1000;` |
| 多个 CLI 命令同时执行 train | 重复训练 | 使用 `portalocker` 库实现文件锁（`~/.pfe/.train.lock`），跨平台兼容（Linux flock / macOS / Windows），获取锁超时 5s 后报错提示已有训练进程 |

### 4.7 训练容错与恢复

| 故障场景 | 恢复机制 |
|---------|---------|
| 训练中途 OOM | 捕获异常，清理不完整的 adapter 目录，保持 latest 指向上一版本不变 |
| 训练中途断电/进程被杀 | 启动时检查 `~/.pfe/adapters/` 下是否有不完整的版本目录（无 `adapter_model.safetensors` 文件），自动清理 |
| Checkpoint 策略 | 每个 epoch 结束保存 checkpoint 到 `~/.pfe/cache/train_checkpoint/`，训练中断后可通过 `pfe train --resume` 恢复 |
| 评测失败 | adapter 已保存但未更新 latest，等待下次评测或手动 `pfe adapter promote <version>` |

### 4.8 Adapter 生命周期

为避免“保存”和“上线”语义混淆，adapter 的状态机统一如下：

| 状态 | 含义 | 允许操作 |
|------|------|---------|
| `training` | 训练尚未完成，产物不可信 | checkpoint / abort |
| `pending_eval` | 训练完成，产物已落盘，但还未决定是否上线 | eval / promote / archive |
| `promoted` | 已通过评测并成为 `latest` | serve / rollback / archive |
| `failed_eval` | 已完成评测但不建议上线 | manual promote / archive |
| `archived` | 历史版本，不参与默认 serve | inspect / delete |

状态流转规则：

1. `Trainer` 只负责产生产物并写入版本目录，完成后状态设为 `pending_eval`
2. `Evaluator` 只负责生成报告和 recommendation，不直接覆盖 `latest`
3. `AdapterStore.promote(version)` 是唯一允许更新 `latest` 软链接的入口
4. 自动回滚的本质不是“删除失败版本”，而是把失败版本标记为 `failed_eval`，同时保持上一版 `promoted` 版本继续服务

## 五、插件体系设计

所有核心模块通过抽象基类定义接口，Collector、Curator、Evaluator 支持插件替换：

```python
# 注册自定义 Collector
from pfe_core import register_collector

@register_collector("my_collector")
class MyCollector(SignalCollector):
    def collect(self, event: InteractionEvent) -> RawSignal:
        ...

# 注册自定义 Curator 策略
from pfe_core import register_curator

@register_curator("my_strategy")
class MyStrategy(CurationStrategy):
    def curate(self, signals: list[RawSignal]) -> list[TrainingSample]:
        ...

# 注册自定义 Evaluator
from pfe_core import register_evaluator

@register_evaluator("my_evaluator")
class MyEvaluator(BaseEvaluator):
    def evaluate(self, ...) -> EvalReport:
        ...
```

> **Trainer 和 Router 暂不开放插件注册。** Trainer 涉及底层训练框架（unsloth/peft/mlx）的深度集成，接口尚不稳定，计划 Phase 2 稳定后再开放。Router 的路由策略通过 `RouterConfig` 配置即可满足大部分需求，无需插件化。如有特殊需求，可直接继承 `Trainer` / `Router` 基类覆盖方法。

插件发现机制：基于 Python entry_points，第三方包可以通过 `pyproject.toml` 注册插件：

```toml
[project.entry-points."pfe.collectors"]
my_collector = "my_package:MyCollector"
```

## 六、Signal SDK 与事件接入协议

为了把“推理兼容”升级为“个性化闭环”，PFE 定义一层独立的事件接入协议。推荐形态：

- `pfe-signal-sdk`：给 Web / Desktop / CLI 应用使用的轻量 SDK
- `POST /pfe/signal`：无 SDK 时的裸 HTTP 入口

最小事件载荷示例：

```json
{
  "event_id": "evt_123",
  "request_id": "req_456",
  "session_id": "sess_001",
  "parent_event_id": "evt_122",
  "adapter_version": "latest",
  "scenario": "life-coach",
  "event_type": "chat",
  "user_action": "edit",
  "action_detail": {
    "edited_text": "..."
  }
}
```

> **接入要求：** 只有当上层应用能稳定上报 `request_id / session_id / adapter_version / user_action` 时，PFE 才能可靠地构建 DPO 样本和长期版本分析。

## 七、API 设计（pfe-server）

本地推理服务兼容 OpenAI Chat Completions API：

```
POST /v1/chat/completions
{
    "model": "local",              # 使用本地模型 + 最新 adapter
    "messages": [...],
    "adapter_version": "latest"    # PFE 扩展字段：指定 adapter 版本
}
```

> **兼容边界：** 上述兼容仅覆盖推理请求。标准 OpenAI 客户端不会自动上报接受/拒绝/编辑等反馈，因此不能单靠该接口实现个性化训练闭环。

PFE 扩展端点：

```
GET  /pfe/adapters              # 列出所有 adapter 版本
POST /pfe/feedback              # 上报 accept/edit/reject/regenerate/delete
POST /pfe/signal                # 上报行为信号
POST /pfe/distill/run           # 触发一次 Teacher / Distillation 任务
GET  /pfe/status                # 引擎状态（训练进度、样本统计等）
POST /pfe/train/trigger         # 触发一次微调
GET  /pfe/eval/latest           # 最新评测报告
```

鉴权规则：

- 本机环回地址访问时，可使用 `auth_mode = "local_optional"`
- 任意非本机访问，或 `host = 0.0.0.0` 时，必须切换为 `auth_mode = "api_key_required"`
- 管理接口默认要求 `Authorization: Bearer $PFE_API_KEY`

## 八、日志规范

使用 Python 标准 `logging` 模块，每个模块使用独立的 logger：

```python
import logging
logger = logging.getLogger("pfe.trainer")  # pfe.collector, pfe.curator, pfe.trainer, pfe.evaluator, pfe.router, pfe.inference, pfe.server
```

| 级别 | 使用场景 |
|------|---------|
| DEBUG | 训练 loss 每步输出、信号采集详情、SQL 查询 |
| INFO | 训练开始/结束、adapter 保存、评测结果、服务启动 |
| WARNING | 样本质量低被丢弃、adapter 回滚、磁盘空间不足 |
| ERROR | 训练失败、模型加载失败、API 调用失败 |

日志同时输出到终端（INFO 及以上）和文件（DEBUG 及以上）。训练过程中的 loss 曲线额外写入 `~/.pfe/logs/train_{version}.jsonl`。

## 九、错误处理策略

所有模块使用统一的异常层次结构：

```python
class PFEError(Exception):
    """PFE 基础异常"""
    code: str  # 错误码

class ModelNotFoundError(PFEError):       # code: "E001"
    """基座模型未找到或无法下载"""

class TrainingError(PFEError):            # code: "E100"
    """训练过程中的错误（OOM、数据格式错误等）"""

class AdapterError(PFEError):             # code: "E200"
    """Adapter 加载/保存/切换失败"""

class EvalError(PFEError):                # code: "E300"
    """评测过程中的错误（Judge API 调用失败等）"""

class InferenceError(PFEError):           # code: "E400"
    """推理过程中的错误（模型加载失败、生成超时等）"""

class DataError(PFEError):                # code: "E500"
    """数据相关错误（SQLite 损坏、样本格式错误等）"""

class ConfigError(PFEError):              # code: "E600"
    """配置错误（缺少必要配置、值不合法等）"""

class AuthError(PFEError):                # code: "E700"
    """认证/鉴权失败（缺失 API Key、权限不足等）"""
```

CLI 层捕获所有 `PFEError`，输出用户友好的错误信息和建议操作。非 PFEError 异常视为 bug，输出完整 traceback 并提示用户提交 issue。
