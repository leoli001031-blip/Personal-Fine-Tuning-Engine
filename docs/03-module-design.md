# PFE 核心模块详细设计

## 模块 1：Signal Collector（信号采集层）

### 职责
从用户交互中采集原始行为信号。不做判断，只做记录。

### 核心接口

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime


class SignalCollector(ABC):
    @abstractmethod
    def collect(self, event: InteractionEvent) -> RawSignal:
        """将一次用户交互转化为原始信号"""
        pass
```

### 内置采集器

| 采集器 | 信号来源 | 采集内容 | 开发阶段 |
|--------|---------|---------|---------|
| `ChatCollector` | 对话交互 | 接受/拒绝/编辑/重新生成 | Phase 1 |
| `EditCollector` | 文本编辑 | 用户对模型输出的修改 diff | Phase 1 |
| `CopyCollector` | 复制行为 | 哪些输出被复制使用 | Phase 1 |
| `InterruptCollector` | 打断行为 | 在哪个 token 位置被打断 | Phase 1 |

### 存储
- 本地 SQLite，按时间分区
- 支持导出（JSONL）和定期清理
- 默认保留 90 天原始信号

> **信号 vs 样本的保留关系：** 原始信号（signals.db）90 天后自动清理（每次 CLI 命令启动时检查），但在清理前已经被 Data Curator 转化为训练样本（samples.db）的数据会永久保留。即：信号是临时的原始数据，样本是持久的训练资产。如果用户需要保留更长时间的原始信号用于回溯分析，可通过 `config.toml` 的 `signal_retention_days` 调整，或在清理前执行 `pfe data export --format jsonl` 导出归档。

### 接入要求

- 上层应用至少要上报 `event_id`、`request_id`、`session_id`、`adapter_version`、`user_action`
- 如果只接了 OpenAI 兼容推理接口，没有接事件上报，PFE 只能提供推理服务，不能自动构建训练闭环

### 开发优先级
Phase 0 先验证 `generate -> train -> eval`；`ChatCollector` 和其他隐式信号采集器统一放在 Phase 1 落地。

---

## 模块 2：Data Curator（样本构建层）

### 职责
把原始信号转化为可用于微调的高质量训练样本。**这是整个引擎最核心的模块。**

### 核心流程

```
RawSignal → 信号评分 → 样本筛选 → 样本格式化 → 去重/平衡 → TrainingDataset
```

### 数据分层策略

Data Curator 在进入训练前，必须先回答“这条信息到底属于 memory、profile、prompt context，还是训练样本”。

- 身份事实（名字、职业）优先进入 `memory`
- 长期偏好优先进入 `profile`
- 一次性任务背景只进入 `prompt_context`
- accept / edit / reject / regenerate 等行为反馈进入 `signal`
- 只有经过筛选和配对后的高质量反馈才进入 `SFT / DPO`

详细策略见 [reference/data-layering-strategy.md](reference/data-layering-strategy.md)。

### 信号评分策略

```python
class SignalScorer:
    def score(self, signal: RawSignal) -> float:
        """
        评分逻辑：
        - 用户接受且未编辑 → 正样本，高分 (0.9)
        - 用户接受但做了小幅编辑 → 正样本（用编辑后版本），中高分 (0.7)
        - 用户大幅编辑 → 配对样本（原始=负，编辑后=正），中分 (0.5)
        - 用户拒绝/重新生成 → 负样本，低分 (0.2)
        - 用户复制使用 → 正样本，高分 (0.8)
        - 用户中途打断 → 弱负样本，低分 (0.1)
        """
```

### DPO 样本来源

DPO 样本（含 chosen + rejected 对）的自动生成逻辑：

| 用户行为 | chosen | rejected | 说明 |
|---------|--------|----------|------|
| 大幅编辑后接受 | 编辑后的版本 | 模型原始输出 | 编辑 diff 超过 30% 视为"大幅" |
| 拒绝后重新生成并接受 | 重新生成后被接受的版本 | 被拒绝的原始输出 | 同一 session 内配对 |
| Teacher 模式生成 | 大模型生成的高质量回复 | 基座模型的零样本回复 | 冷启动阶段批量生成 |

> **注意：** 用户不需要手动提供 rejected 输出。DPO 样本完全从用户的隐式行为（编辑、拒绝）中自动提取。如果某个信号无法构成 chosen/rejected 对，则只生成 SFT 样本。

> **配对前提：** 自动构造 DPO 样本必须依赖完整的事件链路（`session_id + request_id + source_event_ids`）。如果无法确认“哪个 rejected 对应哪个 chosen”，则禁止猜测配对。

### 样本输出格式

支持两种训练范式：

**SFT 格式（监督微调）：**
```json
{
    "instruction": "我和同事吵架了，心情很差",
    "output": "听起来你现在很沮丧，和同事的冲突让你感到不舒服..."
}
```

**DPO 格式（偏好对齐）：**
```json
{
    "instruction": "我和同事吵架了，心情很差",
    "chosen": "听起来你现在很沮丧，和同事的冲突让你感到不舒服...",
    "rejected": "你应该冷静下来，吵架解决不了问题..."
}
```

### 关键设计决策

1. **去重：** 基于 embedding 相似度（阈值 0.92），使用 `BAAI/bge-small-zh-v1.5` 模型（中文优化、体积小、推理快），避免重复样本主导训练
2. **样本平衡：** 按场景/主题分桶，避免某类场景过度代表
3. **质量门槛：** score < 0.3 的样本直接丢弃
4. **数据增强：** 样本不足时，支持调用云端大模型做 Teacher 模式增强

### Teacher 模式（冷启动专用）

```python
class TeacherCurator:
    """用云端大模型批量生成场景化训练数据"""

    def generate(
        self,
        scenario_template: str,    # 场景模板
        style_description: str,    # 风格描述
        num_samples: int = 200
    ) -> list[TrainingSample]:
        """
        流程：
        1. 根据场景模板生成多样化的用户输入
        2. 用大模型按指定风格生成回复
        3. 对生成结果做质量筛选（基于规则 + 语义相似度去重）
        4. 格式化为 TrainingSample
        """
```

**隐私边界：**

- Teacher 模式默认关闭，只有在 `privacy.allow_teacher_cloud=true` 时才能调用云端 API
- 发往云端的输入先走脱敏流程；脱敏失败则终止本次生成
- 每次调用都写入 `egress_audit_log`，记录时间、模型、场景和样本数量

### Distillation Pipeline（正式蒸馏流程）

Teacher 模式在 PFE 中不只是“生成点数据”，而是一套明确的数据蒸馏管线：

```
Seed Inputs
  ├─ 场景模板
  ├─ 用户历史高质量样本
  └─ 低质量/待重写样本
        ↓
TeacherDistiller
  ├─ synthetic_generate      # 冷启动生成
  ├─ rewrite_weak_sample     # 重写弱样本
  ├─ expand_instruction      # 扩写用户输入分布
  └─ build_preference_pair   # 构造 chosen / rejected
        ↓
Filter / Dedup / Split
        ↓
SFT / DPO Dataset
```

```python
@dataclass
class DistillationConfig:
    teacher_model: str
    teacher_prompt_version: str = "v1"
    temperature: float = 0.7
    max_samples: int = 200
    rewrite_weak_samples: bool = True
    generate_dpo_pairs: bool = True
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


class TeacherDistiller:
    def distill_from_scenario(self, scenario: str, style: str, num_samples: int) -> list[TrainingSample]:
        """从场景模板蒸馏冷启动数据"""

    def distill_from_history(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        """基于用户历史高质量样本扩写或改写"""

    def rewrite_weak_sample(self, sample: TrainingSample) -> TrainingSample:
        """重写低质量/弱样本，输出更稳定的 teacher 版本"""

    def build_dpo_pair(self, prompt: str, chosen: str, rejected: str) -> TrainingSample:
        """显式构造 DPO 偏好对"""
```

蒸馏输出要求：

- 每条 `source="teacher"` 的样本都必须带 `metadata.teacher_model`
- 必须记录 `teacher_prompt_version`、`generation_config`、`dataset_split`
- `dataset_split` 只能是 `train` / `val` / `test`
- `test` 切分不得参与训练，也不得被 Replay Buffer 混入

推荐的蒸馏来源：

1. **Scenario Distillation：** 用场景模板生成 Day 1 冷启动数据
2. **History Distillation：** 用已有高质量样本扩写边界场景
3. **Repair Distillation：** 对人工发现的低质量回复做 teacher 重写
4. **Preference Distillation：** 让 teacher 为同一 prompt 构造 `chosen/rejected`

**冷启动解法（Day 1 价值方案）：**
- 提供预训练好的场景 adapter 作为默认起点（如 life-coach-base adapter）
- 用户首次使用即有个性化体验，后续交互数据持续优化
- 场景 adapter 模板由社区贡献，存放在 `plugins/curators/templates/`

**质量筛选标准：**
1. **长度过滤**：回复长度 < 20 字符或 > 2000 字符视为低质量
2. **重复检测**：基于 embedding 相似度去重（阈值 0.95），避免生成内容过于相似
3. **格式检查**：确保输出符合对话格式（无特殊标记、无截断）
4. **风格一致性**：随机采样 10% 生成结果，用 Judge LLM 评估是否符合指定风格

---

## 模块 3：Trainer（微调训练层）

### 职责
拿到训练样本后，在本地执行 LoRA/QLoRA 微调。

### 核心接口

```python
@dataclass
class TrainerConfig:
    base_model: str              # 基座模型路径或 HF ID
    method: str = "lora"         # lora | qlora
    train_type: str = "sft"      # sft | dpo
    quantization: str = "4bit"   # 4bit | 8bit | none
    lora_r: int = 16
    lora_alpha: int = 32
    epochs: int = 3
    learning_rate: float = 2e-4
    max_samples: int = 500       # 单次微调最大样本数
    device: str = "auto"         # auto | cpu | cuda | mps
    dpo_beta: float = 0.1        # DPO loss 的 beta 参数（仅 dpo 模式）


class Trainer:
    def train(
        self,
        config: TrainerConfig,
        dataset: TrainingDataset
    ) -> AdapterResult:
        """
        执行微调，返回 adapter 路径和训练指标。
        根据 config.train_type 自动选择 SFT 或 DPO 训练流程。
        """

    def train_incremental(
        self,
        base_adapter: str,
        new_dataset: TrainingDataset
    ) -> AdapterResult:
        """增量微调：在已有 adapter 基础上继续训练"""
```

### 技术选型
- 底层：`unsloth`（CUDA，速度快 2x，显存省 60%）/ `mlx-lm`（Apple Silicon）/ `peft` + `trl`（fallback）
- 默认配置：CUDA 走 4-bit QLoRA（8GB 显存即可跑 3B 模型），Mac 走 mlx-lm 原生精度
- 平台支持：Apple Silicon（mlx-lm 优先，MPS + peft 作为 fallback）+ CUDA + CPU

### 训练平台决策树

```
用户执行 pfe train
    │
    ├── CUDA GPU 可用？
    │   ├── 是 → unsloth（首选）→ 不支持当前模型？→ peft + trl（fallback）
    │   └── 否 ↓
    ├── Apple Silicon Mac？
    │   ├── 是 → mlx-lm 微调（原生 MLX 路径，无需量化，统一内存直接使用）
    │   │        注意：mlx-lm 有独立的 LoRA API，不走 peft/unsloth
    │   └── 否 ↓
    └── CPU only → peft + trl（CPU 模式，慢但可用，建议样本数 < 100）
```

> **关键决策：** Mac 用户走 `mlx-lm` 路径而非 MPS + peft。原因：(1) mlx-lm 对 Apple Silicon 做了深度优化，性能远优于 MPS 后端；(2) 避免 MPS 训练不稳定的已知问题；(3) mlx-lm 原生支持 LoRA 微调，API 简洁。代价是需要维护两套训练后端（unsloth/peft 和 mlx-lm），但接口层统一为 `Trainer`。

### SFT 训练流程

使用 `trl.SFTTrainer`（或 unsloth 封装）：
1. 加载基座模型 + LoRA 配置
2. 将 `TrainingSample`（sample_type="sft"）格式化为 `instruction → chosen` 对
3. 执行标准 SFT 训练循环
4. 保存 adapter 权重

### DPO 训练流程

使用 `trl.DPOTrainer`：
1. 加载基座模型 + 已有 SFT adapter（DPO 通常在 SFT 之后进行）
2. 将 `TrainingSample`（sample_type="dpo"）格式化为 `(instruction, chosen, rejected)` 三元组
3. 执行 DPO 训练（beta 参数控制偏离参考模型的程度）
4. 保存 adapter 权重

> **DPO 参考模型（reference model）：** 使用当前 SFT adapter 的冻结副本作为参考模型。DPO 训练时有两个角色：
> - **参考模型（reference model）**：`SFT adapter`（冻结，不更新权重）
> - **策略模型（policy model）**：从 `SFT adapter` 复制初始权重的新 adapter（可训练）
> 
> 训练目标是让策略模型学习偏好对齐，同时通过 KL 散度约束不要偏离参考模型太远。不使用裸基座模型作为参考，因为 SFT 后的分布更接近目标分布。

> **推荐训练顺序：** 先用 SFT 样本训练基础风格 adapter，再用 DPO 样本做偏好对齐。Phase 0 只实现 SFT，Phase 1 补充 DPO。

### 增量微调策略

增量微调是区别于一次性微调的关键能力：

| 问题 | 解法 |
|------|------|
| 灾难性遗忘 | Replay Buffer：混入 30% 历史样本（比例可配置） |
| 版本管理 | 每次训练产出新版本，支持回滚 |
| 触发条件 | 累积 N 条新样本后自动触发（默认 50 条） |
| 质量保障 | 每次训练后自动评测，退化则自动回滚 |

> **Replay Buffer 定义：** "历史样本"指 samples.db 中 `used_in_version IS NOT NULL` 的已训练样本。每次增量微调时，从**最近 5 个版本**的历史样本中按 score 加权随机采样 30%（可通过 `config.toml` 的 `replay_ratio` 配置），与新样本混合组成训练集。限制最近 5 个版本是为了避免过旧样本导致模型"回退"。历史样本跟随 samples.db 的保留策略（默认永久保留），确保 Replay Buffer 始终有足够的数据源。

### 训练触发机制

```python
class TrainTrigger:
    """判断是否应该触发一次新的微调"""

    def should_train(self) -> bool:
        new_samples = self.curator.count_new_samples()
        last_train_time = self.store.get_latest_version().created_at

        # 条件 1：新样本数 >= 阈值
        if new_samples >= self.config.min_new_samples:  # default: 50
            return True
        # 条件 2：距上次训练超过 N 天且有新样本
        if new_samples > 0 and (now() - last_train_time).days >= self.config.max_interval_days:  # default: 7
            return True
        return False
```

---

## 模块 4：Evaluator（效果评测层）

### 职责
量化微调效果，回答"这次微调有没有让模型更懂用户"。

### 三层评测体系

#### Level 1：自动评测（AutoEvaluator）

```python
class AutoEvaluator:
    """基于 LLM-as-a-Judge 的自动评测"""

    def evaluate(
        self,
        test_prompts: list[str],
        base_responses: list[str],       # 微调前
        adapted_responses: list[str],    # 微调后
        reference_responses: list[str]   # 用户历史真实回复（如有）
    ) -> EvalReport:
        """
        评测维度：
        - style_match: 风格匹配度（0-1）
        - preference_alignment: 偏好对齐度（0-1）
        - quality_preservation: 基础质量是否退化（0-1，>0.8 为合格）
        - personality_consistency: 人格一致性（0-1）
        """
```

> **默认实现：** `AutoEvaluator` 优先支持本地 Judge；若使用云端 Judge，必须显式开启 `privacy.allow_judge_cloud=true`，并在评测报告中记录 Judge 模型与出境次数。

### Judge 评测协议

为避免“teacher 既出题又判卷”导致评测失真，Judge 评测需要遵守以下协议：

1. 默认使用 holdout 测试集，不得直接评估训练集
2. `test` 切分中的蒸馏样本不能回流训练
3. 若条件允许，Teacher 模型和 Judge 模型应分离
4. 必须记录 `judge_model`、`judge_prompt_version`、`judge_run_id`
5. 对同一版本的评测尽量固定 prompt 模板和评分 rubric，确保横向可比

```python
@dataclass
class JudgeConfig:
    judge_model: str
    judge_prompt_version: str = "v1"
    require_holdout_split: bool = True
    forbid_teacher_test_overlap: bool = True


class JudgeProtocol:
    def prepare_eval_set(self, samples: list[TrainingSample]) -> list[TrainingSample]:
        """只挑选 holdout / test 切分样本"""

    def compare(self, base_output: str, adapted_output: str, reference: str | None) -> dict:
        """输出结构化评分与理由"""
```

#### Level 2：A/B 盲测（ABTestEvaluator）

```python
class ABTestEvaluator:
    """给用户展示两个回复（不标注来源），收集偏好"""

    def create_test(self, test_prompts: list[str]) -> ABTest:
        """生成盲测任务"""

    def collect_result(self, test_id: str, preferences: list[str]) -> EvalReport:
        """收集用户偏好，生成报告"""
```

**验证标准（补充立项文档缺失项）：**
- 盲测 A/B 中，用户对微调版的偏好率 ≥ 65% 视为验证通过
- quality_preservation ≥ 0.8（基础质量不退化）
- style_match 提升 ≥ 20%（相比基座模型）

#### Level 3：长期追踪（LongTermTracker）

```python
class LongTermTracker:
    """追踪用户行为指标随时间的变化"""
    # 追踪指标：接受率、编辑率、重新生成率、平均对话轮数
    # 按 adapter 版本分组，生成趋势图
    # 单独记录 teacher 数据占比与真实用户数据占比，防止长期漂移
```

### 评测决策逻辑

```python
def make_recommendation(report: EvalReport, prev_report: EvalReport | None) -> str:
    """
    Args:
        report: 当前 adapter 的评测报告
        prev_report: 上一版 adapter 的评测报告（首次训练时为 None）
    """
    if report.scores["quality_preservation"] < 0.8:
        return "keep_previous"  # 质量退化，不部署
    if prev_report is None:
        return "deploy"         # 首次训练，直接部署
    if report.scores["style_match"] > prev_report.scores["style_match"]:
        return "deploy"         # 风格匹配提升，部署
    if report.scores["preference_alignment"] > prev_report.scores["preference_alignment"]:
        return "deploy"         # 偏好对齐提升，部署
    return "needs_more_data"    # 无明显变化，继续积累数据
```

---

## 模块 5：Router（推理路由层）

### 职责
判断一个请求应该由本地模型处理还是路由到云端。

### 路由配置

```python
@dataclass
class RouterConfig:
    local_model: str                    # 本地模型 + adapter
    cloud_model: str                    # 云端模型 API
    strategy: str = "local_only"        # local_only | keyword | confidence | hybrid
    confidence_threshold: float = 0.7   # 低于此值路由到云端
    always_local: list[str] = []        # 强制本地的场景标签
    always_cloud: list[str] = []        # 强制云端的场景标签
```

### 路由策略（分阶段实现）

| 阶段 | 策略 | 描述 |
|------|------|------|
| Phase 0-1 | `local_only` | 默认只用本地模型，不做云端路由 |
| Phase 2 | 规则路由 | 基于关键词/场景标签匹配 |
| Phase 2 | 置信度路由 | 本地先生成，logprob 低于阈值转云端 |
| Phase 2+ | Fallback 路由 | 本地先试，质量不够自动 fallback |

### 路由判断逻辑

```python
class Router:
    def route(self, request: str, context: dict) -> RoutingDecision:
        """
        判断逻辑：
        1. 规则匹配：命中 always_local/always_cloud 直接决定
        2. 复杂度估算：基于输入长度、是否需要多步推理、是否涉及专业知识
        3. 置信度检测：本地模型先生成，如果 logprob 低于阈值，转云端
        """

    def route_with_fallback(self, request: str) -> Response:
        """本地先试，质量不够自动 fallback 到云端"""
```

> **注意：** "简单 vs 复杂"的判断本身是个隐藏难题——本地小模型可能高估自己的能力。Phase 2 需要专门设计路由准确率的评测方法。
>
> **隐私默认值：** 即使实现了 Router，只要 `privacy.allow_router_cloud=false`，所有请求仍强制本地处理。

---

## 模块 6：Adapter Store（Adapter 管理层）

### 职责
管理所有微调产出的 adapter，支持版本控制、切换、回滚。

### 核心接口

```python
class AdapterStore:
    def save(self, adapter: AdapterResult) -> str:
        """保存训练产物并置为 pending_eval，返回版本 ID"""

    def load(self, version: str = "latest") -> str:
        """加载指定版本 adapter，返回路径"""

    def rollback(self, version: str):
        """回滚到指定版本（更新 latest 软链接）"""

    def list_versions(self) -> list[AdapterMeta]:
        """列出所有版本及其评测指标"""

    def promote(self, version: str):
        """将某个 pending_eval / failed_eval 版本提升为 latest"""

    def prune(self, keep_last_n: int = 10):
        """清理旧版本，节省磁盘空间（默认保留 10 个版本）"""
```

### 自动回滚机制

```python
class AutoRollback:
    """训练后自动评测，退化则回滚"""

    def post_train_check(self, new_adapter: AdapterResult):
        report = self.evaluator.evaluate(new_adapter)
        if report.recommendation == "keep_previous":
            self.store.mark_failed_eval(new_adapter.version, report)
            self.store.rollback(self.store.get_previous_version())
            log.warning(f"Adapter {new_adapter.version} 质量退化，保留产物并继续服务上一版本")
        else:
            self.store.attach_eval_report(new_adapter.version, report)
            self.store.promote(new_adapter.version)
```

---

## 模块 7：Inference Engine（推理引擎层）

### 职责
负责模型加载、adapter 挂载、推理执行。是 Router 和 pfe-server 的底层依赖。

### 核心接口

```python
@dataclass
class InferenceConfig:
    base_model: str                    # 基座模型路径或 HF ID
    adapter_path: str | None = None    # adapter 路径（None 则使用基座模型）
    backend: str = "auto"              # auto | transformers | llama_cpp | mlx
    quantization: str = "4bit"         # 推理时的量化方式
    max_new_tokens: int = 512
    device: str = "auto"


class InferenceEngine:
    def __init__(self, config: InferenceConfig):
        """加载模型和 adapter"""

    def generate(self, messages: list[dict], **kwargs) -> str:
        """生成回复（兼容 OpenAI messages 格式）"""

    def generate_stream(self, messages: list[dict], **kwargs) -> Iterator[str]:
        """流式生成"""

    def swap_adapter(self, adapter_path: str):
        """热切换 adapter（不重新加载基座模型）"""

    def unload(self):
        """释放模型资源"""
```

> **产物约束：** `InferenceEngine` 只接受 manifest 已声明兼容当前后端的产物。对于 `llama.cpp`，必须先把 LoRA adapter 导出成 `gguf_merged`，不能直接加载 `safetensors` adapter。

### 推理后端决策树

```
pfe serve 启动
    │
    ├── 用户指定 backend？→ 使用指定后端
    └── auto 模式 ↓
        ├── Apple Silicon Mac？→ mlx-lm（原生 MLX 推理，性能最优）
        ├── CUDA GPU 可用？→ transformers + adapter 动态加载
        └── CPU only → llama.cpp（GGUF 格式，CPU 推理优化）
```

### 端口占用处理

执行 `pfe serve` 时，如果默认端口（8921）被占用：
1. **自动递增**：尝试 8922、8923... 直到找到可用端口
2. **提示用户**：输出实际使用的端口地址
3. **手动指定**：用户可通过 `--port` 参数显式指定端口

```bash
pfe serve --port 3000  # 显式指定端口
```

### Adapter 热切换

训练完成后，推理服务无需重启即可加载新 adapter：
1. `AdapterStore.save()` 完成后发送通知
2. `InferenceEngine.swap_adapter()` 加载新权重（LoRA 权重很小，通常 < 100MB）
3. 基座模型保持不变，仅替换 LoRA 层

> **注意：** llama.cpp 后端不支持动态 adapter 切换，需要重新合并模型。因此 llama.cpp 路径下 adapter 更新需要短暂的服务中断（< 30s）。
