# Auto-Train/Eval/Promote 自动闭环策略配置

本文档详细说明 PFE (Personal Finetune Engine) 的自动闭环策略配置系统。

## 概述

PFE 使用分层的策略配置来控制自动训练、评测和晋升的闭环流程。策略配置分为五个主要部分：

1. **Train Trigger Policy** - 训练触发策略
2. **Eval Gate Policy** - 评测门控策略
3. **Promote Gate Policy** - 晋升门控策略
4. **Confirmation Policy** - 人工确认策略
5. **Queue Review Policy** - 队列审核策略

## 配置结构

```toml
[trainer.trigger]
enabled = false
min_new_samples = 50
max_interval_days = 7
auto_evaluate = false
auto_promote = false

[trainer.trigger.train_trigger_policy]
enabled = false
min_new_samples = 50
max_interval_days = 7
min_trigger_interval_minutes = 60
failure_backoff_minutes = 30
consecutive_failure_threshold = 3
consecutive_failure_backoff_multiplier = 2.0
max_queue_depth = 10
pause_on_queue_full = true

[trainer.trigger.train_trigger_policy.signal_quality_gate]
minimum_confidence = 0.65
reject_conflicted_signal_quality = true
minimum_signal_length = 5
maximum_signal_length = 4096
require_complete_event_chain = true
require_user_action = true

[trainer.trigger.eval_gate_policy]
auto_trigger = false
trigger_delay_seconds = 0.0
eval_split_ratio = 0.2
min_eval_samples = 5
max_eval_samples = 200
eval_frequency_hours = 24
re_evaluate_on_promote = false
require_holdout_split = true
forbid_teacher_test_overlap = true

[trainer.trigger.promote_gate_policy]
auto_promote = false
min_quality_score = 0.7
min_style_match_score = 0.6
min_preference_alignment_score = 0.6
min_quality_preservation_score = 0.8
require_eval_recommendation_deploy = true
compare_with_previous = true
min_improvement_delta = 0.05
require_manual_confirm_on_regression = true
max_promote_frequency_hours = 1

[trainer.trigger.confirmation_policy]
first_training_requires_confirm = true
quality_regression_requires_confirm = true
rapid_trigger_requires_confirm = true
rapid_trigger_threshold_minutes = 30
queue_confirmation_default_approved = false
auto_approve_below_quality_threshold = false

[trainer.trigger.queue_review_policy]
default_review_mode = "auto_approve"
priority_policy = "hybrid"
quality_score_weight = 0.3
batch_size = 5
max_concurrent_jobs = 1
auto_retry_failed = false
max_retry_attempts = 2
retry_backoff_minutes = 10
```

## 策略详解

### 1. Train Trigger Policy (训练触发策略)

控制何时自动触发训练。

#### 信号数量阈值
- `min_new_samples`: 默认 50 条信号触发训练
- `max_interval_days`: 最长 7 天，超过后即使样本不足也会触发

#### 质量阈值
- `signal_quality_gate.minimum_confidence`: 0.65，低于此值的信号会被丢弃
- `signal_quality_gate.reject_conflicted_signal_quality`: true，冲突信号是否丢弃

#### Cooldown 策略
- `min_trigger_interval_minutes`: 60 分钟，两次训练间隔至少多久
- `failure_backoff_minutes`: 30 分钟，失败后冷却时间
- `consecutive_failure_threshold`: 3 次，连续失败阈值
- `consecutive_failure_backoff_multiplier`: 2.0，失败退避倍数

#### 最大队列深度
- `max_queue_depth`: 10，超过此值暂停自动触发
- `pause_on_queue_full`: true，队列满时是否暂停

### 2. Eval Gate Policy (评测门控策略)

控制何时自动触发评测。

#### 自动触发时机
- `auto_trigger`: false，默认不自动触发
- `trigger_delay_seconds`: 0.0，训练完成后延迟多久触发

#### 评测样本选择
- `eval_split_ratio`: 0.2，holdout/test 比例
- `min_eval_samples`: 5，最少评测样本数
- `max_eval_samples`: 200，最多评测样本数

#### 评测频率
- `eval_frequency_hours`: 24 小时，定期重新评测频率
- `re_evaluate_on_promote`: false，晋升时是否重新评测

### 3. Promote Gate Policy (晋升门控策略)

控制模型晋升的质量门槛。

#### 质量门槛
- `min_quality_score`: 0.7，整体质量评分
- `min_style_match_score`: 0.6，风格匹配度
- `min_preference_alignment_score`: 0.6，偏好对齐度
- `min_quality_preservation_score`: 0.8，质量保持度（最重要）

#### Compare Gate
- `compare_with_previous`: true，是否与前版本比较
- `min_improvement_delta`: 0.05，最小改进幅度
- `require_manual_confirm_on_regression`: true，质量下降时是否需要人工确认

#### 自动/人工确认
- `auto_promote`: false，默认不自动晋升
- `require_eval_recommendation_deploy`: true，需要评测推荐为 "deploy"
- `max_promote_frequency_hours`: 1，晋升频率限制

### 4. Confirmation Policy (确认策略)

控制何时需要人工确认。

- `first_training_requires_confirm`: true，首次训练需要确认
- `quality_regression_requires_confirm`: true，质量下降时需要确认
- `rapid_trigger_requires_confirm`: true，连续快速触发时需要确认
- `rapid_trigger_threshold_minutes`: 30，快速触发阈值（分钟）
- `queue_confirmation_default_approved`: false，队列确认默认是否批准

### 5. Queue Review Policy (队列审核策略)

控制队列处理行为。

#### 默认行为
- `default_review_mode`: "auto_approve"，默认自动批准

#### 优先级策略
- `priority_policy`: "hybrid"，优先级策略
  - `fifo`: 先进先出
  - `quality_score`: 按质量分数排序
  - `hybrid`: 混合策略（年龄 + 质量权重）
- `quality_score_weight`: 0.3，质量分数权重

#### 批量策略
- `batch_size`: 5，一次处理多少条
- `max_concurrent_jobs`: 1，最大并发任务数

## 使用策略评估函数

```python
from pfe_core.config import PFEConfig
from pfe_core.trainer.policy import (
    evaluate_train_trigger_policy,
    evaluate_eval_gate_policy,
    evaluate_promote_gate_policy,
    evaluate_confirmation_policy,
    evaluate_queue_review_policy,
    build_policy_summary,
)

# 加载配置
config = PFEConfig.load()
trigger_config = config.trainer.trigger

# 评估训练触发策略
trigger_result = evaluate_train_trigger_policy(
    trigger_config.train_trigger_policy,
    eligible_samples=60,
    days_since_last_training=2.0,
    last_trigger_at=None,
    last_failure_at=None,
    consecutive_failures=0,
    current_queue_depth=2,
)
print(f"Ready: {trigger_result['ready']}")
print(f"Blocked reasons: {trigger_result['blocked_reasons']}")

# 评估评测门控策略
eval_result = evaluate_eval_gate_policy(
    trigger_config.eval_gate_policy,
    holdout_samples=20,
    last_eval_at=None,
    training_completed_at=None,
)
print(f"Should eval: {eval_result['should_eval']}")

# 评估晋升门控策略
promote_result = evaluate_promote_gate_policy(
    trigger_config.promote_gate_policy,
    eval_scores={
        "overall": 0.75,
        "style_match": 0.65,
        "preference_alignment": 0.70,
        "quality_preservation": 0.85,
    },
    eval_recommendation="deploy",
    previous_scores=None,
    last_promote_at=None,
)
print(f"Should promote: {promote_result['should_promote']}")

# 获取策略摘要
summary = build_policy_summary(trigger_config, workspace="user_default")
print(summary)
```

## 设计取舍

### 1. 向后兼容性
- 新的策略配置嵌套在现有的 `TrainerTriggerConfig` 中
- 原有字段（如 `enabled`, `min_new_samples`）仍然有效
- 新旧配置可以共存，旧代码无需修改

### 2. 默认保守策略
- 所有自动功能默认关闭（`enabled = false`）
- 质量门槛设置较高（quality_preservation >= 0.8）
- 人工确认默认开启（首次训练、质量下降）

### 3. 分层策略
- 信号层：SignalQualityGate 过滤低质量信号
- 触发层：TrainTriggerPolicy 控制训练时机
- 评测层：EvalGatePolicy 控制评测行为
- 晋升层：PromoteGatePolicy 控制晋升质量
- 确认层：ConfirmationPolicy 控制人工介入

### 4. 可观测性
- 每个策略评估函数返回详细的 `policy_summary`
- 阻塞原因明确分类（数据不足、冷却中、质量不达标等）
- 支持构建完整的策略状态快照

## 迁移指南

### 从旧配置迁移

旧配置：
```toml
[trainer.trigger]
enabled = true
min_new_samples = 30
auto_evaluate = true
auto_promote = true
```

新配置（等效）：
```toml
[trainer.trigger]
enabled = true
min_new_samples = 30
auto_evaluate = true
auto_promote = true

[trainer.trigger.train_trigger_policy]
enabled = true
min_new_samples = 30

[trainer.trigger.eval_gate_policy]
auto_trigger = true

[trainer.trigger.promote_gate_policy]
auto_promote = true
```

### 推荐配置（生产环境）

```toml
[trainer.trigger]
enabled = true
queue_mode = "deferred"
require_queue_confirmation = true

[trainer.trigger.train_trigger_policy]
min_new_samples = 100
min_trigger_interval_minutes = 120
failure_backoff_minutes = 60

[trainer.trigger.eval_gate_policy]
auto_trigger = true
trigger_delay_seconds = 30

[trainer.trigger.promote_gate_policy]
auto_promote = false  # 生产环境建议人工确认
require_manual_confirm_on_regression = true

[trainer.trigger.confirmation_policy]
first_training_requires_confirm = true
quality_regression_requires_confirm = true
```
