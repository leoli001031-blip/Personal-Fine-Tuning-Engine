# ChatCollector 使用文档

ChatCollector 是 PFE 的信号采集模块，负责从用户对话交互中自动提取隐式反馈信号。

## 概述

ChatCollector 监听对话事件流，检测用户行为模式，并将这些行为转换为可用于训练的信号。

## 信号类型

### 1. 采纳信号 (Accept)

当用户接受了 assistant 的回复（无编辑、继续对话）时产生。

```python
{
    "event_type": "accept",
    "context": "用户输入",
    "model_output": "模型回复",
    "user_action": {
        "type": "accept",
        "accepted_text": "模型回复"
    }
}
```

**置信度**: 0.88 (高)

### 2. 拒绝信号 (Reject)

当用户删除或明确拒绝回复时产生。

```python
{
    "event_type": "reject",
    "context": "用户输入",
    "model_output": "模型回复",
    "user_action": {
        "type": "reject",
        "rejected_text": "模型回复"
    }
}
```

**置信度**: 0.56 (中)

### 3. 编辑信号 (Edit)

当用户编辑了回复时产生。编辑距离作为质量指标。

```python
{
    "event_type": "edit",
    "context": "用户输入",
    "model_output": "原始回复",
    "user_action": {
        "type": "edit",
        "original_text": "原始回复",
        "edited_text": "用户编辑后的文本"
    }
}
```

**置信度**: 0.76 (中高)

### 4. 重生成信号 (Regenerate)

当用户要求重新生成回复时产生。

```python
{
    "event_type": "regenerate",
    "context": "用户输入",
    "model_output": "模型回复",
    "user_action": {
        "type": "regenerate"
    }
}
```

**置信度**: 0.60 (中)

## 事件链追踪

ChatCollector 维护完整的事件链以确保信号质量：

```
session_id → request_id → source_event_ids → event_id
```

- **session_id**: 对话会话标识
- **request_id**: 单次请求标识
- **source_event_id**: 源事件标识（用于追踪对话历史）
- **event_id**: 当前事件唯一标识

## 信号质量评估

每个信号都有质量评分，基于以下因素：

1. **回复类型**: accept > edit > reject > other
2. **事件链完整性**: 完整的事件链增加置信度
3. **文本完整性**: 非空文本增加置信度
4. **用户行为明确性**: 明确的用户操作增加置信度

### 质量过滤配置

```yaml
[signal_quality]
minimum_confidence = 0.65          # 最低置信度阈值
reject_conflicted_signal_quality = true  # 是否拒绝矛盾信号
minimum_signal_length = 5          # 最小信号长度
maximum_signal_length = 4096       # 最大信号长度
require_complete_event_chain = true # 要求完整事件链
require_user_action = true          # 要求用户行为
```

## CLI 命令

### 查看采集统计

```bash
pfe collect status
```

输出示例：
```json
{
  "total_signals": 150,
  "accepted": 120,
  "rejected": 15,
  "edited": 15,
  "avg_confidence": 0.82,
  "filtered_signals": 10
}
```

### 人工审核信号

```bash
pfe collect review
```

交互式审核提取的信号，可以：
- 确认信号质量
- 标记错误信号
- 调整置信度

### 启停信号采集

```bash
# 启动采集
pfe collect start

# 停止采集
pfe collect stop
```

## HTTP API

### 提交信号

```bash
POST /pfe/signal
Content-Type: application/json

{
    "event_type": "accept",
    "session_id": "session-123",
    "request_id": "request-456",
    "context": "用户输入",
    "model_output": "模型回复",
    "user_action": {
        "type": "accept",
        "accepted_text": "模型回复"
    }
}
```

### 查询信号

```bash
GET /pfe/signals?session_id=session-123&limit=50
```

## 集成测试

ChatCollector 的集成测试位于 `tests/test_e2e_collect_train_loop.py`：

```bash
# 运行信号采集集成测试
pytest tests/test_e2e_collect_train_loop.py -v
```

测试场景包括：
1. 信号采集自动触发训练
2. 信号质量过滤
3. 对话会话追踪
4. 边界条件处理

## 配置示例

完整的 ChatCollector 配置：

```yaml
[chat_collector]
enabled = true
auto_submit = true
confidence_threshold = 0.65

[chat_collector.extractors]
accept.enabled = true
reject.enabled = true
edit.enabled = true
edit.similarity_threshold = 0.8
regenerate.enabled = true

[chat_collector.session]
timeout_seconds = 300
max_turns = 50
```

## 注意事项

1. **隐私保护**: 信号采集遵循 `strict_local` 模式，所有数据本地处理
2. **数据保留**: 信号默认保留 90 天，可通过配置调整
3. **质量优先**: 低质量信号会被过滤，不会进入训练集
4. **事件链完整性**: 不完整的信号会降低置信度，但不会被丢弃

## 故障排查

### 信号未被采集

检查：
1. ChatCollector 是否已启动 (`pfe collect status`)
2. 信号质量是否达到阈值
3. 事件链是否完整

### 信号质量过低

可能原因：
1. 缺少 `session_id` 或 `request_id`
2. 文本内容过短
3. 用户行为不明确

### 训练未被触发

检查：
1. 信号数量是否达到 `train_trigger.min_samples`
2. 自动触发是否启用 (`train_trigger.enabled`)
3. 冷却期是否已过
