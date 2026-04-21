# DPO 训练使用文档

DPO (Direct Preference Optimization) 训练允许模型直接从用户偏好信号中学习，无需显式的奖励模型。

## 概述

DPO 训练使用 preference pairs（偏好对）进行训练：
- **Chosen**: 用户偏好的回复
- **Rejected**: 用户不喜欢的回复

PFE 支持两种 DPO 数据来源：
1. **Accept/Reject pairs**: 同一问题的接受和拒绝回复
2. **Edit pairs**: 原始回复和用户编辑后的回复

## 数据格式

### Preference Pair 格式

```python
{
    "prompt": "用户输入",
    "chosen": "用户偏好的回复",
    "rejected": "用户不喜欢的回复"
}
```

### 从信号构建 Preference Pairs

PFE 自动从以下信号组合构建 preference pairs：

#### 1. Accept + Reject

同一 `request_id` 下的接受和拒绝信号：

```python
# Signal 1: Accept
{
    "event_type": "accept",
    "request_id": "req-123",
    "context": "Explain Python",
    "model_output": "Python is a high-level programming language...",
    "user_action": {"type": "accept"}
}

# Signal 2: Reject
{
    "event_type": "reject",
    "request_id": "req-123",
    "context": "Explain Python",
    "model_output": "Python is a snake.",
    "user_action": {"type": "reject"}
}
```

#### 2. Edit pairs

编辑信号自动构建为 preference pair：

```python
{
    "event_type": "edit",
    "request_id": "req-456",
    "context": "Write a greeting",
    "model_output": "Hello.",  # rejected (original)
    "user_action": {
        "type": "edit",
        "edited_text": "Hello! Welcome! How can I help you today?"  # chosen
    }
}
```

## CLI 命令

### 运行 DPO 训练

```bash
# 基础 DPO 训练
pfe train --method dpo

# 指定基础 adapter
pfe train --method dpo --base-adapter v001

# 使用特定配置
pfe train --method dpo --config dpo_config.yaml
```

### 查看 DPO 训练状态

```bash
pfe training status
```

### 创建 Preference Pairs

```bash
# 从现有信号创建 preference pairs
pfe curator build-preference-pairs

# 查看创建的 pairs
pfe curator list-preference-pairs
```

## 配置

### DPO 训练配置

```yaml
[training]
method = "dpo"           # 训练方法: sft 或 dpo
backend = "peft"         # 训练后端: peft, unsloth, mlx
base_model = "Qwen/Qwen2.5-3B-Instruct"

[dpo]
beta = 0.1               # DPO 温度参数，控制与参考模型的偏离程度
label_smoothing = 0.0    # 标签平滑
max_length = 512         # 最大序列长度
max_prompt_length = 128  # 最大 prompt 长度

[lora]
r = 16
alpha = 32
dropout = 0.05
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

[training.hyperparameters]
epochs = 3
batch_size = 4
learning_rate = 0.0001
warmup_ratio = 0.1
```

### Beta 参数说明

- **beta = 0.1** (默认): 较小的偏离，保持与参考模型接近
- **beta = 0.5**: 中等偏离，允许更多学习
- **beta = 0.01**: 较小的偏离，更保守的学习

较小的 beta 值产生更保守的更新，较大的值允许模型更积极地适应偏好。

## SFT → DPO 渐进训练

推荐流程：先 SFT 后 DPO

```bash
# Step 1: SFT 训练
pfe train --method sft --epochs 3

# Step 2: 使用 SFT adapter 作为基础进行 DPO
pfe train --method dpo --base-adapter latest
```

### 渐进训练的优势

1. **SFT 阶段**: 学习基础知识和格式
2. **DPO 阶段**: 学习用户偏好和对齐

## HTTP API

### 提交 DPO 训练任务

```bash
POST /pfe/training/jobs
Content-Type: application/json

{
    "method": "dpo",
    "base_adapter": "v001",
    "config": {
        "beta": 0.1,
        "epochs": 3
    }
}
```

### 创建 Preference Pairs

```bash
POST /pfe/curator/preference-pairs
Content-Type: application/json

{
    "pairs": [
        {
            "prompt": "Explain quantum computing",
            "chosen": "Quantum computing uses qubits...",
            "rejected": "It's complicated quantum stuff."
        }
    ]
}
```

## 评估

### DPO 评估指标

DPO 训练后的评估包含以下指标：

```json
{
    "preference_alignment": 0.75,      # 偏好对齐度
    "chosen_win_rate": 0.68,           # chosen 回复胜率
    "rejection_rate": 0.15,            # 拒绝率
    "style_consistency": 0.82          # 风格一致性
}
```

### 运行评估

```bash
# 评估 DPO adapter
pfe eval --version v002

# 对比 SFT 和 DPO 版本
pfe eval --compare v001 v002
```

## 集成测试

DPO 训练的集成测试位于 `tests/test_e2e_dpo_pipeline.py`：

```bash
# 运行 DPO 集成测试
pytest tests/test_e2e_dpo_pipeline.py -v
```

测试场景包括：
1. DPO 训练完整流程
2. Preference pair 构建
3. 编辑回复处理
4. SFT → DPO 渐进训练

## 故障排查

### 训练失败：缺少 preference pairs

检查：
1. 是否有足够的 accept/reject 信号对
2. 信号是否有相同的 `request_id`
3. 事件链是否完整

### 训练发散

可能原因：
1. Beta 值过大，尝试减小到 0.05
2. 学习率过高
3. Preference pairs 质量不一致

### 效果不明显

可能原因：
1. Preference pairs 数量不足（建议至少 50 对）
2. Chosen 和 rejected 差异不够明显
3. Beta 值过小

## 最佳实践

### 1. 数据质量

- 确保 preference pairs 有明确的质量差异
- 过滤低置信度的信号
- 保持 chosen 和 rejected 的多样性

### 2. 渐进训练

- 先进行 SFT 建立基础能力
- 再进行 DPO 学习偏好
- 避免直接从 base model 进行 DPO

### 3. 超参数调优

```yaml
# 保守配置（推荐）
dpo.beta = 0.1
training.learning_rate = 0.0001
training.epochs = 3

# 积极配置
dpo.beta = 0.2
training.learning_rate = 0.0002
training.epochs = 5
```

### 4. 评估策略

- 使用 holdout test set 评估
- 对比 SFT-only 和 SFT+DPO 的效果
- 监控 preference alignment 指标

## 参考

- [Direct Preference Optimization Paper](https://arxiv.org/abs/2305.18290)
- [TRL DPO Trainer Documentation](https://huggingface.co/docs/trl/dpo_trainer)
