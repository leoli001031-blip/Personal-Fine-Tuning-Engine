# PFE Phase7 Real Signal Training Runbook

Phase7 验证 PFE 的核心产品假设：训练不是一次性上传资料，而是持续收集交互信号，只把有价值、可追溯、低风险的信号变成小批量候选训练。

```text
public sources -> source manifest + PII/license gate
-> persona/scenario interaction -> signal inbox
-> eligible signal routing -> candidate samples
-> Qwen/MLX adapter trial -> holdout eval
-> promote/archive/collect more signal
```

## 1. 产品目标

Phase7 不扩 UI，重点验证真实训练闭环：

- 公开资料必须可引用、可追溯、可重新拉取。
- `accept` / `edit` / `correction` / reinforced `preference` 可以进入训练候选。
- `reject` 不能单独训练，`safety_block` 必须排除。
- PII、安全和高风险行业边界先过 gate，再考虑训练。
- holdout prompt 必须隔离，不能进入训练样本。
- 没有真实 adapter 和真实 base/local holdout 生成时，不能 promote。

## 2. Source Manifest

默认数据来自 Common Paper 公开标准和样例协议，目标是合同摘要和风险提示资料整理，不输出法律结论。

默认 smoke 当前实测：

```text
source_count=11
training_allowed=10
review_only=1
review_only_source=cp-csa
review_only_reason=pii_audit_high
```

`cp-csa` 会被收集进 manifest，但因为 PII audit 为 high，只能 review，不进入训练。

## 3. Signal Routing

Phase7 smoke 会合成一组代表真实交互反馈的 signal：

```text
signal_types=accept,reject,edit,correction,preference,safety_block
signal_count=6
eligible_count=4
training_candidate=accept,edit,correction,reinforced_preference
excluded=reject,safety_block
```

路由规则：

- `memory`：用户确认过的稳定任务偏好或可复用输出格式。
- `profile`：长期偏好，例如引用格式、安全提示结构。
- `training_candidate`：有正向目标输出、可追溯来源、低风险的信号。
- `excluded`：PII、高风险结论、安全阻断、单独 reject、holdout。

## 4. Candidate Samples

默认 smoke 会把 eligible signals 和可训练来源变成 holdout-free 样本：

```text
candidate_samples=40
split_counts.train=34
split_counts.val=6
split_counts.test=0
holdout_count=16
holdout.not_for_training=true
```

训练样本必须保留：

- source id
- chunk id
- provenance
- signal id
- persona/scenario context
- safety boundary

## 5. Default Smoke

默认 smoke 不下载 Qwen3.6 权重，不做真实训练：

```bash
.venv/bin/python tools/phase7_real_signal_training_smoke.py \
  --timeout 240 \
  --evidence-dir docs/demo/phase7-real-training/evidence \
  --clean-evidence
```

当前本地实测：

```text
preflight.status=needs_model_download
training.real_training=not_started
eval.real_model_calls=false
eval_gate.status=blocked
decision.action=archive
```

这是正确结果：默认 smoke 只证明闭环和 gate，不声称模型效果提升。

## 6. Real Model Trial

Phase7 分两档模型执行：

- 目标模型：`mlx-community/Qwen3.6-27B-4bit`
- 首轮闭环验证模型：`mlx-community/Qwen3-0.6B-4bit`

27B 用来验证目标部署方向和机器边界；0.6B 用来快速验证真实训练、adapter 生成、base/adapter eval 和 gate。

### 6.1 Target Model Boundary

目标模型：

```text
mlx-community/Qwen3.6-27B-4bit
```

上游基座：

```text
Qwen/Qwen3.6-27B
```

本机真实试验结果：

```text
model=mlx-community/Qwen3.6-27B-4bit
download=completed
training=blocked
failure_category=killed_oom
stderr=METAL Command buffer execution failed: Insufficient Memory
decision=archive
```

证据目录：

```text
docs/demo/phase7-real-training/evidence-qwen36-27b-oom/
```

结论：27B 适合保留为 Phase7/Phase8 目标模型，但不适合作为首轮真实闭环验证模型。继续硬跑 27B 不符合产品目标，因为这一阶段要验证 signal-gated training loop，而不是消耗时间在大模型显存边界上。

### 6.2 First-Pass Real Trial

首轮真实闭环验证模型：

```text
mlx-community/Qwen3-0.6B-4bit
```

命令：

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python tools/phase7_real_signal_training_smoke.py \
  --model-id mlx-community/Qwen3-0.6B-4bit \
  --allow-remote-download \
  --run-real-training \
  --run-real-eval \
  --strict-real \
  --strict-real-eval \
  --candidate-limit 12 \
  --holdout-count 4 \
  --eval-samples 1 \
  --eval-max-tokens 120 \
  --timeout 1800 \
  --evidence-dir docs/demo/phase7-real-training/evidence-real-qwen3-0.6b \
  --clean-evidence \
  --keep-workdir
```

当前本地实测：

```text
training.real_training=completed
train_log.returncode=0
eval.real_model_calls=true
candidate_samples=12
split_counts.train=10
split_counts.val=2
holdout_count=4
adapter.citation_hit_rate=0.75
adapter.structure_hit_rate=0.75
adapter.safety_boundary_rate=0.75
decision.action=archive
```

结论：真实训练和真实 base/adapter eval 已跑通，但 adapter 没有达到 promotion 阈值，所以 archive 是正确结果。下一步应该收集更多高质量 correction/edit signals，而不是降低 gate 标准。

首次下载大模型或目标模型时建议禁用 Hugging Face Xet 传输，避免大文件下载中断：

```bash
HF_HUB_DISABLE_XET=1 .venv/bin/python tools/phase7_real_signal_training_smoke.py \
  --allow-remote-download \
  --run-real-training \
  --run-real-eval \
  --strict-real \
  --strict-real-eval \
  --eval-samples 1 \
  --eval-max-tokens 120 \
  --timeout 7200 \
  --evidence-dir docs/demo/phase7-real-training/evidence-real \
  --clean-evidence \
  --keep-workdir
```

真实训练必须产出：

- MLX adapter artifact
- adapter manifest
- training job result
- base holdout output
- adapter holdout output
- eval report
- promote/archive decision

如果缺少这些证据，Phase7 只能报告为 preflight 或 training attempt，不能报告为真实效果提升。

## 7. Eval Gate

Promotion 条件：

```text
real training completed
real base/local model calls completed
candidate samples preserve source/chunk/provenance/signal_id
holdout not used for training
adapter beats base on holdout structure/citation/safety criteria
unsupported assertions do not increase
```

否则 decision 必须是：

- `archive`
- `collect_more_signal`
- `fix_preflight`

## 8. Evidence

默认 evidence：

```text
docs/demo/phase7-real-training/evidence/source_manifest.json
docs/demo/phase7-real-training/evidence/source_ingest.json
docs/demo/phase7-real-training/evidence/signal_evidence.json
docs/demo/phase7-real-training/evidence/candidate_samples.jsonl
docs/demo/phase7-real-training/evidence/holdout.json
docs/demo/phase7-real-training/evidence/trial_manifest.json
docs/demo/phase7-real-training/evidence/training_attempt.json
docs/demo/phase7-real-training/evidence/eval_report.json
docs/demo/phase7-real-training/evidence/decision.json
docs/demo/phase7-real-training/evidence/summary.md
```

真实训练 evidence：

```text
docs/demo/phase7-real-training/evidence-qwen36-27b-oom/
docs/demo/phase7-real-training/evidence-real-qwen3-0.6b/
```

`evidence-qwen36-27b-oom` 证明 27B 目标模型当前受 Metal memory 限制；`evidence-real-qwen3-0.6b` 证明首轮真实训练闭环已经跑通，但 gate 正确拒绝 promote。

## 9. Tests

Targeted tests：

```bash
.venv/bin/python -m pytest tests/test_phase7_real_signal_training.py
```

Recommended validation：

```bash
.venv/bin/python -m py_compile \
  pfe-core/pfe_core/phase7_real_signal_training.py \
  pfe-core/pfe_core/trainer/mlx_backend.py \
  tools/phase7_real_signal_training_smoke.py \
  tests/test_phase7_real_signal_training.py

.venv/bin/python -m pytest tests/test_phase7_real_signal_training.py
.venv/bin/python tools/phase7_real_signal_training_smoke.py --timeout 240
git diff --check
make test-unit test-surface test-e2e-mock smoke-beta
```

## 10. Known Limits

- 默认 smoke 不下载 Qwen3.6 权重。
- 默认 eval 是 deterministic gate evidence，不是模型调用效果。
- 真实训练需要显式 `--run-real-training` 和 `--run-real-eval`。
- 本轮不修改 `videos/`。
- 不提交模型权重、HF cache 或大体积 adapter artifact。
