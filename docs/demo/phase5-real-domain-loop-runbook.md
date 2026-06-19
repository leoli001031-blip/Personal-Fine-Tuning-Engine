# PFE Phase5 Real Domain Loop Runbook

Phase5 目标不是做展示页，而是证明一条真实资料驱动的 loop engineering：

```text
真实公开资料 -> 训练候选 -> 小批量真实训练 -> base/local 对比评测 -> correction/preference signal -> 下一轮 candidate plan
```

## 1. 场景边界

第一轮垂类场景：

- 合同摘要
- 风险标注
- 引用依据
- 需人工确认提示

明确边界：

- 不输出法律结论
- 不判断合法/违法
- 不替代律师或专业人士意见
- 只基于给定资料整理，不补外部事实
- 证据不足时输出需补充资料/需人工确认

## 2. 资料来源

Phase5 使用 Common Paper 标准协议作为第一批真实公开资料。Common Paper 标准协议页面说明其协议可在 CC BY 4.0 下使用和修改。

固定 source manifest 由 `pfe_core.phase5_real_domain_loop.COMMON_PAPER_CONTRACT_SOURCES` 生成，smoke 会写入：

```text
$PFE_HOME/phase5/workspaces/<workspace>/commonpaper-sources.json
```

本轮默认导入 10 个 source：

- Common Paper Cloud Service Agreement
- Common Paper Mutual NDA
- Common Paper Data Processing Agreement
- Common Paper Service Level Agreement
- Common Paper Professional Services Agreement
- Common Paper Business Associate Agreement
- Common Paper Software License Agreement
- Common Paper Design Partner Agreement
- Common Paper Pilot Agreement
- Common Paper AI Addendum

每个 source 记录：

- `title`
- `source_url`
- `page_url`
- `repo`
- `file_path`
- `license_status`
- `license_note`
- `retrieved_at`
- `domain`
- `risk_labels`
- `training_allowed`

未明确允许训练的 source 必须标记为 `training_allowed=false` 或 `user_review_required`，不得进入训练。

## 3. 数据集形态

Phase5 复用 Phase4 corpus store：

- source/chunk/provenance 仍由 Phase4 管理
- Phase5 只增加 curated source manifest、holdout prompts、eval report、loop evidence
- 训练候选仍导出到现有 samples DB

默认规模：

- source: 10
- training candidates: 60
- eligible training samples: 约 57
- train/val/test split: 约 45/6/6
- holdout prompts: 16，不进入训练样本库

## 4. 运行 Smoke

默认路径：导入真实资料、生成候选、导出样本、生成 holdout eval、记录 loop evidence，但不真实训练：

```bash
.venv/bin/python tools/phase5_real_domain_loop_smoke.py --timeout 180
```

真实 tiny 训练路径：

```bash
.venv/bin/python tools/phase5_real_domain_loop_smoke.py \
  --prepare-tiny-model \
  --strict-real \
  --timeout 180
```

成功输出应包含：

- `source_ingest.ingested_count=10`
- `candidate_count=60`
- `eligible_count` 在 40-80 范围内
- `sample_export.saved_samples`
- `sample_export.split_counts`
- `holdout_count=16`
- `eval_gate.status=pass` 或 `review`
- `route_summary.memory`
- `route_summary.profile`
- `route_summary.training_candidate`
- `route_summary.excluded`

真实训练路径还应包含：

- `real_training.real_training=completed`
- `real_training.adapter_version`
- `real_training.manifest_path`
- `real_training.real_execution_summary.kind=real_peft`
- `real_training.real_execution_summary.path=real_import`
- `real_training.real_execution_summary.num_examples`
- `real_training.real_execution_summary.train_loss`

## 5. 当前实测结果

2026-06-19 本地实测：

```text
source_ingest.ingested_count=10
candidate_count=60
eligible_count=57
sample_export.saved_samples=57
split_counts=train:45, val:6, test:6
holdout_count=16
eval_gate.status=pass
real_training=completed
real_execution_summary.kind=real_peft
real_execution_summary.path=real_import
real_execution_summary.num_examples=45
real_execution_summary.train_loss=~4.8
```

这证明的是链路真实性和可回放性，不等于证明 tiny 模型已有生产级合同理解能力。

## 6. Base/Local Eval

Phase5 eval 是一个离线、可重复的 loop-engineering eval，不调用大模型。

评估重点：

- 是否稳定输出 `摘要 / 风险提示 / 引用依据 / 人工确认`
- 是否保留 source/chunk citation
- 是否减少 unsupported assertions
- 是否避免法律结论
- 证据不足时是否转为拒绝推断/人工确认

机器可读报告：

```text
$PFE_HOME/phase5/workspaces/<workspace>/eval/phase5-real-domain-eval-report.json
```

人可读摘要：

```text
$PFE_HOME/phase5/workspaces/<workspace>/eval/phase5-real-domain-eval-summary.md
```

## 7. Loop Evidence

Phase5 smoke 会记录一轮闭环证据：

```text
$PFE_HOME/phase5/workspaces/<workspace>/loop-evidence.json
```

本轮记录三类信号：

- correction：进入 `memory` 和 `training_candidate`
- preference：进入 `profile`，若 repeated/confirmed 则进入 `training_candidate`
- safety_block：进入 `discard/review`，不进入训练

这对应：

```text
eval 发现 base 输出缺少 citation/结构化边界
-> correction/preference signal
-> route 到 memory/profile/training candidate
-> safety_block 被排除
-> 重新生成 Phase3/Phase4 candidate plan
```

## 8. 测试

相关测试：

```bash
.venv/bin/python -m pytest tests/test_phase5_real_domain_loop.py
```

目标验证：

- real source metadata/provenance
- holdout split 不进入训练
- high-risk / insufficient-evidence 样本处理
- base/local eval report schema
- loop signal routing

推荐完整验证：

```bash
.venv/bin/python -m pytest tests/test_phase5_real_domain_loop.py tests/test_phase4_real_train_smoke.py tests/test_phase4_real_corpus.py
.venv/bin/python tools/phase5_real_domain_loop_smoke.py --prepare-tiny-model --strict-real --timeout 180
.venv/bin/python tools/phase4_real_train_smoke.py --prepare-tiny-model --strict-real --timeout 120
git diff --check
make test-unit test-surface test-e2e-mock smoke-beta
```

## 9. 下一轮建议

下一轮可以从 eval report 中挑选失败或弱项：

- 引用命中不足
- 风险点结构不稳定
- 人工确认提示过泛
- 某些 safety case 没有拒绝推断

然后将人工 correction/preference signal 加入 signal inbox，再生成新的 candidate plan。大模型真实效果验证应使用更强的本地模型路径；tiny 模型只用于证明训练产物和接口闭环。
