# PFE Phase4 Real Corpus Runbook

Phase4 的目标是把真实资料纳入微调闭环：采集资料、生成可审计训练候选、导出到现有 SFT 样本库、生成 candidate adapter，并用 base/local 对比评测检查是否真的改善。

## 1. 准备资料

支持三类本地文件：

- `md`
- `txt`
- `pdf`，需要本机 Python 环境可读取文本型 PDF；如果缺少 `pypdf` 或 PDF 没有可抽取文本，ingest 会给出明确错误

URL 只支持单个、小范围、可审计导入，不做 crawler。URL 默认标记为 `user_review_required`，需要人工确认来源许可和内容质量。

每个 source 会保存：

- `source_id`
- `title`
- `source_path` 或 `source_url`
- `source_type`
- `content_hash`
- `license_status`
- `created_at`
- `metadata`

每个 chunk 会保存：

- `chunk_id`
- `source_id`
- `text`
- `char_count`
- `token_count`
- `provenance`
- `safety_flags`

## 2. 采集 corpus

通过 API 导入本地资料：

```bash
curl http://127.0.0.1:8921/pfe/phase4/sources \
  -H "content-type: application/json" \
  -d '{"path":"/absolute/path/to/research.md","title":"Research notes","license_status":"local_user_provided"}'
```

查看状态：

```bash
curl http://127.0.0.1:8921/pfe/phase4
curl http://127.0.0.1:8921/pfe/phase4/chunks
```

Studio 里可以用“真实资料闭环”面板查看：

- Sources
- Chunks
- Candidates
- Adapter
- Delta
- Gate

## 3. 生成训练候选

Phase4 从 corpus chunks 生成四类候选：

- summary samples
- citation-grounded answer samples
- structured notes samples
- insufficient-evidence refusal samples

生成并导出 JSONL：

```bash
curl http://127.0.0.1:8921/pfe/phase4/training-candidates \
  -H "content-type: application/json" \
  -d '{"limit":12,"export":true}'
```

每条候选保留：

- `source_ids`
- `chunk_ids`
- `provenance`
- `safety_metadata`
- `eligible_for_training`
- `excluded_reason`

## 4. 安全与质量排除

候选进入训练前会检查：

- PII audit
- high-risk domain labels: `legal`、`medical`、`financial`
- source/chunk provenance 是否完整
- chunk 是否低质量
- 是否包含法律、医学、金融确定性结论

排除示例：

- 包含手机号、邮箱、证件号、银行账户等高风险 PII：`pii_audit_blocked`
- 缺少 source/chunk/provenance：`missing_provenance`
- chunk 过短或不可用：`low_quality_chunk`
- 输出或来源包含确定性法律/医学/金融结论：`high_risk_deterministic_conclusion`

高风险行业资料不一定全部排除；如果输出只做资料整理、风险提示和人工确认提醒，可以保留为训练候选。

## 5. 导出到训练样本库

导出 eligible candidates 到现有 SFT samples DB：

```bash
curl http://127.0.0.1:8921/pfe/phase4/training-candidates/export \
  -H "content-type: application/json" \
  -d '{"target":"samples_db"}'
```

导出样本会使用现有训练格式：

- `sample_type=sft`
- `source=signal`
- `dataset_split=train/val/test`
- `metadata.phase=phase4`

这样 `/pfe/training/jobs` 可以复用现有训练 job。

## 6. 训练或明确 skip

真实小批量训练优先使用已有 training job：

```bash
curl http://127.0.0.1:8921/pfe/training/jobs \
  -H "content-type: application/json" \
  -d '{"method":"sft"}'
```

确认后启动：

```bash
curl http://127.0.0.1:8921/pfe/training/jobs \
  -H "content-type: application/json" \
  -d '{"method":"sft","confirm":true}'
```

如果当前机器没有配置可训练小模型，可以运行 Phase4 real train smoke。它会明确输出 skip reason，并用现有 adapter store 生成一个 mock fallback candidate adapter：

```bash
python tools/phase4_real_train_smoke.py
```

要尝试真实训练，需要显式设置本地模型路径：

```bash
PFE_PHASE4_REAL_TRAIN_MODEL=/absolute/path/to/Qwen2.5-0.5B-Instruct \
  python tools/phase4_real_train_smoke.py
```

如果模型路径不存在或 preflight 未就绪，smoke 会输出 `real_training=skipped` 和具体原因。

## 7. Base/Local 对比评测

Phase4 eval 对同一批 holdout prompts 生成 base/local 对比报告：

```bash
curl http://127.0.0.1:8921/pfe/phase4/eval \
  -H "content-type: application/json" \
  -d '{"adapter_version":"20260619-001","attach_to_adapter":true}'
```

报告路径：

```text
$PFE_HOME/phase4/workspaces/<workspace>/eval/phase4-eval-report.json
```

评估指标：

- citation hit rate
- summary coverage
- unsupported assertions
- refusal / insufficient-evidence boundary
- local delta

`eval_gate.status` 会输出：

- `pass`
- `review`
- `fail`

如果 attach 到 adapter，报告也会写入现有 adapter store，供 promote/archive gate 使用。

## 8. Smoke

Corpus 和 candidate 导出：

```bash
python tools/phase4_corpus_smoke.py
```

Base/local eval：

```bash
python tools/phase4_eval_smoke.py
```

训练 handoff 或明确 skip：

```bash
python tools/phase4_real_train_smoke.py
```

## 9. Phase4 成功判断

Phase4 最小成功标准：

- 可以导入真实资料
- 可以生成 source/chunk，并保留 provenance
- 可以生成带 source/chunk 引用的训练候选
- 可以排除 PII、高风险确定性结论、低质量或缺少 provenance 的样本
- 可以导出到现有 SFT samples DB
- 可以通过现有 adapter store 产生 candidate adapter，或明确说明真实 LoRA 训练 skip 原因
- 可以生成 base/local eval report
- eval gate 能给出 pass/review/fail
- Studio 能看到 Phase4 sources/chunks/candidates/adapter/delta/gate 状态
