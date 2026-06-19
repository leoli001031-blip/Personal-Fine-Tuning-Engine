# PFE Phase6 Candidate Adapter Trial Runbook

Phase6 把真实训练放回 PFE 的产品理念里：训练不是入口，训练是一次候选适配器试验的执行步骤。

```text
任务使用 -> signal inbox -> 有价值信号筛选 -> candidate samples
-> candidate adapter trial -> real training preflight/train
-> base/local eval -> promote/archive/collect more signal
```

## 1. 产品模式

Phase6 不把 `Qwen3.6` 做成前台主按钮。

前台和 API 语义围绕：

- create trial
- collect signal
- generate candidate samples
- train candidate adapter
- compare base/local
- promote/archive

模型、LoRA rank、batch size、backend 是高级配置和 runbook 参数。用户看到的核心是：本轮候选改进是否有证据通过 eval gate。

## 2. 模型选择

第一目标模型：

```text
mlx-community/Qwen3.6-27B-4bit
```

上游基座：

```text
Qwen/Qwen3.6-27B
```

选择理由：

- Qwen3.6-27B 是 dense 模型，比 MoE 路径更容易在第一轮 trial 中排障。
- MLX 4bit 权重适合 Apple Silicon。
- 当前机器是 Apple Silicon / 128GB unified memory / 约 1TB free disk，适合做小批量 LoRA trial。

## 3. Preflight

Phase6 preflight 检查：

- Apple Silicon
- unified memory
- disk free
- `mlx` / `mlx_lm`
- local model path 或是否显式允许 remote download

安装 MLX runtime：

```bash
.venv/bin/python -m pip install 'mlx>=0.5' 'mlx-lm>=0.5'
```

默认 smoke 不下载 Qwen3.6 权重：

```bash
.venv/bin/python tools/phase6_candidate_adapter_trial_smoke.py --timeout 180
```

当前本地实测：

```text
preflight.dependencies.mlx=true
preflight.dependencies.mlx_lm=true
preflight.system.machine=arm64
preflight.system.memory_gb=128.0
preflight.disk.free_gb≈1004
preflight.model_id=mlx-community/Qwen3.6-27B-4bit
preflight.model_status=download_required
preflight.status=needs_model_download
preflight.ready_for_real_training=false
```

这表示 runtime 已经准备好，但权重下载和真实训练仍需显式打开。

## 4. Trial Smoke

默认 smoke 会：

- 复用 Phase5 Common Paper 真实资料导入
- 生成 Phase4/Phase5 candidates 和 holdout
- 从 eligible signal + corpus samples 生成 Phase6 candidate samples
- 运行 Qwen3.6/MLX preflight
- 生成 trial manifest
- 生成 eval report
- 生成 decision

命令：

```bash
.venv/bin/python tools/phase6_candidate_adapter_trial_smoke.py --timeout 180
```

当前本地实测：

```text
phase5.source_count=10
phase5.ingested_count=10
phase5.candidate_count=60
phase5.eligible_count=57
candidate_samples.count=51
candidate_samples.requires=source, chunk, provenance, signal_id
holdout.count=16
holdout.not_for_training=true
trial_status=created
training_result.training.real_training=not_started
eval_gate.status=blocked
eval_gate.promotion_allowed=false
decision.action=archive
```

`archive` 是正确结果：当前没有真实 Qwen3.6 训练和真实 base/local 生成调用，所以不能 promote。

## 5. Real Qwen3.6 Trial

显式允许下载和真实训练：

```bash
.venv/bin/python tools/phase6_candidate_adapter_trial_smoke.py \
  --allow-remote-download \
  --run-real-training \
  --strict-real \
  --timeout 7200
```

如果已经有本地 MLX 模型目录：

```bash
.venv/bin/python tools/phase6_candidate_adapter_trial_smoke.py \
  --model-path /absolute/path/to/Qwen3.6-27B-4bit \
  --require-local-model \
  --run-real-training \
  --strict-real \
  --timeout 7200
```

初始训练参数：

```text
backend=mlx
train_type=sft
seq_length=2048
batch_size=1
grad_accumulation=8
lora_rank=8
epochs=1
```

真实训练必须产出：

- adapter artifact
- adapter manifest
- train_loss 或等价训练指标
- num_examples
- model/backend/runtime metadata
- artifact path

未产出这些证据时，不允许把 trial 描述成真实训练完成。

## 6. Eval Gate

Phase6 eval gate 关注：

- citation hit rate
- structure adherence：摘要 / 风险提示 / 引用依据 / 人工确认
- unsupported assertions
- legal conclusion avoidance
- insufficient evidence handling
- human confirmation quality

Promotion 条件：

```text
real training completed
real base/local model calls completed
candidate samples preserve source/chunk/provenance/signal_id
holdout not used for training
local beats base on eval gate
```

否则 decision 必须是：

- `archive`
- `collect_more_signal`
- `fix_preflight`

不能 promote。

## 7. API

最小 API surface：

```text
GET  /pfe/phase6
GET  /pfe/phase6/preflight
POST /pfe/phase6/preflight
GET  /pfe/phase6/trial
POST /pfe/phase6/trial
POST /pfe/phase6/trial/eval
```

Demo trial，不联网：

```bash
curl -X POST http://127.0.0.1:8921/pfe/phase6/trial \
  -H 'content-type: application/json' \
  -d '{"demo": true, "require_local_model": true, "model_path": "/missing/qwen36"}'
```

## 8. Tests

Targeted tests：

```bash
.venv/bin/python -m pytest tests/test_phase6_candidate_adapter_trial.py
.venv/bin/python -m pytest tests/test_server_http.py -k "phase3 or phase4 or phase6"
```

Recommended validation：

```bash
.venv/bin/python -m py_compile \
  pfe-core/pfe_core/phase6_candidate_adapter_trial.py \
  tools/phase6_candidate_adapter_trial_smoke.py \
  pfe-server/pfe_server/app.py \
  tests/test_phase6_candidate_adapter_trial.py \
  tests/test_server_http.py

.venv/bin/python -m pytest \
  tests/test_phase6_candidate_adapter_trial.py \
  tests/test_phase5_real_domain_loop.py \
  tests/test_phase4_real_train_smoke.py \
  tests/test_phase4_real_corpus.py

.venv/bin/python tools/phase6_candidate_adapter_trial_smoke.py --timeout 180
git diff --check
make test-unit test-surface test-e2e-mock smoke-beta
```

## 9. Known Limits

- 默认 smoke 不下载 15GB 级 Qwen3.6 权重。
- 默认 eval 仍是 trial evidence gate，不声称真实模型效果提升。
- 只有 `--run-real-training` 并完成真实 base/local model calls 后，才允许 promote。
- 本轮不修改 `videos/`，不提交模型权重、adapter artifacts 或 cache。
