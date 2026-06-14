# Known limitations

更新时间：2026-06-15

这份文档列出 PFE Phase 2 release candidate 的已知限制。它们不是隐藏缺陷，而是当前 release 范围之外或需要额外环境验证的边界。

## 1. Runtime 范围

- 当前 release evidence 证明的是 local-first、single-user、本地 workspace 闭环。
- 当前没有承诺多租户隔离、远程队列、多用户权限或云端高可用部署。
- Live server 主要用于 loopback、本机 dashboard 和 OpenAI-compatible local endpoint。

## 2. Real-local 训练范围

- strict gate 使用 tiny Hugging Face GPT-2 兼容模型验证 PEFT happy path。
- 这能证明 real-local dependency、training dispatch、adapter manifest、eval、promote 和 serve preview 可达。
- 它不代表大型模型、长数据集或 GPU 长训性能已经被证明。
- 当前 release evidence 没有覆盖 full-size production model 的训练质量。

## 3. CUDA / Metal / MLX 覆盖

- 本机 evidence 来自当前 macOS 环境，runtime device 显示为 `mps`。
- CUDA/Linux 真实训练仍应使用 `docs/guides/cuda-real-training-validation.md` 单独验证。
- `mlx` / `mlx-lm` 是可选 extra，当前 strict release gate 不依赖它们。

## 4. Optional training packages

- `trl` 和 `datasets` 当前未安装。
- Phase 2 strict PEFT happy path 不依赖它们。
- DPO 或更完整训练工作流需要单独安装并验证相关 extra。

## 5. Export / llama.cpp

- `pfe doctor` 可能报告 llama.cpp export tool missing。
- 当前 release candidate 的核心闭环不要求 GGUF export 成功。
- 如果发布目标包含 llama.cpp export，需要单独配置 `PFE_LLAMA_CPP_EXPORT_TOOL` 或工具路径，并增加 release evidence。

## 6. Dashboard

- Dashboard 已经 offline-first，不再拉外部 Chart.js 或 Google Fonts。
- 内联 `OfflineChart` 是 release fallback，覆盖当前页面使用的 line、bar、doughnut 图表。
- 它不是完整 Chart.js API 替代品。
- Dashboard metrics 使用短 TTL cache 和 stale fallback；返回 payload 会包含 `dashboard_cache` 元数据。

## 7. Soak 与 CI

- 本机已通过 30 分钟 long soak。
- 正式发布前可按需要再跑 60 分钟窗口。
- GitHub Actions workflow 已定义，但远端 manual/nightly 结果需要在推送后验证。

## 8. Data 与隐私

- 当前流程默认 local-first，smoke/soak 使用临时 workspace。
- 用户真实数据、模型权重和本地 cache 不应直接提交。
- Release evidence 中的 `/tmp` report 路径是本机验证产物，不是仓库内长期 artifact。

## 9. Performance budget

`make benchmark-release` 默认执行当前 release budget：

- `first_run_full <= 20.0s / 600.0 MB`
- `browser_ui_strict <= 20.0s / 1500.0 MB`
- `real_local_happy <= 30.0s / 1200.0 MB`
- `release_soak_short <= 35.0s / 1000.0 MB`

这些预算是本地 release candidate 预算，不是所有硬件上的 SLA。新机器若需要采样原始 baseline，可以运行 `tools/release_perf_benchmark.py --no-thresholds`，但不能把它当作 release 通过证据。
