# PFE Phase 2 release candidate notes

更新时间：2026-06-15

这份 release note 面向 Phase 2 release candidate。它描述当前 checkout 已经验证过的能力、用户可期待的行为，以及还没有承诺的边界。

## 主要变化

- 完成 `collect -> curate -> train -> eval -> promote -> serve` 的本地闭环验证。
- 新增 `smoke-release-strict`，把 beta smoke、真实浏览器 UI smoke、real-local tiny model happy path 串成同一条 release gate。
- 新增可复现 tiny Hugging Face GPT-2 兼容模型生成工具：`tools/prepare_tiny_hf_model.py`。
- 新增 30 分钟 release soak：持续覆盖 health/status、dashboard API、queue/daemon/runner、chat/feedback round trip。
- 新增 `benchmark-release`：记录并执行 first-run、browser UI、real-local happy path、短 soak 的性能和内存预算。
- Dashboard 移除外部 Chart.js / Google Fonts，使用内联 `OfflineChart`，支持本地/offline-first 验证。
- Dashboard metrics 增加短 TTL cache 和 stale fallback，避免慢刷新阻塞 live dashboard。
- Worker daemon 启动时显式继承当前 `PFE_HOME`，避免临时 workspace 下的 daemon 落到用户级 `~/.pfe`。
- Release soak server 日志改为落盘，避免长时间 uvicorn access log 写满 PIPE 后阻塞请求。
- 新增 GitHub Actions workflow：PR fast gate 与 manual/nightly strict release gate 分层。
- PFE Studio 已成为默认浏览器入口，集中承载模型选择、API 地址交接和版本管理。

## 当前验证证据

本机已通过：

- `PFE_REAL_LOCAL_MODEL=$HOME/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict`
- `.venv/bin/python tools/release_soak_smoke.py --duration-seconds 1800 --interval-seconds 2 --report-path /tmp/pfe-release-soak-30m-report.json`
- `make benchmark-release`
- `.venv/bin/python -m pytest tests/test_dashboard.py -q`
- `.venv/bin/python tools/dashboard_console_live_smoke.py`
- `.venv/bin/python tools/browser_ui_live_smoke.py --strict --browser-timeout-ms 45000`
- `git diff --check`

远端已通过：

- workflow: `PFE release gates`
- run: `https://github.com/leoli001031-blip/Personal-Fine-Tuning-Engine/actions/runs/27518991700`
- branch: `main`
- commit: `0c08d2791edce2ac6c48ce1f432a1eb6716fca8d`
- conclusion: `success`

30 分钟 soak 结果：

- duration: `1806.17s`
- iterations: `847`
- probes: `12426`
- chat_turns: `282`
- latency_ms: `avg=8.67`, `p95=89.53`, `max=3033.63`
- daemon: `healthy/fresh/valid`
- report: `/private/tmp/pfe-release-soak-30m-report.json`

Performance budget 结果：

- total: `36.55s`
- threshold_violations: `[]`
- report: `/private/var/folders/3s/4nftc3d52xd3j1yqbm1gf78m0000gn/T/pfe-release-perf-report.json`

完整证据见 `docs/reference/release-readiness-evidence.md`。

## 用户可期待的能力

- 可以初始化本地 workspace，并通过 `pfe doctor` 查看 local model、trainer dependency、adapter、signal chain、queue/daemon readiness。
- 可以打开 `/` 进入 PFE Studio，完成模型选择、API 地址复制和版本管理。
- 可以生成样本、采集反馈信号、触发 deferred queue 训练、评估 adapter、promote adapter，并通过 live server 提供 OpenAI-compatible chat endpoint。
- 可以通过 `/dashboard` 查看 adapter、signal、training、queue 和 daemon 状态。
- 可以用 strict release gate 验证浏览器 UI、real-local tiny model 训练路径和本地服务闭环。

## 已知限制

这不是大模型生产训练包，也不是云端多租户服务。当前 release candidate 仍以 local-first、single-user、可验证闭环为核心。详细限制见 `docs/reference/known-limitations.md`。

## 升级注意

- 推荐使用 Python 3.11 和 `.venv/bin/python`。
- real-local strict gate 需要 `torch`、`transformers`、`peft`、`accelerate`、`safetensors`。
- 浏览器 strict gate 需要 `.[e2e]` 和 Playwright Chromium。
- 旧环境中如果 `.venv/bin/pfe` 指向过期路径，使用 `tools/bootstrap_py311_env.sh` 重建。

## Release readiness 状态

release evidence 已覆盖 Phase 2 strict gate、30 分钟 soak、performance budget、dashboard offline-first 和远端 GitHub Actions strict release gate。当前 release gate 阻塞项为 0；后续只剩非阻塞维护项，例如 GitHub Actions Node.js 20 deprecation annotation 的升级窗口。
