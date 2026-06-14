# Release readiness evidence

更新时间：2026-06-15 04:20 CST

这份记录只描述当前 checkout 的验证证据，不把未完成项包装成已完成。

## 当前结论

- `smoke-beta`：通过。
- `browser_ui_live_smoke.py --strict`：通过，Playwright/Chromium strict blocker 已解除。
- `real_local_happy_path_smoke.py --strict`：通过，已用本地 tiny Hugging Face GPT-2 兼容模型跑通 PEFT real-local happy path。
- `make smoke-release-strict`：通过。
- `release_soak_smoke.py --duration-seconds 1800 --interval-seconds 2`：通过，已跑 30 分钟长稳态 live soak，覆盖 server/dashboard/queue/daemon/chat/feedback。
- `make benchmark-release`：通过，已记录并执行 first-run、strict browser UI、real-local happy path、短 soak 的耗时和进程树峰值 RSS release budget。
- `make release-local-evidence`：已作为一键本地 release evidence 入口落地，顺序执行 mock e2e、strict smoke、benchmark、audit report 和 bundle manifest。
- Dashboard：已移除外部 Chart.js / Google Fonts 依赖，改为内联 `OfflineChart` canvas fallback；dashboard live smoke 会拒绝外部资源 URL。
- CI workflow contract：本地测试已验证 release workflow 引用的 Makefile 目标存在，并保留 strict release gate 与 JSON artifact 上传步骤。
- Release evidence audit：`make audit-release-evidence` 已可复核本地证据、workflow 契约、dashboard offline-first、根目录卫生和远端 CI 缺口；`make audit-release-evidence-report` 可写出 JSON 证据报告。
- Release evidence bundle：`bundle-release-evidence` 已可校验 performance report、evidence audit report 和远端 Actions evidence report，并写出带 sha256 / size / status 的 bundle manifest。
- Remote Actions evidence：`record-remote-release-evidence` 已可在 workflow 推送并跑通后抓取最新成功 GitHub Actions run，写出 run URL / status / conclusion / branch / commit JSON。
- Remote evidence Markdown：`render-remote-release-evidence` 已可把远端 run JSON 和 bundle manifest 渲染成可写入本文件的 Markdown 摘要。

因此当前状态是：Phase 2 功能闭环、真实浏览器 UI strict smoke、real-local readiness、real-local happy path 已在本机通过同一条 strict release gate 验证；queue/daemon/server/dashboard 也已有 30 分钟长稳态 soak、性能/内存 budget、dashboard offline-first 和发布材料证据。完整 release-ready 仍需要补远端 CI 执行证据。

## 最新本地复核

2026-06-15 03:29 CST 复核：

- `.venv/bin/python -m pytest tests/test_release_workflow_contract.py tests/test_dashboard.py -q`：`21 passed in 0.49s`。
- `PFE_REAL_LOCAL_MODEL=/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict`：通过。
- `ruby -e 'require "yaml"; ...'`：workflow YAML 可解析，包含 `fast-gate,release-gate`。
- `git diff --check`：通过。
- `gh api repos/leoli001031-blip/Personal-Fine-Tuning-Engine/actions/workflows`：`workflow_count=0`，远端 CI 证据仍缺。

2026-06-15 03:32 CST 复核：

- `.venv/bin/python -m pytest tests/test_release_evidence_audit.py tests/test_release_workflow_contract.py -q`：`4 passed in 0.02s`。
- `make audit-release-evidence`：通过，本地 release evidence 审计无 blocker，远端 run URL 缺失以 warning 形式保留。
- `.venv/bin/python tools/release_evidence_audit.py --require-remote --check-remote --skip-process-check`：按预期返回 `exit_code=2`，blocker 为 `remote_ci_run_evidence` 和 `remote_workflow_registered`。

2026-06-15 03:36 CST 复核：

- `.venv/bin/python -m pytest tests/test_release_evidence_audit.py -q`：`3 passed in 0.03s`。
- `make audit-release-evidence-report AUDIT_REPORT=/tmp/pfe-release-evidence-audit-test.json`：通过，写出机器可读 JSON 报告。
- `.venv/bin/python -m json.tool /tmp/pfe-release-evidence-audit-test.json`：通过，报告 `status=passed`、`summary={blocker: 0, ok: 10, total: 11, warn: 1}`。

2026-06-15 03:38 CST 复核：

- `.venv/bin/python -m pytest tests/test_release_evidence_audit.py tests/test_release_workflow_contract.py -q`：`5 passed in 0.02s`。
- `make audit-release-evidence`：通过，workflow contract 已覆盖 `audit-release-evidence-report` 和 `actions/upload-artifact@v4`。
- `ruby -e 'require "yaml"; ...'`：workflow YAML 可解析，包含 `fast-gate,release-gate`。

2026-06-15 03:43 CST 复核：

- `.venv/bin/python -m pytest tests/test_github_actions_release_evidence.py tests/test_release_evidence_audit.py tests/test_release_workflow_contract.py -q`：`9 passed in 0.02s`。
- `record-remote-release-evidence` 入口已加入 Makefile，并被 `release_evidence_audit.py` 纳入必备证据链检查。
- `make record-remote-release-evidence REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence-current.json`：当前按预期返回 `exit_code=2`，JSON 为 `status=missing`，blocker 为 `matching GitHub Actions run not found`。
- 当前远端仍为 `workflow_count=0`，因此 `record-remote-release-evidence` 只能在 workflow 提交/推送并跑通后作为最终通过证据。

2026-06-15 03:48 CST 复核：

- `make benchmark-release PERF_REPORT=/tmp/pfe-release-perf-report-bundle.json`：通过，`total=36.744s`，四个任务均在预算内。
- `make audit-release-evidence-report AUDIT_REPORT=/tmp/pfe-release-evidence-audit-bundle.json`：通过，远端 run URL 缺失以 warning 保留。
- `make bundle-release-evidence PERF_REPORT=/tmp/pfe-release-perf-report-bundle.json AUDIT_REPORT=/tmp/pfe-release-evidence-audit-bundle.json BUNDLE_REPORT=/tmp/pfe-release-evidence-bundle.json`：通过，生成 bundle manifest。
- bundle manifest：`status=passed`、`summary={blockers: 0, present: 2, total: 3, warnings: 1}`；`remote_actions` 当前缺失，作为 warning 保留。

2026-06-15 03:51 CST 复核：

- `render-remote-release-evidence` 入口已加入 Makefile，并被 `release_evidence_audit.py` 纳入必备证据链检查。
- `tools/render_remote_release_evidence.py` 可把 `pfe-github-actions-release-evidence.json` 和 `pfe-release-evidence-bundle.json` 渲染为 Markdown 摘要；当前缺远端成功 run 时，`--require-success` 会阻塞。

2026-06-15 03:58 CST 复核：

- `.venv/bin/python -m pytest tests/test_release_evidence_audit.py tests/test_github_actions_release_evidence.py tests/test_render_remote_release_evidence.py tests/test_release_evidence_bundle.py tests/test_release_workflow_contract.py -q`：`14 passed in 0.04s`。
- `make record-remote-release-evidence REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence.json`：按预期返回 `exit_code=2`，写出 recorder JSON；当前 `status=missing`、`release_ready=false`、blocker 为 `matching GitHub Actions run not found`。
- `make benchmark-release PERF_REPORT=/tmp/pfe-release-perf-report.json`：通过，`total=36.437s`；`first_run_full=7.046s / 264.08 MB`，`browser_ui_strict=5.349s / 936.06 MB`，`real_local_happy=5.874s / 718.03 MB`，`release_soak_short=18.168s / 617.77 MB`。
- `make audit-release-evidence-report AUDIT_REPORT=/tmp/pfe-release-evidence-audit.json`：通过，报告 `status=passed`、`summary={blocker: 0, ok: 10, total: 12, warn: 2}`；warning 为 `remote_ci_run_evidence` 和 `remote_evidence_report`。
- `make bundle-release-evidence PERF_REPORT=/tmp/pfe-release-perf-report.json AUDIT_REPORT=/tmp/pfe-release-evidence-audit.json REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence.json BUNDLE_REPORT=/tmp/pfe-release-evidence-bundle.json`：通过，bundle manifest `status=passed`、`summary={blockers: 0, present: 3, total: 3, warnings: 1}`；`remote_actions` 当前为 `status=missing` warning。
- `make render-remote-release-evidence REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence.json BUNDLE_REPORT=/tmp/pfe-release-evidence-bundle.json REMOTE_EVIDENCE_MARKDOWN=/tmp/pfe-remote-release-evidence.md`：按预期返回 `exit_code=2`，并写出 `status=missing` / `release_ready=no` 的远端证据 Markdown 摘要。

2026-06-15 04:02 CST 复核：

- `make test-unit`：`863 passed, 16 skipped, 30 deselected in 31.21s`。
- `make test-surface`：`144 passed in 20.71s`。
- `PFE_REAL_LOCAL_MODEL=/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict`：通过；其中 `smoke-beta` 覆盖 first-run、auto-train queue、real-local readiness、server live、dashboard console，随后 `browser_ui_live_smoke.py --strict` 与 `real_local_happy_path_smoke.py --strict` 均通过。
- strict run 尾部确认：`BROWSER UI LIVE SMOKE PASSED`，`REAL-LOCAL HAPPY PATH SMOKE PASSED`，`execution: kind=real_peft | path=real_import`。
- `make audit-release-evidence-report AUDIT_REPORT=/tmp/pfe-release-evidence-audit.json`：通过，gate 后无 release smoke/server/daemon 进程残留；本地 blocker 为 0，远端 run URL 与远端 evidence JSON 仍为 warning。
- `make bundle-release-evidence PERF_REPORT=/tmp/pfe-release-perf-report.json AUDIT_REPORT=/tmp/pfe-release-evidence-audit.json REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence.json BUNDLE_REPORT=/tmp/pfe-release-evidence-bundle.json`：通过，`remote_actions: status=missing release_ready=False` 仍为唯一 warning。

2026-06-15 04:08 CST 复核：

- `make test-e2e-mock`：`12 passed, 22 deselected in 30.23s`。
- `.venv/bin/python -m pytest tests/test_release_workflow_contract.py -q`：`2 passed in 0.01s`。
- `ruby -e 'require "yaml"; ...'`：workflow YAML 可解析，包含 `fast-gate,release-gate`。
- `make audit-release-evidence`：通过，本地证据、workflow 契约、dashboard offline-first、根目录卫生和进程残留检查均无 blocker；远端 run URL 与远端 evidence JSON 仍为 warning。

2026-06-15 04:11 CST 复核：

- `.github/workflows/pfe-release-gates.yml` 已将 `make test-e2e-mock` 纳入 PR fast gate，并在 release/nightly gate 的 `smoke-release-strict` 前执行。
- `make test-e2e-mock`：`12 passed, 22 deselected in 29.80s`。
- `.venv/bin/python -m pytest tests/test_release_workflow_contract.py tests/test_release_evidence_audit.py -q`：`6 passed in 0.02s`。
- `make audit-release-evidence`：通过，workflow target contract 已包含 `test-e2e-mock`。

2026-06-15 04:16 CST 复核：

- `make release-local-evidence` 已加入 Makefile，作为本地 release evidence 一键入口。
- `.venv/bin/python -m pytest tests/test_release_workflow_contract.py tests/test_release_evidence_audit.py -q`：通过；contract 覆盖 `release-local-evidence` 的执行顺序和 audit required target。
- `make audit-release-evidence`：通过，Make target contract 已包含 `release-local-evidence`。

2026-06-15 04:20 CST 复核：

- `PFE_REAL_LOCAL_MODEL=/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local make release-local-evidence`：通过，已实际串起 mock e2e、strict smoke、benchmark、audit report 和 bundle manifest。
- `test-e2e-mock` 段：`12 passed, 22 deselected in 29.98s`。
- `smoke-release-strict` 段：通过，`BROWSER UI LIVE SMOKE PASSED`，`REAL-LOCAL HAPPY PATH SMOKE PASSED`，`execution: kind=real_peft | path=real_import`。
- `benchmark-release` 段：通过，`total=37.439s`；`first_run_full=7.908s / 278.27 MB`，`browser_ui_strict=5.478s / 953.58 MB`，`real_local_happy=5.816s / 717.77 MB`，`release_soak_short=18.237s / 617.66 MB`。
- `audit-release-evidence-report` 段：通过，本地 blocker 为 0；远端 run URL 与远端 evidence JSON 仍为 warning。
- `bundle-release-evidence` 段：通过，唯一 warning 为 `remote_actions: status=missing release_ready=False`。

本次 strict run 的关键尾部：

```text
DASHBOARD CONSOLE LIVE SMOKE PASSED
workspace:     dashboard_console_live
version:       20260615-001
base_url:      http://127.0.0.1:49795

BROWSER UI LIVE SMOKE PASSED
workspace: browser_ui_live
version:   20260615-001
base_url:  http://127.0.0.1:49810

REAL-LOCAL HAPPY PATH SMOKE PASSED
workspace:  real_local_happy
version:    20260615-001
base_model: /Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local
execution:  kind=real_peft | path=real_import
```

## 环境证据

- Python：`.venv/bin/python`，版本 `3.11.15`。
- pip：通过 `.venv/bin/python -m ensurepip --upgrade` 补齐。
- e2e 依赖：通过 `.venv/bin/python -m pip install -e '.[e2e]'` 安装。
- Playwright：`1.60.0`。
- Chromium cache：
  - `/Users/lichenhao/Library/Caches/ms-playwright/chromium-1223`
  - `/Users/lichenhao/Library/Caches/ms-playwright/chromium_headless_shell-1223`
- PEFT real-local runtime：已安装 `torch`、`transformers`、`peft`、`accelerate`、`safetensors`。
- 当前未安装：`trl`、`datasets`。本轮 PEFT happy path 不依赖它们。
- 可复现 tiny model：
  - 生成命令：`.venv/bin/python tools/prepare_tiny_hf_model.py`
  - 模型目录：`/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local`
  - 配置：`/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local/config.json`
  - 权重：`/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local/model.safetensors`
  - manifest：`/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local/pfe_tiny_model_manifest.json`

## 通过命令

```bash
.venv/bin/python tools/browser_ui_live_smoke.py --strict --browser-timeout-ms 45000
```

结果：

```text
BROWSER UI LIVE SMOKE PASSED
workspace: browser_ui_live
version:   20260615-001
base_url:  http://127.0.0.1:56585
```

```bash
PFE_REAL_LOCAL_MODEL=/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local \
  .venv/bin/python tools/real_local_happy_path_smoke.py --strict --timeout 120
```

结果：

```text
REAL-LOCAL HAPPY PATH SMOKE PASSED
workspace:  real_local_happy
version:    20260615-001
base_model: /Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local
execution:  kind=real_peft | path=real_import
```

```bash
PFE_REAL_LOCAL_MODEL=/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict
```

结果：通过，覆盖：

- `smoke-first-run`
- `smoke-auto-train-queue`
- `smoke-real-local-readiness`
- `smoke-server-live`
- `smoke-dashboard-console-live`
- `tools/browser_ui_live_smoke.py --strict`
- `tools/real_local_happy_path_smoke.py --strict`

关键尾部证据：

```text
BROWSER UI LIVE SMOKE PASSED
workspace: browser_ui_live
version:   20260615-001
base_url:  http://127.0.0.1:58226

REAL-LOCAL HAPPY PATH SMOKE PASSED
workspace:  real_local_happy
version:    20260615-001
base_model: /Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local
execution:  kind=real_peft | path=real_import
```

## 本轮修复/新增

- `tools/real_local_readiness_smoke.py`：beta readiness 保留 `execution_intent=real_local`，并允许缺少训练依赖时走 `requested_backend=peft | execution_backend=mock_local | execution_mode=fallback`；真实 PEFT 执行仍由 strict gate 覆盖。
- `tools/browser_ui_live_smoke.py`：等待 dashboard 初始化和 refresh 的 `/pfe/dashboard/metrics` 200 响应，避免导航取消未完成 fetch 被误判为浏览器错误。
- `tools/browser_ui_live_smoke.py`：聊天页不再依赖会被 status refresh 覆盖的 footer 文案，改为等待真实 `.bubble.assistant .feedback-btn.accept`。
- `pyproject.toml`：`training` extra 补入 `accelerate>=0.23`，与 real-local PEFT runtime 的实际导入需求一致。
- `tools/prepare_tiny_hf_model.py`：新增可复现 tiny HF model 生成工具，避免 release gate 依赖外部下载。
- `pfe-core/pfe_core/pipeline.py`：daemon 子进程启动时显式传入当前 `PFE_HOME`，避免 server 在临时 workspace 中启动 daemon 后，子进程落到用户级 `~/.pfe`。
- `tools/release_soak_smoke.py`：新增 bounded release soak，启动隔离 live server 和真实 worker daemon，持续轮询 health/status/dashboard/queue/daemon/runner，并穿插 chat/feedback。
- `tools/release_perf_benchmark.py`：新增 release benchmark，按任务记录耗时和进程树峰值 RSS；安装 `psutil` 时使用真实 RSS 采样，并默认执行 release budget。
- `pfe-server/pfe_server/static/dashboard.html`：移除 CDN Chart.js 和 Google Fonts，新增本地 `OfflineChart`，dashboard HTML 不再需要外部网络资源。
- `tools/dashboard_console_live_smoke.py`：dashboard live smoke 新增离线资源保护，若 HTML 再出现外部 `http://` 或 `https://` URL 会失败。
- `pfe-server/pfe_server/dashboard_api.py`：dashboard metrics 新增短 TTL cache 和 stale fallback，避免慢刷新拖住 live dashboard 响应。
- `pfe-server/pfe_server/dashboard/metrics.py`：dashboard system health 不再调用完整 `PipelineService.status()`，改为轻量读取 queue state，降低 live polling 成本。
- `tools/release_soak_smoke.py`：server stdout/stderr 改为落盘日志，避免长 soak 中 uvicorn access log 填满 PIPE 导致 server 写日志阻塞；失败时也会写结构化 JSON 报告。
- `pyproject.toml`：`e2e` extra 补入 `psutil`，让 release benchmark 在 CI 中也能采集峰值 RSS。
- `.github/workflows/pfe-release-gates.yml`：新增 GitHub Actions fast beta gate 和 strict release gate；PR 跑 unit/surface/mock e2e/beta fast gate，manual/nightly 跑 mock e2e、strict browser/model、benchmark budget、release evidence audit，并上传 `pfe-release-evidence` artifact。
- `tests/test_release_workflow_contract.py`：新增本地 CI contract test，验证 workflow 中的 `make` 目标存在，并保留 e2e install、Playwright install、tiny model、`PFE_REAL_LOCAL_MODEL`、`test-e2e-mock`、`smoke-release-strict`、`benchmark-release`、`audit-release-evidence-report` 和 artifact 上传配置。
- `tools/release_evidence_audit.py`：新增本地 release evidence 审计，复核 release 文件、Makefile 目标、workflow strict gate、dashboard offline-first、release evidence 记录、根目录卫生和远端 CI run evidence。
- `tools/release_evidence_audit.py`：新增 `--report-path`，可写出机器可读 JSON 报告，方便本地发布证据包或 CI artifact 保存。
- `tests/test_release_evidence_audit.py`：新增审计工具测试，确保默认本地审计通过，`--require-remote` 在缺少 GitHub Actions run URL 时会阻塞，并验证 JSON 报告结构。
- `tools/release_evidence_bundle.py`：新增 release evidence bundle manifest 工具，校验并汇总 performance/audit/remote evidence JSON，记录 sha256、size 和状态。
- `tests/test_release_evidence_bundle.py`：新增 bundle manifest 测试，覆盖本地报告通过、远端报告可选 warning、远端必需时阻塞。
- `tools/github_actions_release_evidence.py`：新增远端 GitHub Actions run 证据采集工具，使用 `gh api` 查询最新匹配 workflow run，并输出机器可读 JSON。
- `tests/test_github_actions_release_evidence.py`：新增远端 run 选择和 success/blocked 语义的纯单元测试。
- `tools/render_remote_release_evidence.py`：新增远端 run Markdown 摘要渲染工具，把 run URL、status、conclusion、branch、commit 和 artifact bundle 摘要转成可写入本文件的片段。
- `tests/test_render_remote_release_evidence.py`：新增远端 run Markdown 渲染测试，覆盖成功 run 与 missing run 两种状态。
- `docs/reference/release-candidate-checklist.md`：新增 release candidate 清单，明确本地必过 gate、performance budget、长 soak、CI 分层和发布材料。
- `docs/reference/release-notes-phase2-rc.md`：新增 Phase 2 release candidate notes。
- `docs/guides/install-upgrade.md`：新增本地安装、升级和 release gate 准备指南。
- `docs/reference/known-limitations.md`：新增 release candidate 已知限制。
- `docs/reference/user-acceptance-checklist.md`：新增真实用户验收 checklist。

## Soak 证据

```bash
.venv/bin/python tools/release_soak_smoke.py --duration-seconds 1800 --interval-seconds 2 --report-path /tmp/pfe-release-soak-30m-report.json
```

结果：通过，覆盖 30 分钟长稳态。

```text
RELEASE SOAK SMOKE PASSED
workspace:  release_soak
version:    20260615-001
base_url:   http://127.0.0.1:63092
duration:   1806.17s
iterations: 847
probes:     12426
chat_turns: 282
latency_ms: {'avg': 8.67, 'max': 3033.63, 'p95': 89.53}
daemon:     {'health_state': 'healthy', 'heartbeat_state': 'fresh', 'lease_state': 'valid', 'lock_state': 'active', 'pid': 21428}
report:     /private/tmp/pfe-release-soak-30m-report.json
```

清理验证：

```text
no pfe_core.worker_daemon / pfe_cli.main serve process remained
.pfe absent
uv.lock absent
videos dir present
```

## Performance / memory 证据

```bash
make benchmark-release
```

结果：通过。

```text
RELEASE PERF BENCHMARK PASSED
tasks:  first_run_full, browser_ui_strict, real_local_happy, release_soak_short
total:  36.55s
report: /private/var/folders/3s/4nftc3d52xd3j1yqbm1gf78m0000gn/T/pfe-release-perf-report.json
- first_run_full: elapsed=7.16s peak_rss_mb=282.69
- browser_ui_strict: elapsed=5.342s peak_rss_mb=994.19
- real_local_happy: elapsed=5.754s peak_rss_mb=717.44
- release_soak_short: elapsed=18.294s peak_rss_mb=614.91
```

报告元数据：

```text
status: passed
memory_sampler: psutil
model: /Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local
thresholds: enforced
threshold_violations: []
budget:
  first_run_full <= 30.0s / 800.0 MB
  browser_ui_strict <= 30.0s / 1600.0 MB
  real_local_happy <= 45.0s / 1800.0 MB
  release_soak_short <= 45.0s / 1400.0 MB
```

## Dashboard offline-first 证据

- `pfe-server/pfe_server/static/dashboard.html` 不包含 `http://` 或 `https://` 外部资源 URL。
- `tools/dashboard_console_live_smoke.py` 会验证 `window.Chart = OfflineChart`，并在 dashboard HTML 出现外部 URL 时失败。
- 应用内 Browser 打开临时 live dashboard 后，console error/warning 为 0，页面外部链接列表为空。
- 4 个 dashboard canvas 均稳定渲染在 `300px` 高的 `.chart-container` 内。

## CI workflow contract 证据

```bash
.venv/bin/python -m pytest tests/test_release_workflow_contract.py -q
```

结果：

```text
2 passed in 0.01s
```

覆盖范围：

- `.github/workflows/pfe-release-gates.yml` 引用的 `make` 目标都存在于 `Makefile`。
- workflow 保留 `release-gate`，并限制为非 PR 事件执行。
- release gate 保留 e2e 安装、Playwright Chromium 安装、tiny model 准备、`PFE_REAL_LOCAL_MODEL`、`test-e2e-mock`、`smoke-release-strict`、`benchmark-release`、`audit-release-evidence-report` 和 `actions/upload-artifact@v4`。

## Release evidence audit 证据

```bash
make audit-release-evidence
```

结果：

```text
RELEASE EVIDENCE AUDIT PASSED
OK      required_files: required release files present
OK      make_targets: release Makefile targets present
OK      workflow_targets: workflow make targets valid: ['audit-release-evidence-report', 'benchmark-release', 'bundle-release-evidence', 'smoke-beta', 'smoke-release-strict', 'test-e2e-mock', 'test-surface', 'test-unit']
OK      workflow_strict_gate: strict workflow gate retained
OK      dashboard_offline: dashboard is offline-first with OfflineChart
OK      release_evidence_doc: release evidence records local gates and remote gap
OK      remote_state_recorded: release evidence records remote CI state
OK      root_pfe_absent: root .pfe absent
OK      uv_lock_absent: root uv.lock absent
OK      process_residue: no release smoke/server/daemon process residue
WARN    remote_ci_run_evidence: GitHub Actions run URL missing from release evidence
```

远端 run 证据采集：

```bash
make record-remote-release-evidence REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence.json
make render-remote-release-evidence REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence.json BUNDLE_REPORT=/tmp/pfe-release-evidence-bundle.json REMOTE_EVIDENCE_MARKDOWN=/tmp/pfe-remote-release-evidence.md
```

该命令要求最新匹配的 `PFE release gates` run 已 `completed/success`。当前 checkout 尚未提交/推送，远端 workflow 仍未注册，因此当前输出为：

```text
GITHUB ACTIONS RELEASE EVIDENCE MISSING
blocker: matching GitHub Actions run not found
exit_code=2
```

这是最终远端证据步骤，不是当前本地通过证据。

成功采集远端 run 后，`render-remote-release-evidence` 会生成可复制进本文件的 Markdown 摘要；它要求远端证据 JSON 为 `status=passed` 且 `release_ready=true`。

机器可读报告：

```bash
make audit-release-evidence-report AUDIT_REPORT=/tmp/pfe-release-evidence-audit-test.json
.venv/bin/python -m json.tool /tmp/pfe-release-evidence-audit-test.json
```

报告摘要：

```text
status: passed
summary: blocker=0 | ok=10 | total=12 | warn=2
warn: remote_ci_run_evidence
warn: remote_evidence_report
```

bundle manifest：

```bash
make bundle-release-evidence PERF_REPORT=/tmp/pfe-release-perf-report-bundle.json AUDIT_REPORT=/tmp/pfe-release-evidence-audit-bundle.json BUNDLE_REPORT=/tmp/pfe-release-evidence-bundle.json
```

当前结果：

```text
RELEASE EVIDENCE BUNDLE PASSED
summary: blockers=0 | present=3 | total=3 | warnings=1
warning: remote_actions: status=missing release_ready=False
```

严格 release-ready 复核：

```bash
.venv/bin/python tools/release_evidence_audit.py --require-remote --check-remote --skip-process-check
```

当前结果按预期阻塞：

```text
RELEASE EVIDENCE AUDIT BLOCKED
BLOCKER remote_ci_run_evidence: GitHub Actions run URL missing from release evidence
BLOCKER remote_evidence_report: remote evidence report not ready: status=missing release_ready=False run=None
BLOCKER remote_workflow_registered: remote workflow_count=0
exit_code=2
```

## 剩余 release-readiness 缺口

1. CI workflow 已定义且本地 contract 已验证，但远端结果未验证。`gh api repos/leoli001031-blip/Personal-Fine-Tuning-Engine/actions/workflows` 当前返回 `workflow_count=0`；当前 checkout 尚未提交/推送，因此仍需要在 GitHub Actions 上跑一次 manual/nightly job，再用 `make record-remote-release-evidence` 采集 run JSON，并把 run URL、结论和关键日志摘要补进 release evidence。

## 复现顺序

```bash
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -e '.[e2e]'
.venv/bin/python -m pip install 'torch>=2.1' 'transformers>=4.36' 'peft>=0.7' 'accelerate>=0.23' 'safetensors>=0.4'
.venv/bin/python -m playwright install chromium
.venv/bin/python tools/prepare_tiny_hf_model.py
PFE_REAL_LOCAL_MODEL=/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict
make soak-release
.venv/bin/python tools/release_soak_smoke.py --duration-seconds 1800 --interval-seconds 2 --report-path /tmp/pfe-release-soak-30m-report.json
make benchmark-release
make audit-release-evidence
make audit-release-evidence-report AUDIT_REPORT=/tmp/pfe-release-evidence-audit.json
make bundle-release-evidence PERF_REPORT=/tmp/pfe-release-perf-report.json AUDIT_REPORT=/tmp/pfe-release-evidence-audit.json BUNDLE_REPORT=/tmp/pfe-release-evidence-bundle.json
make record-remote-release-evidence REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence.json
make render-remote-release-evidence REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence.json BUNDLE_REPORT=/tmp/pfe-release-evidence-bundle.json REMOTE_EVIDENCE_MARKDOWN=/tmp/pfe-remote-release-evidence.md
```
