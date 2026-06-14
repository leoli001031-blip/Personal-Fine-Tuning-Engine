# Release candidate checklist

更新时间：2026-06-15

这份清单定义 PFE 从 Phase 2 closeout 进入 release candidate 的最低证据。它不替代 `release-readiness-evidence.md`；后者记录当前 checkout 已经跑过什么。

## 1. 本地 release gate

在同一个 checkout 中依次执行：

```bash
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -e '.[e2e]'
.venv/bin/python -m pip install 'torch>=2.1' 'transformers>=4.36' 'peft>=0.7' 'accelerate>=0.23' 'safetensors>=0.4'
.venv/bin/python -m playwright install chromium
.venv/bin/python tools/prepare_tiny_hf_model.py
PFE_REAL_LOCAL_MODEL=$HOME/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict
make benchmark-release
make audit-release-evidence
make audit-release-evidence-report AUDIT_REPORT=/tmp/pfe-release-evidence-audit.json
make bundle-release-evidence PERF_REPORT=/tmp/pfe-release-perf-report.json AUDIT_REPORT=/tmp/pfe-release-evidence-audit.json BUNDLE_REPORT=/tmp/pfe-release-evidence-bundle.json
```

等价的一键本地证据入口：

```bash
PFE_REAL_LOCAL_MODEL=$HOME/.cache/pfe/release-models/tiny-gpt2-local make release-local-evidence
```

这个目标会顺序执行 `test-e2e-mock`、`smoke-release-strict`、`benchmark-release`、`audit-release-evidence-report` 和 `bundle-release-evidence`。

通过标准：

- `smoke-release-strict` 必须通过，不允许 Playwright、Chromium 或 `PFE_REAL_LOCAL_MODEL` 缺失导致 skip。
- `benchmark-release` 必须通过默认 release budget。
- dashboard HTML 不允许引用外部 `http://` 或 `https://` 资源。
- `audit-release-evidence` 本地审计必须无 blocker；允许在未提交/未推送前保留远端 run URL warning。
- `audit-release-evidence-report` 必须写出可解析 JSON，作为本地发布证据包或 CI artifact 保存。
- `bundle-release-evidence` 必须写出可解析 JSON manifest，记录各 evidence JSON 的 status、size 和 sha256。
- 命令结束后不应残留 `pfe_core.worker_daemon` 或 `pfe_cli.main serve` 进程。
- 仓库根目录不应生成 `.pfe` 或 `uv.lock`。

## 2. Performance budget

`tools/release_perf_benchmark.py` 默认执行以下预算：

| Task | Max elapsed | Max peak RSS |
|------|-------------|--------------|
| `first_run_full` | `30.0s` | `800.0 MB` |
| `browser_ui_strict` | `30.0s` | `1600.0 MB` |
| `real_local_happy` | `45.0s` | `1800.0 MB` |
| `release_soak_short` | `45.0s` | `1400.0 MB` |

如果只是采样新机器的原始 baseline，可以显式运行：

```bash
.venv/bin/python tools/release_perf_benchmark.py --no-thresholds
```

这不应作为 release 通过证据。

## 3. Long soak gate

Release candidate 前至少执行一次长稳态 soak：

```bash
.venv/bin/python tools/release_soak_smoke.py --duration-seconds 1800 --interval-seconds 2
```

通过标准：

- `healthz`、`/pfe/status`、dashboard API、queue history、daemon status/history、runner status/history 持续返回有效响应。
- daemon 保持 `healthy` / heartbeat `fresh` / lease `valid`。
- chat/feedback round trip 在 soak 期间持续成功。
- 结束后不残留 live server 或 worker daemon 进程。
- 报告路径和 summary 写入 `docs/reference/release-readiness-evidence.md`。

30 分钟通过后，正式发布前建议再跑 60 分钟窗口。

## 4. CI strategy

`.github/workflows/pfe-release-gates.yml` 把 CI 分成两层：

- PR fast gate：`make test-unit`、`make test-surface`、`make test-e2e-mock`、`make smoke-beta`。
- Release/nightly gate：安装 e2e/training 依赖，准备 tiny model，执行 `test-e2e-mock`、`smoke-release-strict`、`benchmark-release`、`audit-release-evidence-report`、`bundle-release-evidence`，并上传 `pfe-release-evidence` artifact。

长 soak 不建议阻塞每个 PR；它应作为 nightly、手动 release job 或发布前本地证据执行。

最终标记 release-ready 前，还需要运行严格 evidence 审计：

```bash
make record-remote-release-evidence REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence.json
make render-remote-release-evidence REMOTE_EVIDENCE_REPORT=/tmp/pfe-github-actions-release-evidence.json BUNDLE_REPORT=/tmp/pfe-release-evidence-bundle.json REMOTE_EVIDENCE_MARKDOWN=/tmp/pfe-remote-release-evidence.md
.venv/bin/python tools/release_evidence_audit.py --require-remote --check-remote --remote-evidence-report /tmp/pfe-github-actions-release-evidence.json
```

这一步必须在 GitHub Actions run URL 已写入 `docs/reference/release-readiness-evidence.md`、`record-remote-release-evidence` 生成的 JSON 为 `status=passed` / `release_ready=true`，且远端 workflow 已注册后通过。`render-remote-release-evidence` 生成的 Markdown 摘要可作为写回该文档的输入。

## 5. Release materials

发布材料已经落地：

- release note：`docs/reference/release-notes-phase2-rc.md`
- 安装/升级说明：`docs/guides/install-upgrade.md`
- known limitations：`docs/reference/known-limitations.md`
- 用户验收 checklist：`docs/reference/user-acceptance-checklist.md`

只有本地 release gate、performance budget、长 soak、发布材料和远端 CI release gate 都具备当前证据后，才能把 PFE 标记为 release-ready。
