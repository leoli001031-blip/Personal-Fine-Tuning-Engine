# Install and upgrade guide

更新时间：2026-06-15

这份指南描述 PFE Phase 2 release candidate 的本地安装、升级和 release gate 准备方式。

## 1. Python 环境

推荐 Python 3.11。项目要求 Python `>=3.10`，但当前 release evidence 使用的是：

```bash
.venv/bin/python --version
```

期望：

```text
Python 3.11.15
```

如果当前 `.venv` 可疑，尤其是脚本 shebang 指向旧路径，直接重建：

```bash
tools/bootstrap_py311_env.sh
source .venv/bin/activate
```

## 2. 基础安装

```bash
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

常规开发和 beta smoke：

```bash
.venv/bin/python -m pip install -e '.[e2e]'
make smoke-beta
```

`smoke-beta` 是轻量可用性 gate；它不是 release-ready 证明。

## 3. Strict release gate 依赖

安装浏览器和 real-local tiny model 所需依赖：

```bash
.venv/bin/python -m pip install -e '.[e2e]'
.venv/bin/python -m pip install 'torch>=2.1' 'transformers>=4.36' 'peft>=0.7' 'accelerate>=0.23' 'safetensors>=0.4'
.venv/bin/python -m playwright install chromium
```

准备 no-download tiny model：

```bash
.venv/bin/python tools/prepare_tiny_hf_model.py
```

默认输出：

```text
$HOME/.cache/pfe/release-models/tiny-gpt2-local
```

## 4. Release gate

```bash
PFE_REAL_LOCAL_MODEL=$HOME/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict
make benchmark-release
.venv/bin/python tools/release_soak_smoke.py --duration-seconds 1800 --interval-seconds 2 --report-path /tmp/pfe-release-soak-30m-report.json
```

通过标准：

- `smoke-release-strict` 不允许 skip browser 或 model gate。
- `benchmark-release` 必须通过默认 performance budget。
- 30 分钟 soak 必须保持 daemon `healthy/fresh/valid`，并完成 chat/feedback round trip。
- 命令结束后不应残留 `pfe_core.worker_daemon` 或 `pfe_cli.main serve` 进程。
- 仓库根目录不应生成 `.pfe` 或 `uv.lock`。

## 5. 升级检查

升级已有 checkout 后，依次检查：

```bash
.venv/bin/python -m pip install -e '.[e2e]'
.venv/bin/python tools/prepare_tiny_hf_model.py
PFE_REAL_LOCAL_MODEL=$HOME/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict
make benchmark-release
```

如果 `pfe doctor` 显示 trainer deps 缺失，只影响真实训练能力，不一定影响 light beta smoke。Release gate 需要至少安装 `torch`、`transformers`、`peft`、`accelerate` 和 `safetensors`。

## 6. 常见问题

`smoke-release-strict` 因 `PFE_REAL_LOCAL_MODEL` 缺失失败：

```bash
.venv/bin/python tools/prepare_tiny_hf_model.py
export PFE_REAL_LOCAL_MODEL=$HOME/.cache/pfe/release-models/tiny-gpt2-local
```

Playwright/Chromium 缺失：

```bash
.venv/bin/python -m pip install -e '.[e2e]'
.venv/bin/python -m playwright install chromium
```

Dashboard 访问正常但 metrics 偶发慢：

- 当前 release candidate 已加入 dashboard metrics cache/stale fallback。
- 长 soak 会覆盖 `/pfe/dashboard/metrics` 的持续响应。

本地临时 workspace 污染：

- smoke 和 soak 默认使用临时目录。
- release gate 后检查仓库根目录不应出现 `.pfe` 或 `uv.lock`。
