# Personal Finetune Engine (PFE)

[English](README.md) | 简体中文

Personal Finetune Engine 是一个本地优先的个性化引擎，用来把用户反馈与行为信号沉淀成一个持续的小模型优化闭环。

```text
collect -> curate -> train -> eval -> promote -> serve
```

[快速开始](#快速开始) • [Studio 主路径](#studio-主路径) • [高级-cli](#高级-cli) • [截图](#截图) • [平台支持](#平台支持) • [文档入口](#文档入口)

PFE 更适合被理解为一个本地优先的模型服务工作台。默认用户入口是浏览器里的 PFE Studio：选择本地模型，复制网页/API 地址，管理个性化模型版本。`pfe` CLI 仍保留给高级诊断、自动化和维护场景。

## PFE 覆盖什么

- 本地环境检查、诊断和运维视图
- 信号采集、curation 和数据控制
- SFT 与 DPO 训练路径
- 评估、candidate 处理、promote 与 archive 流程
- 队列、trigger、daemon 与恢复控制
- 用于模型选择、API 交接和版本管理的 PFE Studio
- OpenAI 兼容的本地服务，以及 dashboard 和 legacy chat 配套界面

## 快速开始

先准备本地环境：

```bash
tools/bootstrap_py311_env.sh
source .venv/bin/activate
```

bootstrap 默认只安装轻量 `dev` extra。需要真实训练依赖时再显式打开：

```bash
PFE_BOOTSTRAP_EXTRAS=dev,training tools/bootstrap_py311_env.sh
```

启动本地 Studio 服务：

```bash
.venv/bin/python -m pfe_server --port 8921 --workspace user_default
```

然后打开：

```text
http://127.0.0.1:8921/
```

Studio 里的主路径是：

1. 选择本地模型，或粘贴本地模型路径。
2. 复制网页地址或 OpenAI 兼容 API 地址。
3. 查看、评估、设为当前、回退或归档模型版本。
4. 训练条件 ready 后生成新版本。

运行快速测试分层：

```bash
make smoke-first-run
make smoke-auto-train-queue
make smoke-real-local-readiness
make smoke-server-live
make smoke-dashboard-console-live
make smoke-browser-ui-live
make test
make test-unit
make test-surface
make test-e2e-mock
```

`make smoke-first-run` 会在隔离的临时目录里跑完
`pfe init -> doctor -> next -> generate -> trigger configure -> collect ingest/status/review -> trigger status/process-next -> eval -> promote -> serve`。
它不需要网络、真实模型下载，也不会启动常驻服务。

`make smoke-auto-train-queue` 会跑同一条隔离路径，但在 auto-train 队列任务完成、
mock adapter manifest 可见后就停止，便于快速验证队列闭环。

`make smoke-real-local-readiness` 会验证不下载模型、不安装重训练依赖时的
real-local 预检路径：本地模型发现、`pfe train --real-local --preview`、
serve preview 和 console snapshot。

`make smoke-server-live` 会在隔离目录里构建一个 mock-local adapter，临时启动
`pfe serve --live`，然后通过 HTTP 检查 `/healthz`、`/pfe/status`、`/dashboard`、
`/pfe/dashboard/metrics` 和 `/v1/chat/completions`。

`make smoke-dashboard-console-live` 会启动同类临时 live server，检查 dashboard
HTML/API 端点，并跑一遍 chat console 的 chat/feedback 闭环。

`make smoke-browser-ui-live` 是可选 Playwright smoke。安装 e2e extras 和 Chromium
浏览器后，它会在真实浏览器里执行 dashboard refresh 和 chat/feedback 交互；未安装时
会带 setup 提示跳过。

`make smoke-real-local-happy` 是显式 opt-in。设置 `PFE_REAL_LOCAL_MODEL` 指向本地
模型或 config 目录，并安装 training extras 后，它会跑真实 `pfe train --real-local`
happy path。

默认本地页面：

```text
http://127.0.0.1:8921/          PFE Studio
http://127.0.0.1:8921/dashboard 观测面板
http://127.0.0.1:8921/chat      legacy operations chat
```

说明：

- `pfe serve --port 8921 --live` 仍然可用，但不再是唯一文档化入口。
- `127.0.0.1:8921` 只是默认本地监听地址，不是硬编码要求。
- 如果当前没有 promoted adapter，服务可以保持在 safe 或 mock 模式。
- 真正加载本地模型需要显式 runtime 配置；Studio 已提供当前进程内的真实本地模型开关。

## Studio 主路径

浏览器里的工作流刻意保持很小：

```text
选择模型 -> 复制 API/网页地址 -> 管理版本
```

Studio 把最常用动作收在一起：

- 工作区切换和创建
- 本地模型选择和模型路径校验
- OpenAI 兼容 `/v1/chat/completions` 地址与 `model` 参数交接
- 当前版本、待确认版本、评估、设为当前、回退和归档
- 训练预检、开始、停止和重试
- 真实本地推理 readiness 与当前回复来源

## 高级 CLI

一条典型的操作者路径：

```bash
# 1. 创建本地 .pfe workspace 和默认配置
pfe init --workspace user_default --base-model Qwen/Qwen2.5-3B-Instruct

# 2. 先确认本地运行环境是否 ready
pfe doctor

# 3. 查看当前引擎状态
pfe next --workspace user_default
pfe status --json

# 4. 打开实时终端面板
pfe console --cycles 1

# 5. 在安装重依赖前预检 real local 训练计划
pfe train --backend peft --real-local --preview --epochs 1

# 6. 启动本地服务
pfe serve --port 8921 --live

# 7. 打开观测面板
pfe dashboard
```

命令分组：

- 工作区初始化：`pfe init --workspace user_default --base-model <path-or-id>`
- 检查与观测：`pfe doctor`、`pfe next`、`pfe status --json`、`pfe console`
- 训练与评估：`pfe train`、`pfe dpo`、`pfe eval`
- 生命周期管理：`pfe adapter`、`pfe candidate`
- 自动化控制：`pfe trigger`、`pfe daemon`、`pfe eval-trigger`
- 采集与数据：`pfe collect`、`pfe data`

当 workspace、base model 和 adapter 流程配置好以后，可以继续看：

```bash
pfe train --backend peft --real-local --preview
pfe train --help
pfe dpo --help
pfe eval --help
pfe adapter --help
pfe trigger --help
```

## 截图

以下画面都来自这个仓库上的真实本地运行。

CLI 画面来自真实 `pfe --help` 和 `pfe doctor` 输出：

<p align="center">
  <img src="docs/assets/screenshots/cli-surfaces.png" alt="PFE CLI surface" width="1100">
</p>

执行 `pfe serve --port 8921 --live` 后访问 `/dashboard`：

<p align="center">
  <img src="docs/assets/screenshots/dashboard.png" alt="PFE dashboard" width="1100">
</p>

PFE Studio 是 `/` 上的默认浏览器入口。dashboard 仍作为更深入的运行观测界面。

## 平台支持

- 当前最顺滑、最优先的本地路径：`macOS` + Apple Silicon
- 代码层已覆盖：`Linux/CUDA` 与 `CPU-only` fallback 路径
- `Windows` 目前还不适合写成主要支持平台

## 默认网络配置

- 默认 host：`127.0.0.1`
- 默认 port：`8921`
- 都可以覆盖，例如：`pfe serve --host 127.0.0.1 --port 3000 --live`

## HTTP 与浏览器 Surface

PFE 也提供本地 HTTP 与浏览器配套界面：

- `GET /healthz`
- `GET /pfe/status`
- `GET /pfe/runtime`
- `GET /pfe/models`
- `GET /pfe/adapters`
- `GET /pfe/readiness`
- `GET /dashboard`
- `POST /v1/chat/completions`

仓库内置页面位于：

- `pfe-server/pfe_server/static/dashboard.html`
- `pfe-server/pfe_server/static/studio.html`
- `pfe-server/pfe_server/static/chat.html`

## 仓库结构

```text
pfe-core/    核心引擎与训练流水线
pfe-cli/     CLI 入口与终端工作流
pfe-server/  FastAPI 服务与 HTTP surface
tests/       单元、surface、integration、e2e 测试
docs/        公开文档、指南、参考与归档
examples/    示例资源与场景
tools/       仓库内辅助脚本
```

## 项目状态

- Phase 1 已完成
- Phase 2 功能性收尾完成
- release readiness 已通过本地与远端 GitHub Actions strict release gate 验证
- 公开仓库版本已经整理完成，大体积本地资产继续排除在仓库之外

Phase 2 收尾说明见 [docs/reference/phase2-closeout.md](docs/reference/phase2-closeout.md)。

## 文档入口

- [README.md](README.md)
- [docs/README.md](docs/README.md)
- [docs/guides/beta-local-runbook.md](docs/guides/beta-local-runbook.md)
- [ENGINE_DEV_DOC.md](ENGINE_DEV_DOC.md)
- [docs/reference/phase2-closeout.md](docs/reference/phase2-closeout.md)

## 许可证

MIT，见 [LICENSE](LICENSE)。

## 仓库边界

这个仓库不包含：

- 本地模型权重
- 训练输出
- 虚拟环境
- 包缓存
- vendored `llama.cpp` checkout 和构建产物

这些内容都属于环境相关资产，不应放进公开源码仓库。
