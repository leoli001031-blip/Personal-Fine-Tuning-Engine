# Personal Finetune Engine (PFE)

[English](README.md) | 简体中文

Personal Finetune Engine 是一个本地优先的个性化引擎，用来把用户反馈与行为信号沉淀成一个持续的小模型优化闭环。

```text
collect -> curate -> train -> eval -> promote -> serve
```

PFE 更像一层基础设施，而不是一个开箱即用的消费级聊天产品。它的主入口是 `pfe` CLI，同时配有本地 HTTP 与浏览器界面，用于服务暴露和观测。

## PFE 包含什么

- 本地环境检查、诊断和运维视图
- 信号采集、curation 和数据控制
- SFT 与 DPO 训练路径
- 评估、candidate 处理、promote 与 archive 流程
- 队列、trigger、daemon 与恢复控制
- OpenAI 兼容的本地服务，以及 dashboard / chat 配套界面

## 快速开始

```bash
tools/bootstrap_py311_env.sh
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

建议先跑：

```bash
pfe doctor
pfe status --json
pfe console --cycles 1
```

启动本地服务：

```bash
pfe serve --port 8921 --live
```

打开观测面板：

```bash
pfe dashboard
```

或者直接访问：

```text
http://127.0.0.1:8921/dashboard
http://127.0.0.1:8921/
```

说明：

- 不带 `--live` 的 `pfe serve --port 8921` 只会展示启动计划。
- 如果当前没有 promoted adapter，服务可以保持在 safe / mock 模式。
- 真正加载本地模型通常需要显式 runtime 配置，例如 `--real-local`。

## 常用 CLI 命令

- `pfe doctor`
  检查运行环境是否 ready、后端依赖是否可用、本地模型状态以及 capability blocker。
- `pfe status --json`
  查看 adapter、signal、trigger policy、queue 状态、observability 与 trainer 规划。
- `pfe console`
  打开 Rich 风格的运维终端面板，获得实时终端视图。
- `pfe serve`
  启动本地 FastAPI / OpenAI 兼容推理接口。
- `pfe dashboard`
  在浏览器中打开 observability dashboard。
- `pfe train`、`pfe dpo`、`pfe eval`
  运行核心训练与评估流程。
- `pfe adapter`、`pfe candidate`
  管理 promote、archive 与 candidate 生命周期。
- `pfe trigger`、`pfe daemon`、`pfe eval-trigger`
  管理自动化 gate、worker loop 与恢复流程。
- `pfe collect`、`pfe data`
  管理采集与隐私 / 数据合规 surface。

## 一条典型的使用路径

```bash
# 1. 先看本地运行环境是否 ready
pfe doctor

# 2. 查看当前引擎状态
pfe status --json

# 3. 打开运维终端面板
pfe console --cycles 1

# 4. 启动本地服务
pfe serve --port 8921 --live

# 5. 打开观测面板
pfe dashboard
```

当 workspace、base model 和 adapter 流程配置好以后，可以继续看：

```bash
pfe train --help
pfe dpo --help
pfe eval --help
pfe adapter --help
pfe trigger --help
```

## 截图

以下截图来自这个仓库上的真实本地运行：

`pfe console --cycles 1`

![PFE CLI 终端面板](docs/assets/screenshots/cli-console.png)

执行 `pfe serve --port 8921 --live` 后访问 `/dashboard`

![PFE Dashboard](docs/assets/screenshots/dashboard.png)

## HTTP 与浏览器 Surface

PFE 也提供本地 HTTP 与浏览器配套界面：

- `GET /healthz`
- `GET /pfe/status`
- `GET /dashboard`
- `POST /v1/chat/completions`

仓库内置页面位于：

- `pfe-server/pfe_server/static/dashboard.html`
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
- Phase 2 已完成
- 公开仓库版本已经整理完成，大体积本地资产继续排除在仓库之外

Phase 2 收尾说明见 [docs/reference/phase2-closeout.md](docs/reference/phase2-closeout.md)。

## 文档入口

- [README.md](README.md)
- [ENGINE_DEV_DOC.md](ENGINE_DEV_DOC.md)
- [docs/README.md](docs/README.md)
- [docs/01-overview.md](docs/01-overview.md)
- [docs/02-architecture.md](docs/02-architecture.md)
- [docs/04-roadmap.md](docs/04-roadmap.md)
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
