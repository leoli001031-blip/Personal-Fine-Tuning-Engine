# Personal Finetune Engine (PFE) — 开发文档索引

> 一个让任何人都能基于自己的使用数据，在本地持续微调小模型的开源框架。

## 文档结构

| 文档 | 内容 | 适合谁看 |
|------|------|---------|
| [01-overview.md](docs/01-overview.md) | 项目定位、核心能力、用户体验目标、设计原则、术语表 | 所有人 |
| [02-architecture.md](docs/02-architecture.md) | 目录结构、数据流、数据模型、存储设计（含并发与容错）、插件体系、API 设计 | 开发者 |
| [03-module-design.md](docs/03-module-design.md) | 七大核心模块的详细设计（接口、策略、实现细节，含 LLM 数据蒸馏与 Judge 评测） | 开发者 |
| [04-roadmap.md](docs/04-roadmap.md) | Phase 0/1/2 开发计划、每周任务、验收标准、发布策略 | PM + 开发者 |
| [05-tech-and-risk.md](docs/05-tech-and-risk.md) | 技术选型、风险评估、开源运营、商业模式、质量保障 | 决策者 + 开发者 |

## 快速导航

### 我想了解这个项目是什么
→ [01-overview.md](docs/01-overview.md)

### 我想了解技术架构
→ [02-architecture.md](docs/02-architecture.md)

### 我想了解某个模块怎么设计的
→ [03-module-design.md](docs/03-module-design.md)
- Signal Collector（信号采集）
- Data Curator（样本构建）— 最核心模块
- Distillation Pipeline（Teacher / Distiller）
- Trainer（微调训练）— 含 SFT + DPO 训练流程
- Evaluator（效果评测，含 LLM-as-a-Judge）
- Router（推理路由）
- Adapter Store（版本管理）
- Inference Engine（推理引擎）

### 我想了解开发计划和排期
→ [04-roadmap.md](docs/04-roadmap.md)
- Phase 0：技术验证（4 周）
- Phase 1：产品闭环（8 周）
- Phase 2：生态化（8 周）

### 我想了解技术选型和风险
→ [05-tech-and-risk.md](docs/05-tech-and-risk.md)

## 开发环境

项目声明要求 Python `>=3.10`。当前这台机器的系统 `python3` 仍然是 `3.9`，所以开发时建议直接使用仓库内的 Python `3.11` 虚拟环境：

```bash
# 一次性初始化 Python 3.11 开发环境
tools/bootstrap_py311_env.sh

# 激活环境
source .venv/bin/activate

# 确认版本
python --version

# 运行全量测试
python -m unittest discover -s tests -v
```

如果你的 Python 3.11 不在 `/opt/homebrew/bin/python3.11`，可以先设置：

```bash
export PFE_PYTHON_BIN=/path/to/python3.11
tools/bootstrap_py311_env.sh
```

`pfe doctor` 现在也会显示当前 runtime 的 `python_version`、依赖声明的 `requires_python`，以及 `python_supported` 摘要，方便确认是否真的跑在受支持的解释器上。

### 自动激活 `.venv`

仓库里现在已经带了两种自动激活方案：

```bash
# 方案 1：如果你本机装了 direnv
direnv allow

# 方案 2：当前 zsh 会话里直接启用仓库 hook
source tools/pfe-auto-activate.zsh
```

- `/.envrc`：给 `direnv` 使用，进入项目目录时会自动把 `.venv/bin` 放到前面。
- `/tools/pfe-auto-activate.zsh`：不依赖 `direnv`，适合当前机器直接 `source` 一次；之后进入或离开这个仓库目录时，会自动激活或退出 PFE 的 `.venv`。

这边没法直接替你修改 `~/.zshrc`，所以如果你想长期启用 zsh hook，可以自行把下面这一行加进去：

```bash
source /Users/lichenhao/Desktop/PFE/tools/pfe-auto-activate.zsh
```

## 核心命令

```bash
# 初始化项目（下载基座模型）
pfe init --base-model Qwen/Qwen2.5-3B-Instruct

# 生成训练数据（冷启动）
pfe generate --scenario life-coach --style "温和、共情" --num 200

# 或显式走 Teacher / Distillation 管线
pfe distill --teacher-model gpt-4o --scenario life-coach --style "温和、共情" --num 200

# 微调训练（支持 SFT 和 DPO）
pfe train --method qlora --epochs 3

# 基于上一版 adapter 做增量训练
pfe train --incremental --base-adapter 20260401-001 --method qlora --epochs 1

# 效果评测（对比基座模型和微调后模型）
# --base-model base: 使用基座模型（无 adapter）
# --adapter latest: 使用最新 adapter 版本
pfe eval --base-model base --adapter latest --num-samples 20

# 对比两个 adapter 版本的评测结果
pfe eval --adapter 20260401-001 --compare 20260401-002 --num-samples 20

# 查看机器可读状态快照
pfe status --json

# 查看本地 trainer runtime 以及 Python 兼容性
pfe doctor

# 显式启停信号采集
pfe collect start
pfe collect stop

# 启动推理服务（OpenAI API 兼容）
pfe serve --port 8921
```

**参数说明：**
- `pfe eval --base-model base`：`base` 为特殊值，表示使用基座模型（无 adapter）
- `pfe eval --adapter latest`：`latest` 表示最新版本，也可指定具体版本如 `20260401-001`
- `pfe eval --compare`：对比两个已评测版本，并把 compare gate 摘要写入状态面；报告会同时输出通用质量与个性化维度（style / preference / personality）的解释摘要
- `pfe train --incremental --base-adapter`：基于已有 adapter 继续训练，并在训练结果中保留 lineage
- `pfe status --json`：输出机器可读状态，适合脚本和前端消费
- `pfe collect start/stop`：显式控制 `/pfe/signal` 是否接收并沉淀训练信号
- `pfe serve --port`：默认 8921，若被占用自动递增（8922、8923...）
