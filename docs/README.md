# PFE 文档导航

更新时间：2026-06-15

这套文档按三个层次整理：

- `核心文档`：说明项目是什么、如何设计、当前做到哪一步
- `使用指南`：说明如何接入、配置和运行主要功能
- `参考与归档`：保留长期有效的补充说明和历史记录，但不占主导航第一层

## 推荐阅读顺序

1. [01-overview.md](01-overview.md)
2. [02-architecture.md](02-architecture.md)
3. [03-module-design.md](03-module-design.md)
4. [04-roadmap.md](04-roadmap.md)
5. [05-tech-and-risk.md](05-tech-and-risk.md)

## 核心文档

| 文档 | 说明 |
|------|------|
| [01-overview.md](01-overview.md) | 项目定位、核心能力、设计原则、术语表 |
| [02-architecture.md](02-architecture.md) | 系统架构、目录结构、数据流和接口边界 |
| [03-module-design.md](03-module-design.md) | 主要模块的详细设计 |
| [04-roadmap.md](04-roadmap.md) | 当前阶段状态与后续路线 |
| [05-tech-and-risk.md](05-tech-and-risk.md) | 技术选型、风险与开源策略 |

## 使用指南

| 文档 | 说明 |
|------|------|
| [demo/phase2-demo-runbook.md](demo/phase2-demo-runbook.md) | Phase2 现场 Demo 路径：Studio、记忆样本、训练、自动评估、promote 和 base/local API 对比 |
| [demo/showcase/phase2-product-showcase.md](demo/showcase/phase2-product-showcase.md) | Phase2 产品展示包：真实 Studio 截图、smoke 输出、eval/promote 证据和 base/local API 对比 |
| [guides/openai-closed-loop-integration.md](guides/openai-closed-loop-integration.md) | 如何接入 OpenAI 兼容接口并保持闭环 |
| [guides/auto-loop-policy.md](guides/auto-loop-policy.md) | 自动训练 / 评测 / promote 策略配置 |
| [guides/chat-collector.md](guides/chat-collector.md) | ChatCollector 的信号采集方式 |
| [guides/dpo-training.md](guides/dpo-training.md) | DPO 训练的输入和使用方式 |
| [guides/beta-local-runbook.md](guides/beta-local-runbook.md) | 本地 beta 验证、real-local 预检、live server、dashboard/console 和可选浏览器 smoke |
| [guides/cuda-real-training-validation.md](guides/cuda-real-training-validation.md) | Linux/CUDA 上 PEFT / Unsloth 真实训练验证入口 |
| [guides/install-upgrade.md](guides/install-upgrade.md) | Phase 2 release candidate 的本地安装、升级和 release gate 准备方式 |

## 参考文档

| 文档 | 说明 |
|------|------|
| [reference/data-layering-strategy.md](reference/data-layering-strategy.md) | 数据分层与训练边界 |
| [reference/phase2-closeout.md](reference/phase2-closeout.md) | Phase 2 收尾结果与验证范围 |
| [reference/release-candidate-checklist.md](reference/release-candidate-checklist.md) | Release candidate 必过 gate、性能预算、长 soak 和发布材料清单 |
| [reference/release-readiness-evidence.md](reference/release-readiness-evidence.md) | Release readiness 当前验证证据、strict smoke 结果和剩余 blocker |
| [reference/release-notes-phase2-rc.md](reference/release-notes-phase2-rc.md) | Phase 2 release candidate 变更、证据和用户可期待能力 |
| [reference/known-limitations.md](reference/known-limitations.md) | Phase 2 release candidate 的已知限制和未承诺边界 |
| [reference/user-acceptance-checklist.md](reference/user-acceptance-checklist.md) | 从真实用户视角验收 init/doctor/train/eval/promote/serve/dashboard 闭环 |

## 归档

以下文档保留为历史上下文，但已压缩为少数摘要文件：

- [archive/planning-history.md](archive/planning-history.md)
- [archive/project-history.md](archive/project-history.md)
