# PFE 文档导航

更新时间：2026-04-21

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
| [guides/openai-closed-loop-integration.md](guides/openai-closed-loop-integration.md) | 如何接入 OpenAI 兼容接口并保持闭环 |
| [guides/auto-loop-policy.md](guides/auto-loop-policy.md) | 自动训练 / 评测 / promote 策略配置 |
| [guides/chat-collector.md](guides/chat-collector.md) | ChatCollector 的信号采集方式 |
| [guides/dpo-training.md](guides/dpo-training.md) | DPO 训练的输入和使用方式 |

## 参考文档

| 文档 | 说明 |
|------|------|
| [reference/data-layering-strategy.md](reference/data-layering-strategy.md) | 数据分层与训练边界 |
| [reference/phase2-closeout.md](reference/phase2-closeout.md) | Phase 2 收尾结果与验证范围 |

## 归档

以下文档保留为历史上下文，但已压缩为少数摘要文件：

- [archive/planning-history.md](archive/planning-history.md)
- [archive/project-history.md](archive/project-history.md)
