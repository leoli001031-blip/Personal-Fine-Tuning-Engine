# PFE 项目历史摘要（归档）

更新时间：2026-04-21

本文档压缩保留项目历史 handoff 和阶段交接信息。

之前仓库里曾保留多份按日期拆开的 handoff 文档，以及一份首发 Git 清理流程说明。对内部协作来说这些记录有用，但公开仓库保留太多会让文档层看起来像持续累积的开发现场。

因此这里将它们合并成一份简明历史摘要。

## 1. 2026-03-24：最小闭环成型

这个时间点的项目状态可以概括为：

- `generate / distill (bootstrap) -> train -> eval -> promote -> status -> serve` 的最小闭环已跑通
- `strict_local` 仍是默认边界
- 数据契约、adapter lifecycle、SQLite、CLI、server 和状态面基本定型
- 真实训练和完整 GGUF 工具链还没有完全落地

换句话说，这一阶段解决的是“系统骨架是否成立”。

## 2. 2026-04-08：Phase 0.5 完成，Phase 1 收尾

这个时间点的核心结论是：

- 真实 LoRA 微调已落地
- 真实评测已落地
- `train -> eval -> promote -> serve` 闭环已验证
- llama.cpp GGUF 导出与加载已验证

同时，项目开始从“真实训练能不能做”转向“闭环产品化做得够不够完整”。

这一阶段最重要的工程性推进包括：

- ChatCollector 信号采集
- `/pfe/feedback` 反馈语义统一
- `PFE_HOME` 路径隔离
- DPO 训练后端
- 自动闭环策略配置
- daemon / reliability / queue / console 等运行控制面

## 3. 2026-04-21：Phase 2 收尾与公开仓库整理

这个时间点的核心结论是：

- `Phase 1` 完成
- `Phase 2` 完成
- 项目进入阶段性收尾完成态

在同一轮整理中，还完成了公开仓库首发相关的清理工作：

- 模型权重、训练产物、虚拟环境和缓存不进入源码仓库
- 文档从“开发过程堆叠”改成“核心文档 / 使用指南 / 参考 / 归档”结构
- 项目许可证统一为 `MIT`

## 4. 当前如何使用这份历史摘要

这份文档只回答一类问题：

“这个项目是怎么一步步走到当前状态的？”

如果你想知道：

- 项目当前定位：看 [../01-overview.md](../01-overview.md)
- 当前公开路线：看 [../04-roadmap.md](../04-roadmap.md)
- Phase 2 最终验证边界：看 [../reference/phase2-closeout.md](../reference/phase2-closeout.md)

如果只是想理解历史阶段，这份摘要已经足够，不需要再展开多份旧 handoff 文档。
