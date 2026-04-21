# PFE 路线图

更新时间：2026-04-21

## 当前状态

截至 2026-04-21，PFE 已经完成两个关键阶段：

- `Phase 1`：产品闭环完成
- `Phase 2`：质量深化完成

当前主链能力已经稳定在：

```text
collect -> curate -> train -> eval -> promote -> serve
```

围绕这条主链，仓库已经具备：

- Signal 采集与样本整理
- SFT / DPO 训练路径
- Adapter 生命周期与 promote / rollback
- 评测、状态面、审计与可观测性
- OpenAI 兼容推理服务

`Phase 2` 的收尾验证见 [reference/phase2-closeout.md](reference/phase2-closeout.md)。

## 当前判断

PFE 现在更适合被理解为：

- 一个已经完成 `Phase 1 + Phase 2` 主线目标的本地个性化微调引擎
- 一个适合公开展示和继续扩展的开源仓库
- 一个应当减少过程性开发痕迹，转向更清晰公开文档和阶段性发布的项目

## 下一阶段：Phase 3

后续如果继续推进，优先级建议是 `Phase 3`，重点不再是补 `Phase 2` 的尾巴，而是扩大生态边界：

1. 插件体系
   - Collector / Curator / Evaluator / Trainer Backend 扩展接口
2. 模型兼容矩阵
   - Llama 3、Mistral、Phi、Gemma 等更多基座模型
3. 示例应用
   - Writing Assistant、Code Assistant 等完整官方示例
4. 文档与发布
   - 更清晰的公开文档、使用说明、版本发布节奏
5. 长时稳定性与性能验证
   - soak test、资源占用和更重负载验证

## 公开仓库的文档原则

当前文档结构已经从“开发现场”整理为三层：

- 核心文档：说明项目定位、架构和路线
- 使用指南：说明如何接入和使用
- 参考 / 归档：保留补充说明和历史记录，但不占首页主导航

这意味着后续再继续开发时，建议：

- 新增设计优先落在核心文档或使用指南
- 阶段性交接、草稿规划和发布过程说明优先进入 `docs/archive/`
- 避免再把强时效性、强过程性的文档直接堆在 `docs/` 第一层

## 历史规划

历史规划和阶段性交接已经压缩进归档摘要：

- [archive/planning-history.md](archive/planning-history.md)
- [archive/project-history.md](archive/project-history.md)

这些文档主要用于回看历史决策，不再代表当前公开文档的主入口。
