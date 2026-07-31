# Phase56 Final Decision

## 结论

最终 recommendation 为 **recommend_phase56_span_evaluator_for_manual_review_only**。正式 calibration 准确率 `1.0`，typed exact match `1.0`。独立 holdout 准确率 `1.0`，typed exact match `0.99`，grounding validity `0.99`，无效原子 `9`，危险无依据原子 `0`，字段准确率 `{'source_registration': {'accuracy': 1.0, 'count': 300}, 'test_to_user_outcome_relation': {'accuracy': 0.99, 'count': 300}, 'user_outcome_status': {'accuracy': 1.0, 'count': 300}}`。

本阶段在 Phase55 三个原子字段上增加逐字 evidence span。任何未合格 calibration 尝试只作为 debug evidence，均不得调用 holdout 或成为正式结果。正式独立 holdout 最多调用一次，结果后不得修改 rubric、schema、grounding、composer、fixture 或门槛。

## 架构与边界

- 两个模型只输出三个原子字段及各自的精确原文 span；不输出 accept/edit/reject。
- 无效或缺失 span 不进入 composer；无依据的危险原子保守拒绝。
- 最终标签完全由 deterministic composer 根据 grounded atoms 生成。
- Phase51-55 holdout 只作为历史诊断，没有进入 Phase56 calibration、holdout 或训练。
- runtime replay 调用数：`0`；boundary clause design：未创建。
- Phase56 不训练、不创建 adapter、不接 Hermes、不改 router、runtime prompt 或产品默认路径，不自动 promote。
