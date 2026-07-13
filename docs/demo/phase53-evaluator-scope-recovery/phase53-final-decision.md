# Phase53 Final Decision

## 结论

最终 recommendation 为 **hold_phase53_evaluator_scope_recovery**。正式 calibration 准确率为 `0.9889`，分项为 `{'mixed_affirmative_negative_clauses': {'accuracy': 1.0, 'count': 30}, 'postposed_negation': {'accuracy': 0.9667, 'count': 30}, 'punctuation_dependent_scope': {'accuracy': 1.0, 'count': 30}}`。独立 holdout 准确率为 `0.8519`，分项为 `{'mixed_affirmative_negative_clauses': {'accuracy': 0.8333, 'count': 36}, 'postposed_negation': {'accuracy': 0.8611, 'count': 36}, 'punctuation_dependent_scope': {'accuracy': 0.8611, 'count': 36}}`。失败 `16` 条，其中双模型分歧 `10` 条，标签迁移为 `{'accept->edit': 3, 'accept->reject': 3, 'edit->reject': 10}`。

前两次 calibration 均已保留为非正式证据：attempt-01 暴露 edit 定义和含混文本问题；attempt-02 改善到 `0.9222`，但有 1 条 Qwen JSON 因 192-token 截断而不完整。正式协议固定 `num_predict=384`。

## 决策边界

- Phase51 runtime replay 调用数：`0`。
- boundary clause design：未创建。
- 所有数据均为 simulated evaluator fixture，不是 actual user feedback，不进入训练。
- Phase53 不训练、不创建 adapter、不接 Hermes、不改 router、不改 runtime prompt、不改产品默认路径、不自动 promote。
