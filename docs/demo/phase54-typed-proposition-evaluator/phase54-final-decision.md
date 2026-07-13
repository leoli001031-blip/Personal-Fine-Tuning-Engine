# Phase54 Final Decision

## 结论

最终 recommendation 为 **hold_phase54_typed_proposition_evaluator**。正式 calibration 准确率 `1.0`，typed exact match `0.9833`。独立 holdout 准确率 `0.9933`，typed exact match `0.9433`，字段准确率 `{'current_benefit_assertion': {'accuracy': 1.0, 'count': 300}, 'explicit_provenance_boundary': {'accuracy': 0.9467, 'count': 300}, 'source_eligibility': {'accuracy': 0.9767, 'count': 300}, 'suspended_or_negated_outcome': {'accuracy': 1.0, 'count': 300}}`。

三次未合格 calibration 已作为 debug evidence 保留，分别暴露来源/边界过度推断、Qwen 布尔 schema 不稳定，以及任务背景泄漏；它们均未调用 holdout，也不具备正式结果资格。正式独立 holdout 只调用一次，结果后没有修改 rubric、schema、composer、fixture 或门槛。

## 架构与边界

- 两个模型只输出四个 typed proposition 字段，不输出 accept/edit/reject。
- 最终标签完全由 deterministic composer 生成。
- Phase51-53 holdout 只作为历史诊断，没有进入 Phase54 calibration、holdout 或训练。
- runtime replay 调用数：`0`；boundary clause design：未创建。
- Phase54 不训练、不创建 adapter、不接 Hermes、不改 router、runtime prompt 或产品默认路径，不自动 promote。
