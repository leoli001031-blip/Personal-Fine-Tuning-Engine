# Phase50 Final Decision

## 结论

最终 recommendation 为 **hold_conditional_provenance_guard_evaluator_unstable**。条件路由在 32 个来源外推场景与 32 个 passthrough 场景上的误触发率为 `0.0`，漏触发率为 `0.0`，逐轮序列准确率为 `1.0`。条件 guard 的 provenance 为 `0.375`，unsupported claim 为 `0.0312`。

冻结 scorer 与后验 simulated-user review 的标签一致率只有 `0.2188`；32 条输出中发现 `7` 条明确把测试来源升级为实际反馈或产品收益。后验审计只有否决权，因此 Phase50 不允许进入 manual shadow。

## 真实模型对照

- compact-v1：score `0.7975`，provenance `0.3438`，ordinary latest `0.5`。
- global-v2：score `0.8005`，provenance `0.3125`。
- conditional guard：score `0.785`，provenance `0.375`，ordinary latest `0.5`。
- 三臂各 64 个三轮 session，共 `576` 次真实 Qwen3-4B 调用，无 adapter、无训练。
- Prompt parity `1.0`：激活轮与 global-v2 完全同 prompt，未激活轮与 compact-v1 完全同 prompt。

## Loop Engineering 过程

- attempt-01 完成 576 次 Qwen3-4B 与 64 次 Gemma4 调用后，发现 Phase49 scorer 同时存在否定表达漏判和“人工验证”泛化误判；全部证据已保存在 `evidence-scorer-debug/attempt-01-phase49-scorer-gap/`，并标记 `formal_result_eligible=false`。
- attempt-02 在首个 192-call 控制臂发现新 canary 前缀不受既有隐私转换器支持，因此立即停止剩余两臂；失败转录已保留并排除。
- Phase50 专用 scorer 用 `36` 条自然表达校准，exact accuracy `1.0`；正式 attempt-03 使用全新 holdout，与 attempt-01/02、Phase48、Phase49 和候选数据零精确文本重叠。
- attempt-03 后验 simulated-user review 将 32 条触发输出标为 25 accept、7 reject；与冻结 scorer 的一致率仅 `0.2188`。该结果不能用于 promote，但足以否决当前 guard。

## 独立盲评

- Trigger slice，conditional 对 compact-v1：`15` 胜、`5` 负、`12` 平，非平局胜率 `0.75`。
- Passthrough slice，conditional 对 global-v2：`9` 胜、`6` 负、`17` 平，非平局胜率 `0.6`。

## Failed Checks

- `conditional_provenance_boundary_exact`
- `conditional_unsupported_product_benefit_zero`
- `conditional_score_not_below_v1_minus_0_01`
- `deterministic_trigger_candidate_wins_more`
- `deterministic_passthrough_candidate_not_worse`
- `formal_promotion_evaluator_valid`

## 边界

所有场景均为 simulated_usage，`actual_user_feedback_count=0`。本阶段不训练、不创建 adapter、不接 Hermes、不自动 promote，也不改变产品默认路径。即便全部 gate 通过，也只允许 manual shadow。
