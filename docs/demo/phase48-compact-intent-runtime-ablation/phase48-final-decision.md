# Phase48 Final Decision

## 结论

最终 recommendation 为 **hold_compact_runtime**。compact runtime 在总分和两组独立盲评上表现更好，但没有达到冻结的 latest-intent 增益门，并且低于 full envelope 的 latest-intent 下限，因此不进入产品默认路径，也不进入 manual shadow。

## 真实结果

- 三组均完成 64 个多轮 session，每个 session 3 次生成，共 `576` 次真实 Qwen3-4B 调用；隐私违规、截断和 think 泄漏均为 `0`。
- `base_privacy`：score `0.8953`，latest-intent `0.8281`，repetition `0.1595`。
- `base_compact_intent`：score `0.9098`，latest-intent `0.8438`，repetition `0.1737`。
- `base_full_intent`：score `0.9019`，latest-intent `0.875`，old-goal residue `0.0156`，repetition `0.2829`。
- compact 对 privacy base：score 增益 `0.0145`，latest-intent 增益 `0.0157`，低于冻结门槛 `0.03`。
- Gemma4 盲评 compact 对 base：`22` 胜、`15` 负、`27` 平，非平局胜率 `0.5946`。
- Gemma4 盲评 compact 对 full：`34` 胜、`16` 负、`14` 平，非平局胜率 `0.68`。
- category 级证据显示 compact 改善 `evidence_status` 和 `ordinary_direct_task`，但 `provenance_boundary` 的 latest-intent 相比 base 下降 `0.125`、相比 full 下降 `0.25`。

## Failed Checks

- `compact_latest_intent_gain_at_least_0_03`
- `compact_latest_intent_not_below_full_minus_0_02`

## 产品含义

短契约比 full envelope 更自然、重复更少，盲评偏好明确；但它没有稳定守住来源边界，不能仅凭平均分和 judge 偏好默认启用。Phase48 没有训练、没有新 adapter、没有接入 Hermes，`actual_user_feedback_count=0`，所有场景均为 simulated_usage。
