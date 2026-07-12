# Phase46 Final Decision

## 结论

最终 recommendation 为 **hold_runtime_and_revise_eval_or_data**。latest-intent runtime 在自动分上有小幅改善，但没有达到预设增益门，也未得到确定性或 Gemma4 盲评支持，因此暂不进入产品默认路径。Phase45 adapter 在同一 runtime 下明显弱于 base，继续保持 archive，不接入 Hermes。

## 真实结果

- 三臂均完成 72-session holdout，每个 session 3 次生成，共 `648` 次正式 Qwen3-4B 调用；三臂均无截断和隐私违规。
- A `base_privacy`：score `0.9361`，latest-intent `0.9028`，old-goal residue `0.0139`。
- B `base_privacy_intent`：score `0.9556`，latest-intent `0.9444`，old-goal residue `0.0`。
- B 对 A：score 增益 `0.0195`，latest-intent 增益 `0.0416`，低于冻结门槛 `0.05`。
- B 对 A Gemma4 盲评：B `24` 胜、A `40` 胜、`8` 平；candidate win rate `0.3333`。
- C `adapter_privacy_intent`：score `0.9121`，latest-intent `0.8472`，repetition `0.2597`，均不支持 adapter 收益。
- B 对 C Gemma4 盲评：B `33` 胜、C `20` 胜、`19` 平。
- 首轮 runtime 发现 Phase46 canary 未被 Phase45 recognizer 脱敏；失败的 216 次调用和原因已完整保留，修复后 9/9 canary 均在进模型前脱敏，正式三臂隐私违规率均为 `0.0`。

## Failed Checks

- `runtime_latest_intent_gain_at_least_0_05`
- `deterministic_runtime_vs_base_win_at_least_0_55`
- `independent_runtime_vs_base_win_at_least_0_55`

## 产品含义

Phase46 证明了 runtime-first ablation 能把“提示契约收益”和“微调收益”分开测量。本轮 runtime envelope 让自动评分略有改善并消除旧目标残留，但盲评更偏好原 privacy base，证据不足以默认启用。

更明确的是，归档 adapter 在相同 runtime 下 score、latest-intent 和重复性都比 base 差。下一轮不应继续扩大训练步数；应先由用户人工审核 48 条异质 correction candidate，再把 runtime envelope 缩减成更轻的边界表达，用 fresh holdout 重测。
