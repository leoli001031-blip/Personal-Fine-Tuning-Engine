# Phase44 Final Decision

## 结论

最终 recommendation 为 **archive**。120-step Qwen3-4B SFT 真实训练和 adapter 产物均成功，但产品收益门没有通过，不允许 promote，也不进入 Hermes shadow trial。

## 真实结果

- 训练：120/120 样本各曝光一次，10 个类别全覆盖；loss `6.674794` -> `1.470769`。
- SFT vs base 自动分：`0.8225` vs `0.7855`，增益 `0.037`，低于 `+0.10` 门槛。
- SFT vs base 盲测：deterministic `0.3667`，Gemma4 `0.4167`。
- SFT vs soft runtime 盲测：deterministic `0.3`，Gemma4 `0.3167`。
- 隐私复述：base `0.1167`，SFT `0.1333`，未达到必须为 `0` 的硬门。
- diversity：base `0.85`，SFT `0.7833`；repetition：base `0.1229`，SFT `0.2578`。
- 所有 A/B/C/D 正式输出共 `720` 次 Qwen 调用；独立评审 `180` 次 Gemma4 调用。

## Failed Checks

- `deterministic_win_vs_base_at_least_0_60`
- `independent_win_vs_base_at_least_0_60`
- `deterministic_win_vs_soft_runtime_at_least_0_60`
- `independent_win_vs_soft_runtime_at_least_0_60`
- `preference_score_gain_at_least_0_10`
- `correction_gain_at_least_0_10`
- `privacy_violation_zero`
- `diversity_at_least_0_95`
- `repetition_not_over_base_plus_0_02`
- `judges_agree`

## 产品含义

Phase44 解决了 Phase43 的固定顺序暴露问题，也证明课程可以被模型真实学到；但“学到风格”没有转化为稳定产品收益，反而带来模板复用、重复与隐私复述。adapter 作为归档实验保留，不写入正式 promoted 状态。

下一步应优先把隐私样本改成不含可复制 secret 的结构化占位训练，并缩短 completion、增加普通任务与多样化 paraphrase；继续使用 soft runtime 也必须先修隐私边界。DPO 继续禁用，直到 non-finite 根因被独立解决。
