# Phase45 Final Decision

## 结论

最终 recommendation 为 **archive**。Qwen3-4B 原生多轮 SFT、两条 160-step 候选、80-session 四组评测和 240 对 Gemma4 盲评均真实完成，但 adapter 没有超过相同隐私 runtime 下的 base，不允许接入 Hermes。

## 真实结果

- 隐私边界：A 的 secret echo 为 `0.0125`，B/D 均为 `0.0`；B/D placeholder leak 和 over-redaction 也均为 `0.0`。
- 公平生成：v1 基准截断率 `0.7222`，失败输出已保留；统一 v2 长度契约后四组正式截断率均为 `0.0`。
- 原生多轮训练：候选 A/B 都完成 160/160 样本全覆盖；候选 A 被独立 18-session diagnostic 选中，最终 holdout 未参与选型。
- D vs B 自动分：`0.8455` vs `0.87`，增益 `-0.0245`。
- D vs B correction：`0.7625` vs `0.8375`；latest intent：`0.675` vs `0.725`。
- D vs B 盲测：deterministic `0.1`，Gemma4 `0.3125`。
- D diversity `0.675`，低于 `0.95` 硬门；repetition `0.209`，仍满足不超过 B + 0.02。

## Failed Checks

- `deterministic_D_vs_B_win_at_least_0_60`
- `independent_D_vs_B_win_at_least_0_60`
- `preference_score_gain_vs_B_at_least_0_08`
- `correction_gain_vs_B_at_least_0_10`
- `diversity_at_least_0_95`

## 产品含义

Phase45 证明了“结构化隐私边界”比让模型自行记住不复述规则更可靠：私密 span 在进入模型前已替换，输出在落盘前再次清洗，并且普通 PID、端口、commit 和公开 ID 没有被误删。

但原生多轮 SFT 没有修复最终 holdout 的 latest correction，反而让 D 相比 B 在 score、correction、latest intent 和 diversity 上下降。因此 adapter 必须 archive；隐私 runtime 和统一长度契约可以保留为独立产品能力，但不能把这次训练描述为产品收益。
