# Phase62 Final Decision

## 结论

最终 recommendation 为 **hold_phase62_risk_asymmetric_candidate_consensus**。Protocol preflight 为 `passed`，calibration accuracy 为 `0.9667`，holdout accuracy 为 `None`。

## 冻结边界

- Phase62 复用 Phase61 compact wire，只在 composer 前增加 risk-asymmetric consensus。
- 任一 judge 发现危险候选即保留并触发保守拒绝；安全候选可在另一 judge abstain 时保留；实际候选冲突由危险侧优先。
- 评分门槛、候选生成、hard detector 与 Phase56 composer 未修改；holdout candidate-value conflict 必须为零。
- 6 条 protocol preflight 先于 30 条 calibration；holdout 仅在 calibration qualified 后运行。
- 所有输入都是 simulated evaluator fixtures，不是 actual user feedback，也不用于训练。
- Phase62 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径。
