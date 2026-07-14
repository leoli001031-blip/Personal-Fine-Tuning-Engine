# Phase57 Final Decision

## 结论

最终 recommendation 为 **hold_phase57_span_evaluator_historical_replay**。Phase51-55 历史 holdout 总准确率为 `0.6505`，各阶段结果为 `{'phase51': {'accuracy': 0.6667, 'count': 60, 'false_accept_count': 0}, 'phase52': {'accuracy': 0.6667, 'count': 90, 'false_accept_count': 0}, 'phase53': {'accuracy': 0.5833, 'count': 108, 'false_accept_count': 0}, 'phase54': {'accuracy': 0.6667, 'count': 150, 'false_accept_count': 0}, 'phase55': {'accuracy': 0.6667, 'count': 150, 'false_accept_count': 0}}`，grounding validity 为 `0.79`。

## 冻结边界

- Phase56 prompt、JSON schema、quote mask、clause grounding 和 deterministic composer 未修改。
- 历史 phase、case id 与 gold label 对两个 judge 全部隐藏。
- 历史 replay 结果不得用于本阶段调参；失败即 hold。
- Phase57 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径。
- 所有输入均为历史 simulated evaluator fixtures，不是 actual user feedback，也不用于训练。
