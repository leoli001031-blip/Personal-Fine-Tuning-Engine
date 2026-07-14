# Phase66 Final Decision

## 结论

最终 recommendation 为 **hold_phase66_external_distribution_regression**。全新外部分布 holdout 为 `1.0`，但 Phase51-55 历史分布仅为 `0.6595`，相对 Phase64 `0.6416` 只提升 `0.0179`，未达到冻结的 `0.2` 提升门槛。

## 真实证据

- 新 holdout：150/150，false accepts `0`，schema failures `0`，candidate conflicts `0`。
- 历史 replay：完成 `549`/558，9 个 judge-item 在两次重试后仍格式失败，schema failures `9`，candidate conflicts `3`，false accepts `0`。
- 各阶段准确率：`{'phase51': {'accuracy': 0.65, 'count': 60, 'false_accept_count': 0}, 'phase52': {'accuracy': 0.6111, 'count': 90, 'false_accept_count': 0}, 'phase53': {'accuracy': 0.6111, 'count': 108, 'false_accept_count': 0}, 'phase54': {'accuracy': 0.6533, 'count': 150, 'false_accept_count': 0}, 'phase55': {'accuracy': 0.7333, 'count': 150, 'false_accept_count': 0}}`。

## 边界

- Phase65 candidate rule、Phase63 wire、Phase62 consensus、Phase56 composer 和所有门槛在调用前冻结，调用后未修改。
- 所有输入都是 simulated evaluator fixtures，不是 actual user feedback，也不用于训练。
- Phase66 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改默认路径、不自动 promote。
