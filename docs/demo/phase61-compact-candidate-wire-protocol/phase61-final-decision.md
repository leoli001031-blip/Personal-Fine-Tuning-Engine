# Phase61 Final Decision

## 结论

最终 recommendation 为 **hold_phase61_compact_candidate_wire_protocol**。Compact-wire preflight 为 `passed`，calibration accuracy 为 `0.9`，holdout accuracy 为 `None`。

## 冻结边界

- Phase61 只把 Phase60 的输出传输改成 `PFE1|source|outcome|relation`；候选生成、typed semantics、hard detector、composer 与门槛均未修改。
- 6 条双模型 wire preflight 先于 30 条 calibration；holdout 仅在 calibration qualified 后运行。
- Wire parser 只接受固定顺序、固定数量的 candidate ID；不接受 JSON、YAML、解释或候选含义复制。
- 每个失败调用最多重试两次，原始无效响应与 SHA-256 必须保留。
- 所有输入都是 simulated evaluator fixtures，不是 actual user feedback，也不用于训练。
- Phase61 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径。
