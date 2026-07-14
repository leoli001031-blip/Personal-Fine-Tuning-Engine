# Phase60 Final Decision

## 结论

最终 recommendation 为 **hold_phase60_flat_schema_compatibility_recovery**。Flat-schema preflight 为 `failed`，calibration accuracy 为 `None`，holdout accuracy 为 `None`。

## 冻结边界

- Phase60 只把 Phase59 的三组 candidate_id 改成顶层字符串字段；候选生成、typed semantics、hard detector、composer 与门槛均未修改。
- 6 条双模型 schema preflight 先于 30 条 calibration；holdout 仅在 calibration qualified 后运行。
- 每个失败调用最多重试两次，原始无效响应与 SHA-256 必须保留。
- 所有输入都是 simulated evaluator fixtures，不是 actual user feedback，也不用于训练。
- Phase60 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径。
