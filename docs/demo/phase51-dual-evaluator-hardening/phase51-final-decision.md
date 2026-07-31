# Phase51 Final Decision

## 结论

最终 recommendation 为 **hold_evaluator_runtime_generalization_gap**。Phase51 把旧 scorer 替换为“确定性硬拒绝 + Gemma4/Qwen3.6 双盲语义评审 + 保守裁决”，校准准确率为 `1.0`，独立 holdout 准确率为 `0.9833`，危险 reject 样例 false accept 为 `0`。

但 runtime 后验审计发现 `1` 条条件句作用域误杀：回答明确要求“真实人工实测确认后”才可登记为 actual feedback，两位语义评审均判 accept，硬规则仍强制 reject。该缺口是在正式 runtime 后发现，只能否决，不能回改本轮结果。

## 三臂真实结果

- compact-v1：accept `0.625`，edit `0.375`，reject `0.0`。
- global-v2：accept `0.8333`，edit `0.0833`，reject `0.0833`。
- conditional guard：accept `0.7917`，edit `0.2083`，reject `0.0`。
- 三臂共 `432` 次真实 Qwen3-4B 调用；runtime 双评审共 `144` 次真实 Ollama 调用。
- 条件 guard 比 compact-v1 更完整，但未达到 `1.0`，且低于 global-v2 的 accept rate；5 条输出只要求人工核验，没有明确“非真实来源不能外推用户收益”。

## 路由与普通任务

- 路由误触发率 `0.0`，漏触发率 `0.0`，逐轮序列准确率 `1.0`。
- 条件 prompt parity `1.0`：触发轮与 global-v2 相同，未触发轮与 compact-v1 相同。
- ordinary task latest-intent 三臂均为 `0.8333`，没有发现条件路由污染普通任务。

## 边界

全部 session 均为 `simulated_usage`，不是 actual user feedback，也不进入训练。Phase51 不训练、不创建 adapter、不接 Hermes、不改变产品默认路径、不自动 promote。后验 simulated-user review 只有否决权，24 条 conditional 输出中有 5 条触发 veto。
