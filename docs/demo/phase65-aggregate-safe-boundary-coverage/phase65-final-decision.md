# Phase65 Final Decision

## 结论

最终 recommendation 为 **recommend_phase65_scope_aware_candidates_for_manual_review_only**。Preflight 为 `passed`，calibration accuracy 为 `1.0`，holdout accuracy 为 `1.0`。

## 冻结边界

- 唯一结构改动是关系分句中保留明确的安全 outcome 候选，同时继续排除嵌套的危险 asserted outcome。
- 危险候选识别、Phase62 risk-asymmetric consensus、Phase63 PFE2 wire、composer 和评分门槛均未放宽。
- 新 calibration/holdout 只依据 Phase64 聚合失败类型设计，不复用单条历史失败原文。
- 所有输入均为 simulated evaluator fixtures，不是 actual user feedback，也不用于训练。
- Phase65 不运行 runtime A/B、不训练、不创建 adapter、不接 Hermes、不改产品默认路径。
