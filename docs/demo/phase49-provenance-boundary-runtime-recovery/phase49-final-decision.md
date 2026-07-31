# Phase49 Final Decision

## 结论

最终 recommendation 为 **hold_provenance_compact_v2**。compact-v2 的单句证据边界将 provenance 从 v1 的 `0.875` 提升到 `1.0`，且 unsupported claim 为 `0`；但它让普通任务 latest-intent 从 `0.875` 降至 `0.75`，整体也未在盲评中超过 privacy base。因此不进入产品默认路径或 manual shadow。

## Loop Engineering 过程

- attempt-01 完成 `192` 次 Qwen3-4B 调用后发现 scorer 把“无用户反馈”“无法断言产品收益”等正确表达判为 edit；该 holdout、transcript 和 freeze 已完整保存并标记 `formal_result_eligible=false`。
- scorer 校准扩展到 `26` 条自然表达，exact accuracy `1.0`；attempt-02 使用全新 64-session holdout，与 Phase48 和失效 holdout 均零文本重叠。
- attempt-02 三臂各 64 个三轮 session，共 `576` 次正式 Qwen3-4B 调用；隐私违规、unsupported claim、截断和 think 泄漏均为 `0`。

## 正式结果

- privacy base：score `0.873`，provenance `0.9375`，repetition `0.1051`。
- compact-v1：score `0.8575`，provenance `0.875`，ordinary latest `0.875`。
- compact-v2：score `0.8508`，provenance `1.0`，ordinary latest `0.75`。
- Gemma4 v2 对 v1：`14` 胜、`11` 负、`39` 平，非平局胜率 `0.56`。
- Gemma4 v2 对 base：`17` 胜、`18` 负、`29` 平，非平局胜率 `0.4857`。

## Failed Checks

- `v2_ordinary_latest_not_below_v1_minus_0_02`
- `independent_v2_vs_base_non_tie_win_at_least_0_55`

## 产品含义

证据边界 clause 本身有效，但作为全局 system 指令会干扰普通任务。下一步不应训练，也不应继续加全局提示；应只在识别到“把模拟/自动结果外推为真实反馈或产品收益”的任务时条件启用该 clause，再用 fresh holdout 验证路由误触发率。所有复盘与 holdout 均为 simulated_usage，`actual_user_feedback_count=0`。
