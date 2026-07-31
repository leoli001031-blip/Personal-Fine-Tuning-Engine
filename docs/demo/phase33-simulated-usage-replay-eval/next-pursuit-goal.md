目标：开发并验证 PFE Phase34：真实在线 Agent 使用采集 + Phase33 回放校准闭环。

请在 [LOCAL_PATH] 中完成：

1. 基于 Phase33 的 simulated_usage replay taxonomy，接入真实 Hermes/PFE 在线交互采集。
2. 所有真实采集必须经过用户授权、脱敏、review queue 和 holdout 隔离。
3. 将 Phase33 的模拟评分与真实在线反馈评分做差异分析，找出模拟器过度乐观或漏判的场景。
4. 只有 approved actual_user_feedback 达标后，才生成训练候选；Phase33 simulated_usage 不得进入真实训练。
5. 对 base、Phase32 adapter、真实反馈候选 adapter 做同场景对比，保存 transcripts、评分、隐私扫描和 decision。
6. 不自动 promote；最高 recommendation 仍是 promote_after_manual_review。
