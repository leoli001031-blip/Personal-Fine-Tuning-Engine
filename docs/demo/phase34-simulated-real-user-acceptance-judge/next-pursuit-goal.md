目标：开发并验证 PFE Phase35：真实在线 Agent 使用采集 + Phase34 模拟验收校准闭环。

请在 [LOCAL_PATH] 中完成：

1. 接入 Hermes/PFE 在线交互采集，记录用户目标、Agent 回复、纠正、继续推进和最终验收。
2. 真实记录必须有授权、脱敏、operator、timestamp、source_id，并进入 review queue。
3. 将 Phase34 simulated_user_judgement 与真实在线验收反馈做差异分析，校准模拟验收官。
4. simulated_user_judgement 不得进入 actual_user_feedback 训练候选。
5. 只有 approved actual_user_feedback 达标后，才生成训练候选或 DPO pairs。
6. 不自动 promote；最高 recommendation 仍是 promote_after_manual_review。
