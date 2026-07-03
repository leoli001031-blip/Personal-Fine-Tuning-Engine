目标：开发并验证 PFE Phase36：本地交互 review queue + approved actual feedback candidate generation。

请在 [LOCAL_PATH] 中完成：

1. 读取 Phase35 local interaction store。
2. 提供 review decision：approve_for_candidate / exclude / quarantine。
3. 只有 operator attested actual local interactions 可进入候选。
4. simulated_local_interaction 和缺 consent 的记录必须继续排除。
5. 生成 SFT/DPO candidates，但不自动训练。
6. 保存 evidence、focused tests、runbook、final decision。
7. 不接 Hermes，不修改 videos，不自动 promote。
