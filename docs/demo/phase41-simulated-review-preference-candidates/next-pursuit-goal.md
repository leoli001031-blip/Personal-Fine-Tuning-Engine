目标：开发并验证 PFE Phase42：基于 simulated reviewed preference 的小模型训练 probe。

请基于 Phase41 的 selected_preference_pairs.jsonl 生成一个最小 DPO/SFT 训练尝试，不训练 27B，不自动 promote。训练完成后必须用 Phase40/41 的 holdout-style simulated usage 场景对 base、runtime contract、adapter 做同场对比，并继续标记为 simulated lab evidence，不能宣称 actual product benefit。
