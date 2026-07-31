目标：推进 PFE Phase29：真实反馈达标后的 Qwen 小步训练收益验证。

当前 Phase28 状态：
- actual_feedback_count = 0
- approved_actual_candidate_count = 0
- loop_state = observe
- final_recommendation = collect_more_actual_feedback

下一步只有在真实 approved feedback >= 12 时才启动训练；否则继续收集和审核真实反馈，不使用 template/simulation 数据补齐。
