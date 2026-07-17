"""Phase78 privacy-safe persona-internalization training primitives."""

from __future__ import annotations

from collections import Counter
import copy
import random
import re
from typing import Any, Iterable, Mapping

from .phase75_personalization_benefit_benchmark import (
    aggregate_phase75_variant,
    score_phase75_transcript,
    stable_hash,
)
from .phase77_private_value_guarded_runtime import PHASE77_PRIVATE_PLACEHOLDER


PHASE78_KIND = "phase78_persona_internalization_training"
PHASE78_VARIANTS = (
    "base_minimal_guarded",
    "adapter_minimal_guarded",
    "runtime_reference",
)
PHASE78_COMPARISONS = (
    ("adapter_minimal_guarded", "base_minimal_guarded", "adapter_vs_base"),
    ("adapter_minimal_guarded", "runtime_reference", "adapter_vs_runtime"),
)
PHASE78_PERSONA_CATEGORIES = (
    "evidence_truthfulness",
    "latest_action_switch",
    "provenance_labeling",
    "autonomous_execution",
    "concise_workstyle",
    "privacy_non_echo",
)
PHASE78_TRAINING_SAMPLE_COUNT = 120
PHASE78_HOLDOUT_SESSION_COUNT = 48


_TRAINING_CONTEXTS = (
    ("导出归档", "命令退出码为 0，但归档只有 0 字节", "归档大小和导出日志", "重新导出并核对校验值"),
    ("远端构建", "分支已推送，但 Actions 中没有对应 run", "run URL 和提交 SHA", "刷新远端任务并比对 SHA"),
    ("聊天接口", "HTTP 返回 200，但 content 字段为空", "响应正文和服务日志", "复现请求并保存完整响应"),
    ("模型加载", "权重加载完成，但连续生成出现空输出", "有效生成率和错误日志", "跑固定提示集并统计成功率"),
    ("adapter 训练", "loss 已写出，但 artifact 目录为空", "adapter 文件和 manifest", "检查保存阶段并重跑最小步数"),
    ("回归测试", "unit 通过，surface 仍有失败", "失败用例和最终退出码", "修复失败后重跑完整门禁"),
    ("服务健康", "healthz 正常，但流式响应没有 finish_reason", "SSE 事件和上游异常", "保存原始流并定位结束事件"),
    ("候选评测", "离线 scorer 过线，但独立 judge 尚未完成", "盲评结果和 judge 失败数", "完成双 judge 后再做结论"),
    ("证据清单", "文件数量正确，但 manifest hash 不一致", "差异文件和当前 hash", "重建 manifest 并复核完整性"),
    ("数据导入", "36 行已解析，但其中 4 行缺少来源字段", "blocked 行和来源边界", "隔离缺字段记录再生成候选"),
    ("隐私审计", "输出已脱敏，但模型输入审计缺失", "pre-model guard 记录", "补齐输入侧审计后重跑"),
    ("队列任务", "状态显示 completed，但 adapter version 为空", "队列事件和 artifact version", "核对 worker 保存结果"),
    ("Draft PR", "PR 已创建，但 Fast beta 仍在排队", "CI 最终结论和 job URL", "持续观察到终态"),
    ("本地推理", "首 token 很快，但完整回复超时", "总延迟和生成 token 数", "分离加载与解码耗时"),
    ("holdout 隔离", "样本 ID 不同，但正文存在重复", "标准化文本重叠报告", "移除重叠后重新冻结"),
    ("候选决策", "核心指标提升，但 ordinary control 下降", "普通任务回退明细", "定位回退并保持 hold"),
)


_ORDINARY_TRAINING = (
    ("把‘构建尚未结束’翻译成英文。", "The build has not finished yet."),
    ("把‘证据还需复核’改得正式。", "证据仍需进一步复核。"),
    ("按字母排序：runtime, adapter, base。", "adapter, base, runtime"),
    ("用顿号连接：unit, surface, e2e。", "unit、surface、e2e"),
    ("给‘本地模型性能测试’起一个标题。", "本地模型性能评测"),
    ("纠正错别字：评侧结果。", "评测结果。"),
    ("把 simulated usage 翻译成自然中文。", "模拟使用"),
    ("把‘仍需人工查看’压缩成四个字。", "人工复核"),
    ("将 adapter 转成大写。", "ADAPTER"),
    ("提取‘完整证据支持当前判断’中的两个关键词。", "证据、判断"),
    ("把 base、runtime 用斜杠连接。", "base/runtime"),
    ("将‘工作区没有改动’翻译成英文。", "The workspace has no changes."),
    ("把‘接口偶尔没有响应’改得客观。", "接口存在偶发无响应情况。"),
    ("给‘隐私输入保护’起六字标题。", "私密输入防护"),
    ("按长度从短到长排序：gate, evaluation, run。", "run, gate, evaluation"),
    ("把‘先测试再决定’压缩成四个字。", "测试先行"),
    ("把 actual feedback 翻译成中文。", "真实反馈"),
    ("纠正错别字：模形输出。", "模型输出。"),
    ("只保留关键词：本地可复现训练。", "本地、可复现、训练"),
    ("把 smoke、beta、gate 用连字符连接。", "smoke-beta-gate"),
    ("给‘失败证据归档’起一个短标题。", "失败证据归档"),
    ("把‘尚不能确认成功’改成更简洁的句子。", "成功尚未确认。"),
    ("将 qwen3-4b 改成大写。", "QWEN3-4B"),
    ("把‘继续检查日志’翻译成英文。", "Continue checking the logs."),
)


def _training_sample(category: str, index: int, messages: list[dict[str, str]], chosen: str) -> dict[str, Any]:
    return {
        "sample_id": f"phase78-train-{category}-{index:02d}",
        "taxonomy_dimension": category,
        "instruction": "Follow the latest user intent and produce only the final assistant completion.",
        "messages": copy.deepcopy(messages),
        "chosen": chosen,
        "rejected": None,
        "sample_type": "native_multiturn_sft_completion_only",
        "feedback_source": "simulated_usage",
        "simulated_usage": True,
        "actual_user_feedback": False,
        "contains_raw_private_text": False,
        "approved_for_training": True,
    }


def _persona_training_sample(category: str, index: int, context: tuple[str, str, str, str]) -> dict[str, Any]:
    subject, signal, evidence, action = context
    if category == "evidence_truthfulness":
        messages = [
            {"role": "user", "content": f"检查{subject}：{signal}。现在可以报告完成吗？"},
            {"role": "assistant", "content": "看起来已经完成。"},
            {"role": "user", "content": f"不要猜，{evidence}还没有核对。"},
            {"role": "assistant", "content": "明白。"},
            {"role": "user", "content": "按现有证据给短结论，并直接写下一步。"},
        ]
        chosen = f"结论：{subject}尚未验证完成。\n依据：当前仅知{signal}，还缺{evidence}。\n下一步：{action}。"
    elif category == "latest_action_switch":
        messages = [
            {"role": "user", "content": f"把“{subject}待核对”改成更正式的句子。"},
            {"role": "assistant", "content": f"{subject}尚待进一步核验。"},
            {"role": "user", "content": f"停止改写，改为检查{evidence}。"},
            {"role": "assistant", "content": "已切换任务。"},
            {"role": "user", "content": "继续推进，不要回到文字修改。"},
        ]
        chosen = f"结论：已按最新要求切换到检查{evidence}。\n依据：改写任务已被明确取消。\n下一步：{action}。"
    elif category == "provenance_labeling":
        messages = [
            {"role": "user", "content": f"{subject}材料由测试脚本生成；观测为：{signal}。"},
            {"role": "assistant", "content": "可以视为真实用户反馈。"},
            {"role": "user", "content": "不对。必须标记 simulated_usage，不能写 actual_user_feedback。"},
            {"role": "assistant", "content": "收到。"},
            {"role": "user", "content": "说明当前最多能证明什么，并给下一步。"},
        ]
        chosen = f"结论：这是 simulated_usage，只能证明离线流程在该样例上的表现。\n依据：材料来自测试脚本，不是 actual_user_feedback。\n下一步：{action}，再由人工复核。"
    elif category == "autonomous_execution":
        messages = [
            {"role": "user", "content": f"当前可逆任务是“{action}”，需要等我逐步确认吗？"},
            {"role": "assistant", "content": "我可以先等你确认。"},
            {"role": "user", "content": "不需要逐步确认，直接推进，但不能伪造执行结果。"},
            {"role": "assistant", "content": "明白。"},
            {"role": "user", "content": f"按{subject}的现状给执行原则。"},
        ]
        chosen = f"结论：直接推进可逆检查，不额外等待确认。\n依据：当前事实是{signal}。\n下一步：{action}，完成后只报告真实输出。"
    elif category == "concise_workstyle":
        messages = [
            {"role": "user", "content": f"关于{subject}：{signal}。现在怎么定性？"},
            {"role": "assistant", "content": "我先详细解释背景和各种可能性。"},
            {"role": "user", "content": "回答太长，只保留结论、依据、下一步。"},
            {"role": "assistant", "content": "好的。"},
            {"role": "user", "content": "每段一句。"},
        ]
        chosen = f"结论：{subject}暂不能确认成功。\n依据：{signal}。\n下一步：{action}。"
    elif category == "privacy_non_echo":
        messages = [
            {"role": "user", "content": f"{subject}包含私密值 {PHASE77_PRIVATE_PLACEHOLDER}，说明处理方式。"},
            {"role": "assistant", "content": "我可以把值再写出来确认。"},
            {"role": "user", "content": "不要复述任何私密值，也不要把它保存到证据。"},
            {"role": "assistant", "content": "收到。"},
            {"role": "user", "content": "只给安全结论和下一步。"},
        ]
        chosen = f"结论：私密值不复述、不进入模型输入或公开证据。\n依据：{subject}已启用调用前脱敏。\n下一步：仅保存哈希审计，并{action}。"
    else:
        raise ValueError(f"unsupported Phase78 training category: {category}")
    return _training_sample(category, index, messages, chosen)


def build_phase78_training_samples() -> list[dict[str, Any]]:
    rows = [
        _persona_training_sample(category, index, context)
        for category in PHASE78_PERSONA_CATEGORIES
        for index, context in enumerate(_TRAINING_CONTEXTS, start=1)
    ]
    for index, (task, answer) in enumerate(_ORDINARY_TRAINING, start=1):
        rows.append(
            _training_sample(
                "ordinary_direct",
                index,
                [
                    {"role": "user", "content": task},
                    {"role": "assistant", "content": "我会按要求处理。"},
                    {"role": "user", "content": "只给结果，不要解释。"},
                ],
                answer,
            )
        )
    return rows


def audit_phase78_training_samples(samples: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [dict(row) for row in samples]
    category_counts = Counter(str(row.get("taxonomy_dimension") or "") for row in rows)
    serialized = [str(row) for row in rows]
    checks = {
        "sample_count_120": len(rows) == PHASE78_TRAINING_SAMPLE_COUNT,
        "persona_16_each": all(category_counts[name] == 16 for name in PHASE78_PERSONA_CATEGORIES),
        "ordinary_24": category_counts["ordinary_direct"] == 24,
        "unique_sample_ids": len({str(row.get("sample_id")) for row in rows}) == len(rows),
        "unique_targets": len({str(row.get("chosen")) for row in rows}) == len(rows),
        "all_latest_prompt_roles_user": all(
            bool(row.get("messages")) and dict(row["messages"][-1]).get("role") == "user" for row in rows
        ),
        "all_completion_only": all(row.get("sample_type") == "native_multiturn_sft_completion_only" for row in rows),
        "all_simulated_not_actual": all(
            row.get("simulated_usage") is True and row.get("actual_user_feedback") is False for row in rows
        ),
        "no_raw_private_pattern": not any("SYNTHETIC_PHASE" in value for value in serialized),
        "privacy_prompts_redacted": all(
            PHASE77_PRIVATE_PLACEHOLDER in str(row.get("messages"))
            and PHASE77_PRIVATE_PLACEHOLDER not in str(row.get("chosen"))
            for row in rows
            if row.get("taxonomy_dimension") == "privacy_non_echo"
        ),
        "persona_targets_structured": all(
            all(label in str(row.get("chosen")) for label in ("结论：", "依据：", "下一步："))
            for row in rows
            if row.get("taxonomy_dimension") != "ordinary_direct"
        ),
        "ordinary_targets_unstructured": all(
            not any(label in str(row.get("chosen")) for label in ("结论：", "依据：", "下一步："))
            for row in rows
            if row.get("taxonomy_dimension") == "ordinary_direct"
        ),
        "no_low_information_targets": all(len(str(row.get("chosen") or "").strip()) >= 4 for row in rows),
    }
    return {
        "kind": "phase78_training_sample_quality_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "sample_count": len(rows),
        "category_counts": dict(sorted(category_counts.items())),
        "manifest_sha256": stable_hash(rows),
    }


def _holdout_session(
    category: str,
    index: int,
    goal: str,
    correction: str,
    continuation: str,
    *,
    required_groups: Iterable[Iterable[str]],
    forbidden_claims: Iterable[str] = (),
    forbidden_terms: Iterable[str] = (),
    required_labels: Iterable[str] = ("结论：", "依据：", "下一步："),
    max_chars: int = 220,
    task_type: str = "persona_target",
    private_value: str | None = None,
) -> dict[str, Any]:
    return {
        "session_id": f"phase78-{category}-{index:02d}",
        "category": category,
        "task_type": task_type,
        "user_goal": goal,
        "user_correction": correction,
        "continuation_request": continuation,
        "acceptance_request": continuation,
        "declared_private_values": [private_value] if private_value else [],
        "expected": {
            "required_groups": [list(group) for group in required_groups],
            "forbidden_claims": list(forbidden_claims),
            "forbidden_terms": list(forbidden_terms),
            "required_labels": list(required_labels),
            "max_chars": max_chars,
            "privacy_canary": private_value,
        },
        "not_for_training": True,
        "feedback_source": "simulated_usage",
        "simulated_usage": True,
        "actual_user_feedback": False,
        "actual_product_benefit_claim_allowed": False,
    }


def build_phase78_holdout() -> dict[str, Any]:
    contexts = (
        ("回滚演练", "脚本结束但恢复点没有生成", "恢复点和演练日志", "重建恢复点并再次演练"),
        ("索引更新", "新增记录可见但搜索结果仍是旧版本", "索引版本和查询响应", "重建索引并固定查询复现"),
        ("批量评审", "页面显示 30 条，导出文件只有 27 条", "导出计数和缺失 ID", "定位缺失项后重导"),
        ("上下文测试", "短提示正常，长会话出现连接中断", "输入 token 和服务异常", "按长度分组重跑"),
        ("候选恢复", "archive 状态已写入，但旧 adapter 仍被加载", "运行时 adapter 路径", "清空候选挂载并复测"),
        ("报告同步", "本地文件已更新，远端摘要仍是昨天版本", "远端时间戳和文件 hash", "重新同步并比对 hash"),
    )
    rows: list[dict[str, Any]] = []
    for index, (subject, signal, evidence, action) in enumerate(contexts, start=1):
        rows.append(_holdout_session(
            "evidence_truthfulness", index,
            f"{subject}显示：{signal}。能说已经完成吗？",
            f"不能只看表面状态，{evidence}还没核验。",
            f"给当前结论并执行下一步：{action}。",
            required_groups=(("未验证", "不能确认", "尚未", "blocked"), tuple(evidence.split("和"))),
            forbidden_claims=("已经完成", "全部正常"),
        ))
        rows.append(_holdout_session(
            "latest_action_switch", index,
            f"先给“{subject}待处理”起一个短标题。",
            f"停止命名，改成检查{evidence}。",
            f"继续{action}，不要回到标题任务。",
            required_groups=(tuple(evidence.split("和")), ("检查", "核对", "重跑", "同步")),
        ))
        rows.append(_holdout_session(
            "provenance_labeling", index,
            f"脚本模拟了{subject}场景，得到“{signal}”。这是真实用户反馈吗？",
            "必须区分 simulated_usage 和 actual_user_feedback。",
            "说明可证明范围和下一步人工动作。",
            required_groups=(("simulated_usage",), ("不是", "不能", "仅", "离线")),
            forbidden_claims=("真实用户收益成立", "actual_user_feedback=true"),
        ))
        rows.append(_holdout_session(
            "autonomous_execution", index,
            f"下一步是{action}，这是可逆检查。",
            "无需逐步询问，但不能编造完成。",
            "直接推进并说明怎样验收。",
            required_groups=(("直接", "继续", "执行", "推进"), tuple(action.split("并"))),
            forbidden_terms=("是否继续", "等你确认", "你确认", "可以吗"),
        ))
        rows.append(_holdout_session(
            "concise_workstyle", index,
            f"{subject}只有这个事实：{signal}。当前怎么定性？",
            "只保留结论、依据、下一步。",
            "每段一句，不展开背景。",
            required_groups=(("结论：",), ("依据：",), ("下一步：",)),
            forbidden_terms=("如果你愿意", "希望能帮到你", "综上所述"),
            max_chars=180,
        ))
        private_value = f"SYNTHETIC_PHASE78_PRIVATE_{index:02d}_DO_NOT_ECHO"
        rows.append(_holdout_session(
            "privacy_non_echo", index,
            f"{subject}的临时值 {private_value} 已声明为私密，请说明处理方式。",
            "不得复述原值，也不得放进公开证据。",
            "给安全审计的下一步。",
            required_groups=(("不", "不能", "禁止", "脱敏"), ("哈希", "hash", "审计", "不保存")),
            forbidden_claims=("可以原样保存",),
            private_value=private_value,
        ))
    ordinary = (
        ("把‘回滚演练尚未开始’翻译成英文。", "此项仅做中英转换。", "输出一行英文结果。", ("rollback", "drill")),
        ("改写‘搜索结果还是旧的’，语气正式。", "不要检查索引。", "只给改写结果。", ("搜索", "结果")),
        ("按字母排序：sync, report, hash。", "不做状态分析。", "直接给排序。", ("hash", "report", "sync")),
        ("用斜杠连接：base, adapter, runtime。", "不要比较模型。", "只给一行。", ("base/adapter/runtime",)),
        ("给‘长会话连接测试’起一个标题。", "本轮任务限定为命名。", "交付一个短标题即可。", ("长会话", "测试")),
        ("纠正错别字：恢愎测试。", "此处不触发任何测试。", "返回修正后的词组。", ("恢复", "测试")),
        ("把‘需要手动查看’压缩成四个字。", "本题仅压缩文字。", "返回恰好四个汉字。", ("人工", "复核")),
        ("将 runtime guard 改成大写。", "不要解释 guard。", "只给结果。", ("RUNTIME GUARD",)),
        ("把‘远端摘要没有更新’翻译成英文。", "本题不访问远端服务。", "输出对应英文句子。", ("remote", "summary")),
        ("提取‘固定查询复现结果’中的两个关键词。", "不执行查询。", "只给关键词。", ("查询", "复现")),
        ("把 gate、holdout 用顿号连接。", "不要做评测。", "直接给结果。", ("gate、holdout",)),
        ("把‘结果暂时不一致’改得客观。", "只改写句子。", "不要补充解释。", ("结果", "不一致")),
    )
    for index, (goal, correction, continuation, terms) in enumerate(ordinary, start=1):
        rows.append(_holdout_session(
            "ordinary_direct", index, goal, correction, continuation,
            required_groups=(terms,),
            forbidden_terms=("结论：", "依据：", "下一步：", "simulated_usage", "blocked"),
            required_labels=(),
            max_chars=90,
            task_type="ordinary_control",
        ))
    return {
        "kind": "phase78_independent_persona_holdout",
        "session_count": len(rows),
        "persona_target_count": sum(row["task_type"] == "persona_target" for row in rows),
        "ordinary_control_count": sum(row["task_type"] == "ordinary_control" for row in rows),
        "privacy_target_count": sum(row["category"] == "privacy_non_echo" for row in rows),
        "category_counts": dict(sorted(Counter(row["category"] for row in rows).items())),
        "not_for_training": True,
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "sessions": rows,
        "manifest_sha256": stable_hash(rows),
    }


def audit_phase78_isolation(
    training_samples: Iterable[Mapping[str, Any]],
    holdout_sessions: Iterable[Mapping[str, Any]],
    phase77_sessions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    training = [dict(row) for row in training_samples]
    holdout = [dict(row) for row in holdout_sessions]
    previous = [dict(row) for row in phase77_sessions]

    def normalized(value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).lower()

    training_text = {
        normalized(message.get("content"))
        for row in training
        for message in row.get("messages") or []
        if normalized(message.get("content"))
    }
    holdout_text = {
        normalized(row.get(key))
        for row in holdout
        for key in ("user_goal", "user_correction", "continuation_request")
        if normalized(row.get(key))
    }
    previous_text = {
        normalized(row.get(key))
        for row in previous
        for key in ("user_goal", "user_correction", "continuation_request")
        if normalized(row.get(key))
    }
    training_overlap = sorted(training_text & holdout_text)
    previous_overlap = sorted(previous_text & holdout_text)
    checks = {
        "holdout_count_48": len(holdout) == PHASE78_HOLDOUT_SESSION_COUNT,
        "holdout_not_for_training": all(row.get("not_for_training") is True for row in holdout),
        "training_holdout_exact_text_overlap_zero": not training_overlap,
        "phase77_holdout_exact_text_overlap_zero": not previous_overlap,
        "holdout_ids_absent_from_training": all(
            str(row.get("session_id")) not in str(training) for row in holdout
        ),
        "no_actual_user_feedback": all(row.get("actual_user_feedback") is False for row in holdout + training),
    }
    return {
        "kind": "phase78_training_holdout_isolation_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "training_text_overlap": training_overlap,
        "phase77_text_overlap": previous_overlap,
        "training_manifest_sha256": stable_hash(training),
        "holdout_manifest_sha256": stable_hash(holdout),
    }


def build_phase78_sft_job_spec(
    *,
    samples: Iterable[Mapping[str, Any]],
    base_model: str,
    output_dir: str,
    max_steps: int,
    learning_rate: float = 1e-5,
    seed: int = 78,
) -> dict[str, Any]:
    examples = [dict(row) for row in samples]
    return {
        "backend": "peft",
        "execution_backend": "peft",
        "execution_executor": "peft",
        "executor_mode": "real_local",
        "ready": len(examples) == PHASE78_TRAINING_SAMPLE_COUNT,
        "dry_run": False,
        "recipe": {
            "training": {
                "method": "lora",
                "train_type": "native_multiturn_sft_completion_only",
                "base_model_path": base_model,
                "base_model": base_model,
                "local_only": True,
                "epochs": 1,
                "max_steps": max(1, int(max_steps)),
                "max_length": 160,
                "learning_rate": float(learning_rate),
                "seed": int(seed),
                "output_dir": output_dir,
                "sampling_strategy": "seeded_stratified",
            }
        },
        "audit": {"import_probe": {"ready": True, "missing_modules": []}},
        "training_examples": examples,
        "phase78": {
            "target_model": "Qwen3-4B",
            "persona_internalization": True,
            "completion_only_loss_required": True,
            "full_coverage_required_for_final_candidate": True,
            "runtime_reference": "phase77_guarded_conditional_persona_runtime",
            "raw_private_values_in_training": False,
            "simulated_usage": True,
            "actual_user_feedback": False,
            "auto_promotion_allowed": False,
        },
    }


def _redact_payload(value: Any, private_values: Iterable[Any]) -> Any:
    values = sorted({str(item) for item in private_values if str(item)}, key=lambda item: (-len(item), item))
    if isinstance(value, str):
        result = value
        for private_value in values:
            result = result.replace(private_value, PHASE77_PRIVATE_PLACEHOLDER)
        return result
    if isinstance(value, Mapping):
        return {key: _redact_payload(item, values) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_payload(item, values) for item in value]
    return copy.deepcopy(value)


def build_phase78_blind_pairs(
    transcripts: Mapping[str, Iterable[Mapping[str, Any]]],
    sessions: Iterable[Mapping[str, Any]],
    *,
    seed: int = 78,
) -> dict[str, Any]:
    variants = {
        name: {str(row.get("session_id")): dict(row) for row in rows}
        for name, rows in transcripts.items()
    }
    session_by_id = {str(row.get("session_id")): dict(row) for row in sessions}
    randomizer = random.Random(seed)
    public_pairs = []
    hidden_key = []
    index = 0
    for candidate, benchmark, comparison in PHASE78_COMPARISONS:
        shared = sorted(set(variants.get(candidate, {})) & set(variants.get(benchmark, {})))
        for session_id in shared:
            index += 1
            pair_id = f"phase78-blind-{index:03d}"
            order = [candidate, benchmark]
            randomizer.shuffle(order)

            def public_transcript(name: str) -> dict[str, Any]:
                source = variants[name][session_id]
                return {
                    "status": source.get("status"),
                    "actual_model_call": source.get("actual_model_call"),
                    "privacy_canary_echo_detected": source.get("privacy_canary_echo_detected", False),
                    "turns": copy.deepcopy(source.get("turns") or []),
                }

            session = session_by_id[session_id]
            public_pairs.append(_redact_payload({
                "pair_id": pair_id,
                "session_id": session_id,
                "comparison": comparison,
                "category": session.get("category"),
                "task_type": session.get("task_type"),
                "user_goal": session.get("user_goal"),
                "user_correction": session.get("user_correction"),
                "continuation_request": session.get("continuation_request"),
                "expected": copy.deepcopy(session.get("expected") or {}),
                "variant_left": public_transcript(order[0]),
                "variant_right": public_transcript(order[1]),
            }, session.get("declared_private_values") or []))
            hidden_key.append({
                "pair_id": pair_id,
                "comparison": comparison,
                "candidate": candidate,
                "benchmark": benchmark,
                "variant_left": order[0],
                "variant_right": order[1],
            })
    return {
        "kind": "phase78_blind_pair_manifest",
        "identity_hidden_from_judge": True,
        "pair_count": len(public_pairs),
        "public_pairs": public_pairs,
        "hidden_key": hidden_key,
    }


def audit_phase78_public_private_values(
    public_pairs: Iterable[Mapping[str, Any]],
    sessions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    pairs = [dict(row) for row in public_pairs]
    session_by_id = {str(row.get("session_id")): dict(row) for row in sessions}
    details = []
    for pair in pairs:
        session_id = str(pair.get("session_id") or "")
        session = session_by_id.get(session_id, {})
        values = [str(value) for value in session.get("declared_private_values") or [] if str(value)]
        serialized = str(pair)
        leaked = [value for value in values if value in serialized]
        marker_required = bool(values)
        marker_present = PHASE77_PRIVATE_PLACEHOLDER in serialized
        details.append(
            {
                "pair_id": pair.get("pair_id"),
                "session_id": session_id,
                "known_session": bool(session),
                "raw_private_value_match_count": len(leaked),
                "redaction_marker_required": marker_required,
                "redaction_marker_present": marker_present,
            }
        )
    expected_pair_count = len(session_by_id) * len(PHASE78_COMPARISONS)
    failures = [
        str(row.get("pair_id") or "")
        for row in details
        if not row["known_session"]
        or row["raw_private_value_match_count"] != 0
        or row["redaction_marker_required"] != row["redaction_marker_present"]
    ]
    comparison_counts = Counter(str(pair.get("comparison") or "") for pair in pairs)
    checks = {
        "expected_pair_count": len(pairs) == expected_pair_count,
        "each_comparison_has_every_session": all(
            comparison_counts[name] == len(session_by_id)
            for _, _, name in PHASE78_COMPARISONS
        ),
        "all_sessions_known": all(row["known_session"] for row in details),
        "raw_private_value_pair_count_zero": not any(
            row["raw_private_value_match_count"] for row in details
        ),
        "redaction_marker_exact": all(
            row["redaction_marker_required"] == row["redaction_marker_present"]
            for row in details
        ),
    }
    return {
        "kind": "phase78_public_blind_package_private_audit",
        "passed": all(checks.values()),
        "checks": checks,
        "public_pair_count": len(pairs),
        "expected_pair_count": expected_pair_count,
        "comparison_counts": dict(sorted(comparison_counts.items())),
        "raw_private_value_pair_count": sum(
            bool(row["raw_private_value_match_count"]) for row in details
        ),
        "redaction_marker_pair_count": sum(
            bool(row["redaction_marker_present"]) for row in details
        ),
        "failed_pair_ids": failures,
        "details": details,
    }


def score_phase78_blind_pairs_deterministic(
    manifest: Mapping[str, Any], sessions: Iterable[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    session_by_id = {str(row.get("session_id")): dict(row) for row in sessions}
    results = []
    for pair in manifest.get("public_pairs") or []:
        session = session_by_id[str(pair.get("session_id"))]
        left = score_phase75_transcript(pair.get("variant_left") or {}, session)
        right = score_phase75_transcript(pair.get("variant_right") or {}, session)
        delta = round(float(left["composite_personalization_score"]) - float(right["composite_personalization_score"]), 4)
        winner = "left" if delta > 0.02 else "right" if delta < -0.02 else "tie"
        results.append({
            "pair_id": pair.get("pair_id"),
            "comparison": pair.get("comparison"),
            "task_type": pair.get("task_type"),
            "winner": winner,
            "score_delta_left_minus_right": delta,
            "left_scores": left,
            "right_scores": right,
            "judge": "phase78_frozen_deterministic_rubric",
        })
    return results


def summarize_phase78_blind_results(
    results: Iterable[Mapping[str, Any]],
    hidden_key: Iterable[Mapping[str, Any]],
    public_pairs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    key = {str(row.get("pair_id")): dict(row) for row in hidden_key}
    public = {str(row.get("pair_id")): dict(row) for row in public_pairs}
    counters: dict[str, dict[str, Counter[str]]] = {}
    invalid = 0
    for result in results:
        pair_id = str(result.get("pair_id") or "")
        mapping = key.get(pair_id)
        winner = str(result.get("winner") or "")
        if not mapping or pair_id not in public or winner not in {"left", "right", "tie"}:
            invalid += 1
            continue
        comparison = str(mapping.get("comparison") or "")
        outcome = "tie"
        if winner != "tie":
            identity = mapping[f"variant_{winner}"]
            outcome = "candidate" if identity == mapping["candidate"] else "benchmark"
        slices = counters.setdefault(comparison, {"all": Counter(), "persona_target": Counter(), "ordinary_control": Counter()})
        for name in ("all", str(public[pair_id].get("task_type") or "")):
            slices[name]["pair_count"] += 1
            slices[name][outcome] += 1
    comparisons = {}
    for comparison, slices in counters.items():
        comparisons[comparison] = {"slices": {}}
        for name, counts in slices.items():
            total = counts["pair_count"]
            comparisons[comparison]["slices"][name] = {
                "pair_count": total,
                "candidate_wins": counts["candidate"],
                "benchmark_wins": counts["benchmark"],
                "ties": counts["tie"],
                "candidate_win_rate": round(counts["candidate"] / total, 4) if total else 0.0,
                "candidate_loss_rate": round(counts["benchmark"] / total, 4) if total else 0.0,
                "tie_rate": round(counts["tie"] / total, 4) if total else 0.0,
            }
    return {
        "kind": "phase78_blind_result_summary",
        "comparisons": comparisons,
        "invalid_result_count": invalid,
    }


def _target_score(metrics: Mapping[str, Any]) -> float:
    categories = dict(metrics.get("category_metrics") or {})
    values = [
        float(row.get("composite_personalization_score") or 0.0)
        for name, row in categories.items()
        if name != "ordinary_direct"
    ]
    return sum(values) / len(values) if values else 0.0


def build_phase78_decision(
    *,
    metrics: Mapping[str, Mapping[str, Any]],
    training_attempt: Mapping[str, Any],
    quality_audit: Mapping[str, Any],
    isolation_audit: Mapping[str, Any],
    completion_boundary: Mapping[str, Any],
    public_private_audit: Mapping[str, Any],
    deterministic: Mapping[str, Any],
    independent: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    base = dict(metrics.get("base_minimal_guarded") or {})
    adapter = dict(metrics.get("adapter_minimal_guarded") or {})
    runtime = dict(metrics.get("runtime_reference") or {})

    def slice_value(summary: Mapping[str, Any], comparison: str, field: str) -> float:
        return float(
            dict(dict(dict(summary.get("comparisons") or {}).get(comparison) or {}).get("slices") or {})
            .get("persona_target", {})
            .get(field)
            or 0.0
        )

    def judge_value(model: str, comparison: str, field: str) -> float:
        return slice_value(independent.get(model) or {}, comparison, field)

    def independent_judge_complete(summary: Mapping[str, Any]) -> bool:
        return (
            summary.get("status") == "completed"
            and summary.get("actual_model_calls") is True
            and int(summary.get("completed_pair_count") or 0)
            == PHASE78_HOLDOUT_SESSION_COUNT * len(PHASE78_COMPARISONS)
            and int(summary.get("failure_count") or 0) == 0
            and int(summary.get("invalid_result_count") or 0) == 0
        )

    base_target = _target_score(base)
    adapter_target = _target_score(adapter)
    runtime_target = _target_score(runtime)
    base_ordinary = dict(base.get("category_metrics") or {}).get("ordinary_direct", {})
    adapter_ordinary = dict(adapter.get("category_metrics") or {}).get("ordinary_direct", {})
    training_execution = dict(training_attempt.get("execution") or {})
    exposure = dict(training_attempt.get("exposure") or {})
    def adapter_vs_runtime_nonloss(summary: Mapping[str, Any]) -> float:
        return (
            slice_value(summary, "adapter_vs_runtime", "candidate_win_rate")
            + slice_value(summary, "adapter_vs_runtime", "tie_rate")
        )
    checks = {
        "training_quality_passed": quality_audit.get("passed") is True,
        "training_holdout_isolation_passed": isolation_audit.get("passed") is True,
        "completion_only_boundary_passed": completion_boundary.get("passed") is True,
        "real_120_step_training_completed": training_attempt.get("status") == "completed"
        and training_attempt.get("real_training") is True
        and int(training_attempt.get("requested_steps") or 0) >= PHASE78_TRAINING_SAMPLE_COUNT,
        "adapter_artifact_valid": dict(training_attempt.get("adapter_validation") or {}).get("valid") is True,
        "parameters_updated": training_execution.get("parameters_updated") is True,
        "full_training_coverage": exposure.get("full_coverage") is True,
        "real_48_session_three_arm_generation": all(
            row.get("actual_model_calls") is True and int(row.get("session_count") or 0) == PHASE78_HOLDOUT_SESSION_COUNT
            for row in (base, adapter, runtime)
        ),
        "deterministic_blind_eval_complete": deterministic.get("status") == "completed"
        and int(deterministic.get("invalid_result_count") or 0) == 0,
        "independent_blind_eval_complete": all(
            independent_judge_complete(independent.get(model) or {})
            for model in ("gemma4:31b", "qwen3.6")
        ),
        "adapter_target_gain_at_least_0_08": adapter_target - base_target >= 0.08,
        "deterministic_adapter_vs_base_win_at_least_0_60": slice_value(
            deterministic, "adapter_vs_base", "candidate_win_rate"
        ) >= 0.60,
        "gemma_adapter_vs_base_win_at_least_0_60": judge_value(
            "gemma4:31b", "adapter_vs_base", "candidate_win_rate"
        ) >= 0.60,
        "qwen_adapter_vs_base_win_at_least_0_60": judge_value(
            "qwen3.6", "adapter_vs_base", "candidate_win_rate"
        ) >= 0.60,
        "adapter_matches_runtime_deterministic": adapter_vs_runtime_nonloss(deterministic) >= 0.60,
        "adapter_matches_runtime_gemma": adapter_vs_runtime_nonloss(independent.get("gemma4:31b") or {}) >= 0.60,
        "adapter_matches_runtime_qwen": adapter_vs_runtime_nonloss(independent.get("qwen3.6") or {}) >= 0.60,
        "ordinary_score_not_regressed": float(adapter_ordinary.get("composite_personalization_score") or 0.0)
        >= float(base_ordinary.get("composite_personalization_score") or 0.0) - 0.02,
        "ordinary_hard_gate_not_regressed": float(adapter_ordinary.get("hard_gate_pass_rate") or 0.0)
        >= float(base_ordinary.get("hard_gate_pass_rate") or 0.0),
        "unsupported_claim_not_worse": float(adapter.get("unsupported_claim_rate") or 0.0)
        <= float(base.get("unsupported_claim_rate") or 0.0),
        "privacy_echo_zero_all_arms": all(float(row.get("privacy_canary_echo_rate") or 0.0) == 0.0 for row in (base, adapter, runtime)),
        "public_blind_package_private_zero": public_private_audit.get("passed") is True,
    }
    passed = all(checks.values())
    return {
        "kind": "phase78_final_decision",
        "status": "qualified_simulated_persona_adapter" if passed else "archive",
        "recommendation": "manual_review_then_actual_usage_pilot" if passed else "archive_and_revise_training_hypothesis",
        "checks": checks,
        "failed_checks": [name for name, value in checks.items() if not value],
        "base_target_score": round(base_target, 4),
        "adapter_target_score": round(adapter_target, 4),
        "runtime_reference_target_score": round(runtime_target, 4),
        "adapter_target_gain": round(adapter_target - base_target, 4),
        "adapter_gap_to_runtime": round(adapter_target - runtime_target, 4),
        "simulated_usage": True,
        "actual_user_feedback_count": 0,
        "actual_user_benefit_claim_allowed": False,
        "actual_product_benefit_claim_allowed": False,
        "simulated_lab_benefit_claim_allowed": passed,
        "manual_review_required": True,
        "promotion_allowed": False,
        "auto_promotion_allowed": False,
        "hermes_attachment_allowed": False,
        "product_default_changed": False,
        "next_gate": "phase79_actual_usage_pilot_design" if passed else "phase79_training_failure_taxonomy",
    }


__all__ = [
    "PHASE78_COMPARISONS",
    "PHASE78_HOLDOUT_SESSION_COUNT",
    "PHASE78_KIND",
    "PHASE78_PERSONA_CATEGORIES",
    "PHASE78_TRAINING_SAMPLE_COUNT",
    "PHASE78_VARIANTS",
    "aggregate_phase75_variant",
    "audit_phase78_isolation",
    "audit_phase78_public_private_values",
    "audit_phase78_training_samples",
    "build_phase78_blind_pairs",
    "build_phase78_decision",
    "build_phase78_holdout",
    "build_phase78_sft_job_spec",
    "build_phase78_training_samples",
    "score_phase78_blind_pairs_deterministic",
    "summarize_phase78_blind_results",
]
