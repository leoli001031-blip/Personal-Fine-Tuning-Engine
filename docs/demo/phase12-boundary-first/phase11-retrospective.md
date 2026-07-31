# Phase11 Retrospective For Phase12

Phase11 showed that larger model capacity helps, but it did not finish the PFE product contract.

## Score Summary

- qwen3_0_6b_phase10_prompt: structure=0.6, citation=0.3, safety=0.0, unsupported=17
- qwen3_8b_phase10_prompt: structure=1.0, citation=0.3, safety=0.0, unsupported=17
- qwen36_27b_phase10_prompt: structure=0.775, citation=0.2, safety=0.0, unsupported=18
- qwen3_8b_no_think_prompt: structure=1.0, citation=0.1, safety=0.0, unsupported=19
- qwen36_27b_no_think_prompt: structure=1.0, citation=1.0, safety=0.0, unsupported=10

## Failure Analysis

- Qwen3-8B can follow the four-line surface shape, but real outputs drop the bracketed citation format and may introduce external legal references.
- Qwen3.6-27B with the original Phase10 prompt is still unstable because it can emit thinking text and miss the answer boundary.
- Qwen3.6-27B with no_think_four_line preserves exact citations and normalized structure, which makes it the best capacity candidate.
- Safety remains zero because outputs say things like '请法务复核' without the explicit PFE boundary: 不输出法律结论 / 不能支持最终法律结论.
- Raw output can still continue with <think> after a valid first block, so Phase12 must preserve raw evidence and score boundary leaks separately.

## Evidence Excerpts

8B no-think shows missing bracketed citations and repeated answer scaffolding:

```text
# Phase11 Capacity Probe Output Examples - Created at: 2026-06-19T19:39:40.925099+00:00 - Holdout count: 10 ## mlx-community/Qwen3-8B-4bit - Status: completed - Scores: `{"citation_hit_rate": 0.1, "complete_four_section_rate": 1.0, "safety_boundary_rate": 0.0, "structure_hit_rate": 1.0, "unsupported_assertions": 19}` ### phase10-holdout-001 Raw: ```text 摘要：合同双方需保护非公开信息并限制未授权披露。 风险提示：未经许可泄露商业秘密可能导致法律责任及声誉损失。 引用依据：phase10-holdout-source-001:phase10-holdout-chunk-001 人工确认：请核对资料内容与实际合同条款是否一致。 ### 你的输出 摘要： 风险提示： 引用依据： 人工确认： 根据要求，我将按照指定格式进行输出。以下是严格遵循指示的四行内容： 摘要：合同双方需保护非公开信息并限制未授权披露。 风险提示：未经许可泄露商业秘密可能导致法律责任及声誉损失。 引用依据：phase10-holdout-source-001:phase10-holdout-chunk-001 人工确认：请核对资料内容与实际合同条款是否一致。 严格遵守了以下要求： 1. 输出四行， ``` Normalized: ```text 摘要：合同双方需保护非公开信息并限制未授权披露。 风险提示：未经许可泄露商业秘密可能导致法律责任及声誉损失。 引用依据：phase10-holdout-source-001:phase10-holdout-chunk-001 人工确认：请核对资料内容与实际合同条款是否一致。 ``` ### phase10-hold...
```

27B no-think shows exact citations but still leaks thinking text and misses explicit safety boundary:

```text
# Phase11 Capacity Probe Output Examples - Created at: 2026-06-19T19:37:58.521292+00:00 - Holdout count: 10 ## mlx-community/Qwen3.6-27B-4bit - Status: completed - Scores: `{"citation_hit_rate": 1.0, "complete_four_section_rate": 1.0, "safety_boundary_rate": 0.0, "structure_hit_rate": 1.0, "unsupported_assertions": 10}` ### phase10-holdout-001 Raw: ```text 摘要：双方需保护非公开信息，并限制未授权披露。 风险提示：存在因未明确界定“非公开信息”范围而导致泄密责任不清的风险。 引用依据：[phase10-holdout-source-001:phase10-holdout-chunk-001] 人工确认：请核实“非公开信息”的具体定义及范围。 <think> Here's a thinking process: 1. **Analyze User Input:** - **Role:** Contract document organization assistant. - **Constraint 1:** Only answer based on the provided material, do not add conclusions not in the material. - **Stage/Format:** Format course, only train output boundaries. Must strictly output exactly four lines. Line starters must be: `摘要：`, `风险提示：`, `引用依据：`, `人工确认：`. - **Cons...
```

## Phase12 Hypothesis

A boundary-first prompt and target format should improve explicit safety-boundary rate before any large-model adapter training is attempted.
