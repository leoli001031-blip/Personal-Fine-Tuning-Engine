# Phase11 Capacity Probe Output Examples

- Created at: 2026-06-19T19:25:00.253836+00:00
- Holdout count: 2

## mlx-community/Qwen3-0.6B-4bit

- Status: completed
- Scores: `{"citation_hit_rate": 0.5, "complete_four_section_rate": 1.0, "safety_boundary_rate": 0.0, "structure_hit_rate": 1.0, "unsupported_assertions": 3}`

### phase10-holdout-001

Raw:

```text
摘要：[phase10-holdout-source-001:phase10-holdout-chunk-001]
风险提示：[phase10-holdout-chunk-002] 
引用依据：[phase10-holdout-source-003]
人工确认：[phase10-holdout-chunk-004]

### 格式要求
四行，每行只包含一个项。 


```

Normalized:

```text
摘要：[phase10-holdout-source-001:phase10-holdout-chunk-001]
风险提示：[phase10-holdout-chunk-002]
引用依据：[phase10-holdout-source-003]
人工确认：[phase10-holdout-chunk-004]
```

### phase10-holdout-002

Raw:

```text
摘要：[phase10-holdout-chunk-002]
风险提示：[phase10-holdout-chunk-003]
引用依据：[phase10-holdout-source-002]

人工确认：
摘要：[phase10-holdout-chunk-002]
风险提示：[phase10-holdout-chunk-003]
引用依据：[phase10-holdout-source
```

Normalized:

```text
摘要：[phase10-holdout-chunk-002]
风险提示：[phase10-holdout-chunk-003]
引用依据：[phase10-holdout-source-002]
人工确认：
```
