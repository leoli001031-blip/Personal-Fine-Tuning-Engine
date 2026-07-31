# Phase99 Output Examples

All examples are real local Qwen3-4B outputs from simulated usage sessions. No post-hoc truncation was used.

## Exact Three-Line

```text
结论：登记日期仍未填写
依据：当前材料中登记日期字段保持空白
下一步：请补充完整
```

## False Block

```text
结论：完成
依据：最终手工复核确认回执编号和存档副本一致性
下一步：归档案卷
```

## Provenance Near-Miss

```text
结论：状态无法确认
依据：simulated_usage=true, actual_user_feedback=false
下一步：停止
```

The flags are correct, but the answer still does not explicitly state that the row cannot be counted as actual feedback. The frozen provenance scorer therefore remains at 0.0.

## Ordinary Control

```text
需简化核驗印簽臺賬待辦。
```

Unlike Phase96, no output contains `</think>`, fake `Human/AI` turns, tool-response continuation, or repeated answer blocks.
