# PFE Phase2 网页证据地图

这份表用于把展示页每一段和真实证据对应起来。

| 网页段落 | 主要素材 | 支撑证据 | 可表达的结论 |
|---|---|---|---|
| 首屏产品形态 | `images/studio-workbench.png` | `../screenshots/10-warm-simple-effective-studio.png` | Studio 已收敛为简洁工作台，能看到模型、版本、API 和结果证据。 |
| 结果对比 | `images/base-local-proof.png` | `../evidence/base-response.json`, `../evidence/local-response.json`, `../evidence/base-vs-local-summary.json` | base 未命中目标记忆，local adapter 命中 `DEMO-PHASE2-042`。 |
| 训练完成 | `images/eval-gate.png` | `../evidence/demo-training.txt` | demo 彩排产生了 adapter `20260618-001`。 |
| 评估闸门 | `images/eval-gate.png` | `../evidence/demo-eval-summary.json` | eval 完成，`recommendation=deploy`，promotion gate 放行。 |
| 推广状态 | `images/promoted-adapter.png` | `../evidence/demo-promote-summary.json` | adapter 已推广为 current/latest，并且 `adapter_loaded=true`。 |
| API 接入 | `images/api-handoff.png` | `../evidence/base-response.json`, `../evidence/local-response.json` | Chat API 和 feedback API 可作为外部接入交接材料。 |
| 测试证明 | 可用证据条展示 | `../evidence/demo-phase2-smoke.txt`, `../evidence/smoke-memory-golden.txt` | smoke 和 memory golden 都有真实命令输出。 |

## 推荐页面顺序

```text
1. 首屏：产品形态 + 一句话结果
2. 结果：base/local 对比
3. 闭环：训练 -> eval gate -> promote -> API handoff
4. 证据：截图、日志、JSON response
```
