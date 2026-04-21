# Phase 2 收尾说明

更新时间：2026-04-21

## 1. 结论

截至 **2026-04-21**，`Phase 2` 的已知主线问题已经基本收口，可以将当前状态视为：

- `Phase 2` 核心能力完成
- `Phase 2` 已知 blocker 清零
- 常规单元 / surface / integration / e2e 验证已补齐
- 当前更适合转入 `Phase 3` 或 release 收尾，而不是继续把 `Phase 2` 当成进行中阶段

这次收尾不是只基于静态 review，而是基于真实修复和多轮测试结果给出的判断。

## 2. 本轮收口的关键问题

### 2.1 状态与 latest 指针一致性

本轮修复了几类会直接影响状态面板、服务状态和 adapter 生命周期可见性的缺陷：

- `status` 快照在 latest manifest 不可读时可能抛出异常，而不是降级返回
- `ServerServices.status()` 在非默认 workspace 下读取了错误的 latest adapter
- `adapters/<workspace>/latest` 如果是普通文件而不是 symlink，会一路落到 `os.readlink()` 报错
- signal summary 中 `source_event_id_count / request_id_count / session_id_count` 之前错误地等于总 signal 数

这些问题会让闭环状态看起来“还能跑”，但实际 dashboard、status 和 promote 相关 surface 已经不可信；现在已经统一修正。

### 2.2 路径隔离与测试隔离

这轮也把 `PFE_HOME` 相关的隐式路径问题收紧了：

- `chat_collector`
- `pii_audit`
- 相关 path isolation 测试

目标是让默认持久化都落在可控的 `PFE_HOME` 下，而不是悄悄继承宿主机 home 目录。

### 2.3 训练后端 dispatch 稳定性

`trainer/service.py` 的 backend dispatch 已做收口，避免在 dispatch 阶段为了探测可选后端而硬导入 `mlx` / `mlx_lm` 这类可能在当前环境直接崩掉的依赖。

这类问题的危险点在于：用户只是想选择或探测后端，系统却在还没真正执行训练前就因为 import side effect 失败。现在 dispatch 路径已经更稳，也更符合“optional backend 是可选能力”的预期。

### 2.4 UTC 时间语义统一

最后一轮补掉了两类时间语义问题：

- naive timestamp 与 aware timestamp 混用时，冲突检测和后续分析可能崩溃
- `PII audit` 的文件轮转使用本地月，而报表读取按 UTC 月扫描，月切附近可能漏数

当前收口后的策略是：

- signal 相关时间统一归一到 UTC-aware
- `PII audit` 日志文件按 UTC 月切
- legacy naive 时间戳仍尽量兼容读取
- stale runner 清理也接入统一时间解析 helper

## 3. 验证结果

### 3.1 定向回归

围绕本轮修复点新增并通过了以下回归：

- naive signal timestamp 归一化
- mixed naive/aware timestamp 冲突检测不再崩溃
- `PII audit` UTC 月切文件轮转
- legacy naive heartbeat 时间戳仍可被 stale runner 清理逻辑处理

定向测试结果：

- `106 passed`（2026-04-20）

### 3.2 中风险大子集

在不跑 `integration` / `e2e` 的前提下，执行了更大的常规测试子集：

- `pytest tests -q -m "not integration and not e2e" --tb=short`

结果：

- `959 passed, 21 deselected`（2026-04-20）

这一步主要用于验证 Phase 2 主线修复没有污染常规 HTTP / CLI / pipeline / status surface。

### 3.3 显式 integration / e2e

显式标记的 integration / e2e 套件共有 21 条。

第一次在沙箱内执行时，出现了两类环境性失败：

- 本地监听 `127.0.0.1:9999` 被沙箱拒绝
- `trainer_dpo_real` 所需的外部模型资源访问受限

在沙箱外使用同一测试命令重跑后，结果为：

- `21 passed, 959 deselected, 16 warnings in 803.59s (0:13:23)`（2026-04-21）

这说明之前那批失败不是 Phase 2 代码回归，而是运行环境边界。

### 3.4 补充慢路径 surface

另外补跑了未标 `integration` 但属于慢速闭环 surface 的几组测试：

- `tests/test_server_serve_e2e.py`
- `tests/test_trainer_export_server_e2e.py`
- `tests/test_phase0_lifecycle_e2e.py`

结果：

- `5 passed in 28.32s`（2026-04-21）

## 4. 当前还剩什么

当前没有发现新的 `Phase 2` blocker，但仍有少量边界应区别对待：

- 依赖栈 warning 仍存在：`torch / peft / dataloader` 在真实训练测试里有非阻塞 warning
- 尚未做长时间 soak test：当前结论基于功能正确性和闭环可达性，不等于长时间稳定性验证
- 尚未做专门的性能/内存基准：本轮重点是 correctness 和 surface consistency，不是吞吐或峰值优化

这些点不影响“`Phase 2` 基本完成”的结论，但如果要进入对外发布或更重真实负载，建议单独作为下一阶段任务处理。

## 5. 建议的下一步

如果继续推进，优先级建议如下：

1. 进入 `Phase 3`：插件体系、多模型兼容、示例应用和文档站
2. 做 release 收尾：整理 changelog、对外使用说明、升级路径
3. 单独安排一轮长时稳定性 / 性能验证，而不是继续把功能性问题混在 `Phase 2`

## 6. 最终判断

截至 **2026-04-21**，可以把项目当前状态表述为：

> `Phase 1` 与 `Phase 2` 的主线目标已经完成，`Phase 2` 已知问题基本解决，闭环能力经过真实测试验证，当前进入收尾完成态。
