# PFE Studio v0 plan

更新时间：2026-06-15

## 1. 目标判断

PFE Studio v0 的目标不是把 CLI 搬进网页，而是把普通用户路径从 CLI-first 收敛成一个极简本地模型工作台。

用户只需要理解三件事：

```text
模型 / 接入 / 版本
```

底层仍然保留：

```text
collect -> curate -> train -> eval -> promote -> serve
```

但这些内部流程默认折叠为状态豆、工作单和证据条，不要求用户理解 `adapter`、`daemon`、`queue`、`PEFT`、`promote` 等内部词。

## 2. 产品边界

### 普通用户可见

- 选择基础模型。
- 确认当前是否使用真实本地模型。
- 复制网页地址。
- 复制 OpenAI-compatible API 地址。
- 查看当前模型版本。
- 查看待确认版本。
- 回退到上一版本。
- 查看少量证据：健康检查、最近请求、版本来源。

### 默认隐藏

- CLI 命令。
- 原始日志。
- daemon / queue / trigger / worker runner。
- PEFT / MLX / llama.cpp / backend 细项。
- epochs、LoRA rank、DPO、quantization。
- release evidence、benchmark、soak、CI。

### 需要明确确认

- 切换当前版本。
- 回退版本。
- 启动真实训练。
- 开启远程访问。
- 配置 API key。
- 启用云端 Teacher / Judge / Router。
- 任何可能让数据出本机的动作。

## 3. 信息架构

第一版只保留两个页面：

```text
使用
版本
```

后续再增加：

```text
设置
```

### 使用

回答四个问题：

- 现在用哪个模型？
- 回复来自真实本地模型还是模板回复？
- 网页地址是什么？
- API 地址是什么？

首屏结构：

```text
zc + PFE / 本地模型工作台        可继续

当前模型
[模型选择器]  当前版本：20260615-001 / 使用中

网页和 API 已可用，回复来自真实本地模型。

[打开网页]

网页地址  http://127.0.0.1:8921/
API 地址   http://127.0.0.1:8921/v1/chat/completions

证据
本机服务 / 通过 / 05:05 / 查看
模型版本 / 已验证 / 05:03 / 查看
最近请求 / 真实本地 / 05:04 / 查看
```

### 版本

回答三个问题：

- 当前正在服务哪一版？
- 有没有待确认版本？
- 能不能回退？

版本列表使用用户语言：

```text
使用中
待确认
可回退
已归档
```

不要直接显示 `promoted`、`pending_eval`、`adapter_manifest` 作为主文案。

## 4. 状态词典

界面固定使用短状态豆：

```text
检查中
可继续
需确认
已完成
有问题
可重试
等待中
已保存
```

内部词翻译：

```text
adapter      -> 模型版本
candidate    -> 待确认版本
promote      -> 设为当前版本
archive      -> 归档版本
daemon       -> 后台处理
queue        -> 待处理更新
mock_local   -> 模板回复
real-local   -> 真实本地模型
PEFT         -> 本地微调方式
```

## 5. 后端 API 规划

### v0: 工作台首屏

先补最小 API，让前端能安全显示当前状态，并保存基础模型选择：

```text
GET /studio
GET /pfe/studio
GET /pfe/runtime
GET /pfe/models
GET /pfe/adapters
PUT /pfe/config/model
PUT /pfe/config/real-local
GET /pfe/workspaces
```

这些接口不启动训练、不下载模型、不重载运行时；`PUT /pfe/config/model` 只保存基础模型配置，并支持 `validate_only=true`。
`PUT /pfe/config/real-local` 只切换当前服务进程里的真实本地推理开关，不写永久环境变量。

### v0.1: 配置与 readiness

```text
GET /pfe/readiness
POST /pfe/workspaces
```

`readiness` 是 `doctor` 和 `next` 的 JSON 化结果。模型切换后的运行时热重载、真实模型可用性检查和环境检查放在这一阶段。

当前已落地 `GET /pfe/readiness` 的第一版只读实现：返回本机服务、基础模型源、真实本地模型开关、推理依赖、当前模型版本和下一步动作。它不下载模型、不加载大模型、不启动训练。
当前已落地 `GET /pfe/workspaces` 和 `POST /pfe/workspaces` 的第一版：列出当前服务进程可见的工作区，创建一个合法 slug 工作区，并切换当前服务进程的 workspace。它会同步 `PFE_WORKSPACE` 给当前进程里的 API 调用，但不写永久 shell 环境。

当前也已落地 Studio 主路径 smoke：`tools/studio_model_path_smoke.py` 会打开 Studio HTML 合同、创建并切换工作区、保存一个本地模型路径、开启当前进程的真实本地模型尝试、验证 readiness 识别该路径，并调用 `/v1/chat/completions` 证明 `model=local` 的下一次 API 请求会使用保存后的基础模型配置。`GET /pfe/runtime` 同时返回结构化 `api` 合同，包括 OpenAI-compatible chat endpoint、`model=local` 参数和最小请求体；Studio 地址区会展示网页地址、API 地址、模型参数和可复制的 `curl` 调用示例。该 smoke 已挂入 `make smoke-beta`。
当前服务根路径 `/` 也已切到 Studio，`/studio` 和 `/pfe/studio` 保持为同一主入口别名；旧 chat 页面不再作为默认用户入口。
Studio 主界面已开始把 readiness/preflight 的内部 blocker code 翻译成用户可读文案，例如“还没选择本地模型路径”“还没开启真实本地模型”“缺少本地推理依赖”；API 仍保留原始 code 供测试和诊断使用。
当前也已开始 v0.3 前端收口：首屏从工程式摘要改为“当前工作单”，直接展示模型、回复模式、版本和 API 四个事实；主动作收敛为“复制 API / 测试接入 / 打开网页 / 使用本地模型回复”；模型路径旁新增短状态豆，接入区优先展示聊天 API、反馈 API 和可复制接入信息；调用示例默认折叠；版本空态使用“还没有模型版本”这类用户语言。浏览器验证覆盖 390px 移动宽度无横向溢出，以及 Studio 点击流仍能保存模型路径、开启本地回复并得到“可生成版本”。

技术债治理也已开始落地：Studio runtime/API handoff 合同已抽到 `pfe_server.studio_contracts`；workspaces/models 的发现、校验和 payload 组装已抽到 `pfe_server.studio_resources`；training job 的 URL、事件、列表 payload 合同已抽到 `pfe_server.studio_jobs`；training job 的 JSON 读写、内存态合并、active job 查询、取消和 retry 事件持久化已抽到 `pfe_server.studio_job_store`；eval job 的 running/completed/failed/status payload 合同已抽到 `pfe_server.studio_eval_jobs`；训练 job 的后台启动、执行、失败记录和 overall state 持久化编排已抽到 `pfe_server.studio_training_service`；评估 job 的 running state 创建、后台 evaluate、eval report 读取和 completed/failed state 持久化编排已抽到 `pfe_server.studio_eval_service`。`app.py` 继续负责路由接线、配置读写、当前进程副作用和 adapter/version 校验。后续应继续把配置副作用和路由注册重复问题拆出，并用当前 smoke 保护对外行为不漂。
评估启动的防重入已改为读取当前 eval state，包括磁盘上的 `eval_status.json`；这避免服务重启或内存态丢失后再次并发启动同一 workspace 的评估。

当前已落地训练入口的安全半步：`POST /pfe/training/jobs` 默认只返回训练预检和 `confirmation_required`，不会创建 job，也不会启动后台训练；只有显式传入 `confirm=true`，且预检没有阻塞项，才沿用现有训练路径创建任务。`GET /pfe/training/jobs` 返回当前工作区最近训练任务、最新任务、进行中任务和整体训练状态；单个 `GET /pfe/training/jobs/{id}` 返回稳定 `status_url`、`events_url`、`cancel_url` 和 `retry_url`；`GET /pfe/training/jobs/{id}/events` 返回 queued、started、completed/failed/cancelled/retry_requested 等 JSON 事件列表；`POST /pfe/training/jobs/{id}/cancel` 需要 `confirm=true`，queued 任务会被取消，running 任务只记录取消请求，不假装能强行中断底层训练，并且当前进程内的 running job 会保留同一个 job 对象，避免后台完成时覆盖掉取消请求事件；`POST /pfe/training/jobs/{id}/retry` 只允许 failed/cancelled 任务，复用原 `training_config`，仍然先走 preflight 和确认合同。Studio 右侧“版本生成”面板先调用预检路径，启动后显示最近任务状态和最新事件，活动任务可点击“停止生成”，失败或已取消任务可点击“重新生成”，`tools/studio_model_path_smoke.py` 也会验证预检不会裸触发训练。
当前同一 workspace 默认只允许一个 queued/running training job。若已有活动任务，新的确认启动或 retry 会返回 `training_job_already_active`，由 Studio 继续展示当前活动任务而不是并发触发第二个训练。

### v0.2: 版本确认

```text
POST /pfe/adapters/{version}/promote
POST /pfe/adapters/{version}/rollback
POST /pfe/adapters/{version}/archive
```

这些动作必须要求确认，并返回当前版本、目标版本、评测状态和回退版本。

当前已落地第一版 HTTP action：`promote`、`rollback`、`archive` 均要求 `confirm=true`；`rollback` 可以显式恢复已归档历史版本为当前版本。`GET /pfe/adapters` 也会把原始 `metrics` / `eval_report` 收敛成用户可读的 `training_summary`、`eval_summary` 和 `decision`，让 Studio 版本列表显示训练样本、评估结论和建议动作。Studio 版本列表会在对应版本上显示 `设为当前`、`回退`、`归档` 操作。

当前也已落地评估触发的安全入口：`POST /pfe/eval` 需要 `confirm=true`，请求体传入 `version` 后会启动后台评估；`GET /pfe/eval/status` 返回当前评估状态、目标版本和刷新后的版本列表。Studio 版本列表会对可评估版本显示 `评估`，评估运行时显示 `评估中` 并轮询刷新，完成后沿用 `eval_summary` / `decision` 展示评估结论和建议动作。

### v1: 作业化训练

```text
POST /pfe/training/jobs
GET /pfe/training/jobs/{id}
GET /pfe/training/jobs/{id}/events
POST /pfe/training/jobs/{id}/cancel
POST /pfe/training/jobs/{id}/retry
POST /pfe/eval/jobs
GET /pfe/eval/jobs/{id}
GET/PUT /pfe/auto-train/config
```

真实训练、DPO、eval、export、daemon recovery 必须走 preflight、队列、锁、资源预算和状态机，不能从前端裸触发。

当前只完成了 `POST /pfe/training/jobs` 的 preflight/confirmation gate、`GET /pfe/training/jobs` 的列表化观察面、`GET /pfe/training/jobs/{id}/events` 的 JSON 事件列表、`POST /pfe/training/jobs/{id}/cancel` 的诚实取消合同、`POST /pfe/training/jobs/{id}/retry` 的失败/取消后重试合同，以及 `POST /pfe/eval` / `GET /pfe/eval/status` 的最小评估触发和观察合同；SSE 事件流、资源预算和完整作业状态机仍属于 v1。

## 6. 安全边界

PFE Studio 默认是 local-first 单用户工作台。

必须保留：

- 默认绑定 `127.0.0.1`。
- 明确显示访问范围：`仅本机` 或 `允许远程`。
- 管理接口默认只允许本机访问。
- 远程访问必须配置 API key。
- 云端能力默认关闭。
- 涉及云端 Teacher / Judge / Router 时必须显式确认，并记录 PII / 出境审计。

网页本身不能直接承担：

- 安装 Python / 训练依赖。
- 修改 `.venv`。
- 下载大模型。
- 扫描任意本地路径。
- 启动或停止自身所在的 `pfe-server` 进程。
- 打开 `0.0.0.0` 远程访问。

这些属于 privileged bootstrap，后续应由桌面 wrapper、installer 或本机 helper 处理。

## 7. 第一批落地切片

本轮先完成：

1. `PFE Studio v0` 规划文档。
2. `GET /pfe/runtime`：返回 host、port、base URL、API URL、workspace、auth mode、privacy mode，以及前端可直接展示的 OpenAI-compatible API 调用合同。
3. `GET /pfe/models`：返回当前基础模型和可见本地模型候选。
4. `PUT /pfe/config/model`：保存基础模型选择，支持 `validate_only=true`。
5. `PUT /pfe/config/real-local`：开启或关闭当前服务进程的真实本地模型尝试。
6. `GET/POST /pfe/workspaces`：列出、创建并切换当前服务进程工作区。
7. `GET /pfe/adapters`：返回结构化版本列表、当前使用中版本、待确认版本。
8. `POST /pfe/training/jobs`：默认只做训练预检，显式确认后才允许启动训练任务。
9. `GET /pfe/training/jobs`：返回当前工作区训练任务列表、最新任务、进行中任务和整体状态，供 Studio 显示“最近任务”。
10. `GET /pfe/training/jobs/{id}/events`：返回训练任务的 JSON 事件列表，供 Studio 显示最新过程事件。
11. `POST /pfe/training/jobs/{id}/cancel`：queued 任务可取消，running 任务只记录取消请求，不声明强中断。
12. `GET /studio` / `GET /pfe/studio`：返回 warm-workbench 风格极简页面。
13. 测试覆盖新 API、页面入口和最小调用示例。

不在本轮做：

- 运行时热重载模型。
- 启动真实训练。
- 下载模型。
- 远程访问配置。
- 训练参数 UI。
- 多用户 / 云端部署。

## 8. 成功标准

第一批完成后，打开本地服务应能做到：

```text
打开 /studio
看到当前模型
看到网页地址
看到 API 地址
看到当前版本
看到最近证据
复制地址
理解当前是否可继续
```

用户不需要知道 CLI，不需要知道 daemon/queue/adapter，不需要打开旧 dashboard 才能判断 PFE 是否可用。

## 9. 当前验证证据

截至 2026-06-15，Phase2 第一批闭环已经可以用本地轻量模型跑过完整 release strict gate：

```bash
PFE_REAL_LOCAL_MODEL="$PWD/models/Qwen2.5-0.5B-Instruct-4bit" make smoke-release-strict
```

该命令会先跑 `make smoke-beta`，再跑浏览器级 Studio 点击流和真实本地模型 happy path。当前已验证：

- Studio 浏览器 smoke 会打开 `/`，保存本地模型路径，开启真实本地模型，触发训练预检，并得到 `studio_training: 可生成版本`。
- 真实本地 happy path 会使用 `models/Qwen2.5-0.5B-Instruct-4bit` 完成 PEFT 真实路径，并在 manifest 中记录 `kind=real_peft | path=real_import`。
- 15GB 级别的大模型不是 release gate 的必要前置；优先使用这个 276MB 本地模型做闭环验证。
