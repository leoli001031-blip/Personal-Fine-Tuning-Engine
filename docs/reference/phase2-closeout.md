# Phase 2 收尾说明

更新时间：2026-06-15

## 1. 结论

截至 **2026-06-15**，`Phase 2` 的功能性闭环已经打通，可以将当前状态视为：

- `Phase 2` 核心闭环完成
- 已知功能 blocker 清零
- CLI / queue / server / dashboard / chat console 的 beta smoke 已补齐
- 默认验证链保持无下载、无真实训练依赖
- 真实本地模型 full happy path、浏览器 strict smoke、30 分钟 soak 和 performance budget 已有本机证据

当前判断是：`Phase 2` 可以进入 closeout / release 收尾，不需要继续作为功能研发阶段推进。后续应把远端 CI release gate 的真实执行结果作为 release-ready 的最后关键证据。

## 2. 本轮收口的关键问题

### 2.1 First-run 与 guided next

新增了面向首次使用者的本地入口：

- `pfe init`
- `pfe next`
- `make smoke-first-run`

`make smoke-first-run` 在隔离临时目录中跑通：

```text
init -> doctor -> next -> generate -> trigger configure -> collect ingest/status/review -> trigger status/process-next -> eval -> promote -> serve preview
```

这条链路证明用户可以从空 workspace 进入一个可理解、可观测、可继续操作的闭环，而不是靠隐含状态或人工拼命令。

### 2.2 Auto-train queue 与候选生命周期

`make smoke-auto-train-queue` 现在覆盖自动训练队列的核心路径：

- feedback 信号进入队列
- queue item 完成处理
- mock-local adapter manifest 可见
- `pfe next` 能识别 `candidate_ready`

候选 adapter、queue 状态、candidate action 和 operations console 的状态面已经统一到 CLI 与 HTTP surface。

### 2.3 Real-local readiness

新增了 dependency-safe 的 real-local 预检：

- `pfe train --backend peft --real-local --preview`
- `make smoke-real-local-readiness`

这条链路不会下载模型，也不会要求安装重训练依赖。它验证的是：

- 本地模型路径发现
- doctor readiness 输出
- real-local train plan
- serve preview real-local 标记
- console snapshot wiring

这解决了一个关键问题：用户在安装重依赖之前，就能知道本地配置是否被系统正确识别。

### 2.4 Live server 与 browser-facing surface

新增并通过了两条 live server smoke：

- `make smoke-server-live`
- `make smoke-dashboard-console-live`

覆盖范围包括：

- `GET /healthz`
- `GET /pfe/status?detail=full`
- `GET /dashboard`
- `GET /pfe/dashboard/metrics`
- `GET /pfe/dashboard/training`
- `GET /pfe/dashboard/signals`
- `GET /pfe/dashboard/adapters`
- `GET /pfe/dashboard/health`
- `GET /`
- `POST /v1/chat/completions`
- `POST /pfe/feedback`

这把 server、dashboard API、chat console 和 feedback 闭环从 in-process 测试推进到真实 loopback HTTP 验证。

### 2.5 Browser JS smoke

新增了可选浏览器级验证入口：

- `make smoke-browser-ui-live`
- `tools/browser_ui_live_smoke.py`

未安装 Playwright 时，该目标会跳过并给出 setup 提示；安装 `e2e` extras 并执行 `python -m playwright install chromium` 后，它会在真实 Chromium 中执行：

- dashboard 页面加载
- dashboard refresh
- dashboard 指标 DOM 更新
- chat console 页面加载
- 输入消息并发送
- assistant 回复渲染
- accept feedback 按钮点击
- feedback 状态进入页面文本

本轮还用 Codex 内置浏览器对临时 live server 做过一次真实 JS 交互验证：

- `/dashboard` 标题为 `PFE Observability Dashboard`
- dashboard 指标 DOM 从 loading 状态更新
- `/` chat console 成功发送消息
- assistant bubble 数量为 1
- accept feedback 后页面出现 `feedback accept`
- server log 记录了 `/v1/chat/completions` 与 `/pfe/feedback` 的真实请求

## 3. 当前验证结果

### 3.1 Beta smoke

```bash
make smoke-beta
```

结果：通过。

`smoke-beta` 当前包含：

- `smoke-first-run`
- `smoke-auto-train-queue`
- `smoke-real-local-readiness`
- `smoke-server-live`
- `smoke-dashboard-console-live`

### 3.2 Browser-facing smoke

```bash
make smoke-dashboard-console-live
```

结果：通过。

```bash
make smoke-browser-ui-live
```

当前本机 `.venv` 已安装 e2e extras 和 Chromium；release evidence 已用 strict 浏览器路径通过 dashboard 与 chat console 的真实 JS smoke。

### 3.3 Real-local happy path

```bash
make smoke-real-local-happy
```

当前 release evidence 已设置：

```bash
PFE_REAL_LOCAL_MODEL=/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local
```

并用可复现 tiny Hugging Face GPT-2 兼容模型跑通 strict PEFT real-local happy path。

本机发现了可选本地模型目录：

```text
models/Qwen2.5-0.5B-Instruct-4bit
models/Qwen3-4B
```

其他机器复现真实 full happy path 时，需要先安装 training/e2e 依赖，再执行：

```bash
PFE_REAL_LOCAL_MODEL=/abs/path/to/local-model make smoke-real-local-happy
```

### 3.4 全量默认回归

```bash
make test
```

结果：

```text
990 passed, 16 skipped, 52 deselected
```

### 3.5 工作区卫生

已确认：

- `git diff --check` 通过
- `uv.lock` 不存在
- 项目根目录 `.pfe` 不存在
- `videos/` 仍为未跟踪本地文件，未纳入本轮改动

## 4. 当前还剩什么

当前不再有 Phase 2 功能 blocker，但还有几类 release-readiness 边界：

1. **真实本地模型 full happy path 已有本机 strict 证据**

   当前 release evidence 已使用本地 tiny Hugging Face 兼容模型跑通：

   ```bash
   PFE_REAL_LOCAL_MODEL=/Users/lichenhao/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict
   ```

2. **Playwright 浏览器 smoke 已有 strict 证据**

   本机已安装 e2e extras 和 Chromium，`smoke-release-strict` 会把浏览器 JS smoke 作为必过 gate。若纳入 CI，需要同样安装：

   ```bash
   .venv/bin/python -m pip install -e '.[e2e]'
   .venv/bin/python -m playwright install chromium
   ```

3. **30 分钟长稳态 soak 已有本机证据**

   `release_soak_smoke.py --duration-seconds 1800 --interval-seconds 2` 已通过，覆盖 30 分钟 queue / daemon / server / dashboard / chat / feedback 长稳态轮询。正式发布前仍可按需补 60 分钟窗口。

4. **性能与内存 budget 已有首版**

   `make benchmark-release` 已记录 first-run、strict browser UI、real-local happy path 和短 soak 的耗时/峰值 RSS，并默认执行 release budget。

5. **Dashboard 外部资源已移除**

   dashboard 已移除外部 Chart.js / Google Fonts，改用内联 `OfflineChart` canvas fallback；live smoke 会拒绝外部资源 URL。

## 5. 建议的下一步

优先级建议：

1. 在 GitHub Actions 上验证 `smoke-release-strict` / `benchmark-release`
2. 把远端 CI run URL、结论和关键日志摘要补入 `release-readiness-evidence.md`
3. 进入 Phase 3：多模型兼容、插件体系、示例应用、文档站

## 6. 最终判断

截至 **2026-06-15**，可以把项目当前状态表述为：

> `Phase 2` 功能闭环已经完成，默认 beta 验证链已覆盖 CLI、队列、server、dashboard、chat console 和 feedback。strict release gate、真实浏览器 JS smoke、real-local tiny model happy path、30 分钟长稳态 soak、dashboard offline-first、性能/内存 budget 和最终发布材料已有本机证据。剩余工作不再是 Phase 2 功能研发，而是 release-readiness 收尾：远端 CI 证据。
