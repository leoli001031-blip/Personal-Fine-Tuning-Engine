# User acceptance checklist

更新时间：2026-06-15

这份 checklist 用来从真实用户视角验收 PFE Phase 2 release candidate。它关注用户是否能完成闭环，而不是只看单个测试是否为绿。

## 1. 环境准备

```bash
tools/bootstrap_py311_env.sh
source .venv/bin/activate
.venv/bin/python -m pip install -e '.[e2e]'
.venv/bin/python -m pip install 'torch>=2.1' 'transformers>=4.36' 'peft>=0.7' 'accelerate>=0.23' 'safetensors>=0.4'
.venv/bin/python -m playwright install chromium
.venv/bin/python tools/prepare_tiny_hf_model.py
```

验收标准：

- `.venv/bin/python` 是 Python 3.10 或更新版本。
- `tools/prepare_tiny_hf_model.py` 生成 `$HOME/.cache/pfe/release-models/tiny-gpt2-local`。

## 2. 初始化与诊断

```bash
pfe init --workspace user_default --base-model $HOME/.cache/pfe/release-models/tiny-gpt2-local
pfe doctor --workspace user_default
pfe next --workspace user_default
```

验收标准：

- `pfe init` 创建 workspace config。
- `pfe doctor` 能显示 local model availability、trainer deps、adapter home、signal chain 和 capability boundaries。
- `pfe next` 给出下一步命令，不停在模糊错误上。

## 3. 样本生成与队列训练

```bash
pfe generate --scenario life-coach --style warm --num 8 --workspace user_default
pfe trigger configure --workspace user_default --enable --min-new-samples 1 --queue-mode deferred --max-interval-days 0 --no-require-confirmation --epochs 1 --backend mock_local
pfe collect ingest --workspace user_default --event-id evt-uat-1 --request-id req-uat-1 --session-id sess-uat-1 --source-event-id evt-uat-chat-1 --user-input "Help me pick a focused next step." --model-output "Choose one task that can be completed in 20 minutes." --action accept --scenario life-coach
pfe trigger process-next --workspace user_default
```

验收标准：

- 生成样本保存成功。
- feedback signal 被采集。
- queue item 能完成处理并生成 adapter manifest。
- `pfe next` 进入 candidate/eval/promote 相关状态。

## 4. Eval 与 promote

用 `pfe next` 或 queue smoke 输出中的 adapter version 替换 `<version>`：

```bash
pfe eval --base-model base --adapter <version> --num-samples 3 --workspace user_default
pfe adapter promote <version> --workspace user_default
```

验收标准：

- eval 返回结果摘要。
- promote 后 `pfe doctor` 或 `pfe status` 能看到 latest/promoted adapter。

## 5. Live server 与 chat

```bash
pfe serve --workspace user_default --port 8921 --live
```

打开：

```text
http://127.0.0.1:8921/
http://127.0.0.1:8921/dashboard
```

验收标准：

- `/healthz` 返回 ok。
- Chat console 能发送消息并显示 assistant response。
- feedback accept/reject 控件可用。
- Dashboard 能显示 System Health、Training Loss、Signal Quality、Daily Signal Volume、Adapter Performance Comparison。
- Dashboard 页面无外部 `http://` 或 `https://` 资源依赖。

## 6. Automated release acceptance

```bash
PFE_REAL_LOCAL_MODEL=$HOME/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict
make benchmark-release
.venv/bin/python tools/release_soak_smoke.py --duration-seconds 1800 --interval-seconds 2 --report-path /tmp/pfe-release-soak-30m-report.json
```

验收标准：

- strict release smoke 通过。
- benchmark release 通过默认 budget。
- 30 分钟 soak 通过，daemon 保持 `healthy/fresh/valid`。
- 命令结束后没有残留 `pfe_core.worker_daemon` 或 `pfe_cli.main serve` 进程。
- 仓库根目录没有 `.pfe` 或 `uv.lock`。

## 7. Evidence update

验收完成后，把以下内容写入 `docs/reference/release-readiness-evidence.md`：

- strict release smoke 结果。
- benchmark report path 和 budget summary。
- 30 分钟 soak report path、duration、iterations、probes、chat turns、latency、daemon 状态。
- 清理验证结果。
- 如果是 CI 或 release job，补充 GitHub Actions run URL。
