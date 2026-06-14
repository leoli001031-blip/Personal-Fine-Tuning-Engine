# Beta Local Runbook

This runbook is the shortest path for validating a local PFE beta checkout.

## 1. Bootstrap

```bash
tools/bootstrap_py311_env.sh
source .venv/bin/activate
```

The default install is intentionally light. Add real training packages only when
you are ready to test them:

```bash
PFE_BOOTSTRAP_EXTRAS=dev,training tools/bootstrap_py311_env.sh
```

## 2. First Workspace

Use a real local path when you already have model files, or a model id when you
are only preparing configuration.

```bash
pfe init --workspace user_default --base-model /path/to/local/model
pfe doctor --workspace user_default
pfe next --workspace user_default
```

`pfe doctor` should show `local model: available=yes` before you expect
`--real-local` inference or training to work.

## 3. Dependency-Safe Real-Local Check

Before installing heavy training dependencies, verify that the CLI can discover
the local model and render a real-local training plan:

```bash
pfe train --workspace user_default --backend peft --real-local --preview --epochs 1
```

For an automated no-download check:

```bash
make smoke-real-local-readiness
```

This smoke creates an isolated temp workspace with a minimal local model config
marker. It validates `pfe doctor`, `pfe train --real-local --preview`,
`pfe serve --real-local` preview, and one console snapshot. It does not download
weights or launch real training.

When you do have a real local model path and the training extras installed, run
the opt-in happy path:

```bash
PFE_REAL_LOCAL_MODEL=/path/to/local/model make smoke-real-local-happy
```

This verifies `pfe init`, `pfe doctor`, sample generation, real `pfe train
--backend peft --real-local`, adapter manifest real-execution metadata, eval,
promote, and serve preview. Without `PFE_REAL_LOCAL_MODEL`, the target skips with
an explicit setup hint.

For a no-download release gate model, generate the local tiny HF fixture first:

```bash
.venv/bin/python tools/prepare_tiny_hf_model.py
PFE_REAL_LOCAL_MODEL=$HOME/.cache/pfe/release-models/tiny-gpt2-local make smoke-real-local-happy
```

## 4. Local Loop

```bash
pfe generate --scenario life-coach --style warm --num 8 --workspace user_default
pfe trigger configure --workspace user_default --enable --min-new-samples 1 --queue-mode deferred --max-interval-days 0 --no-require-confirmation --epochs 1 --backend mock_local
pfe collect ingest --workspace user_default --help
pfe trigger process-next --workspace user_default
pfe eval --base-model base --adapter <version> --workspace user_default
pfe adapter promote <version> --workspace user_default
```

The isolated version of this path is:

```bash
make smoke-first-run
make smoke-auto-train-queue
```

## 5. Live Server

Preview first:

```bash
pfe serve --workspace user_default --port 8921
```

Start the loopback server:

```bash
pfe serve --workspace user_default --port 8921 --live
```

Then open:

```text
http://127.0.0.1:8921/dashboard
http://127.0.0.1:8921/
```

For an automated live-server check:

```bash
make smoke-server-live
make smoke-dashboard-console-live
make smoke-browser-ui-live
```

This creates an isolated mock-local adapter, launches `pfe serve --live` on a
temporary loopback port, and probes `/healthz`, `/pfe/status`, `/dashboard`,
`/pfe/dashboard/metrics`, and `/v1/chat/completions`.

`make smoke-dashboard-console-live` adds the UI-facing contract: dashboard HTML,
dashboard API endpoints, chat console HTML, and a chat/feedback round trip.

`make smoke-browser-ui-live` is optional and requires Playwright plus a Chromium
browser:

```bash
pip install -e ".[e2e]"
python -m playwright install chromium
make smoke-browser-ui-live
```

Without Playwright it skips with a setup hint, so the default beta smoke chain
stays dependency-light.

For release readiness, use the strict gate so missing browser dependencies fail
instead of being treated as an acceptable skip:

```bash
PFE_REAL_LOCAL_MODEL=$HOME/.cache/pfe/release-models/tiny-gpt2-local make smoke-release-strict
```

Then run the bounded live soak:

```bash
make soak-release
```

`make soak-release` starts an isolated live server and a real worker daemon,
polls server health, `/pfe/status`, dashboard APIs, queue history, daemon
status/history, runner status/history, and periodically performs chat/feedback
round trips. The default duration is 60 seconds. For a release-candidate soak,
run a longer window explicitly:

```bash
.venv/bin/python tools/release_soak_smoke.py --duration-seconds 1800 --interval-seconds 2
```

Record and enforce the release timing and memory budget:

```bash
make benchmark-release
```

`make benchmark-release` runs the first-run smoke, strict browser UI smoke,
real-local happy path, and a short release soak. It records elapsed time and peak
process-tree RSS with `psutil` when available, enforces the default release
budget, then writes a JSON report to `/tmp/pfe-release-perf-report.json`. Use
`tools/release_perf_benchmark.py --no-thresholds` only when you need raw numbers
without failing the release budget.

## 6. Beta Smoke Chain

```bash
make smoke-beta
make test
```

`make smoke-beta` runs the CLI first-run path, auto-train queue path,
real-local readiness path, live-server HTTP path, and dashboard/console live
surface path.

`make smoke-release-strict` runs the same chain and then requires both the
Playwright browser UI smoke and the real-local model happy path. It is expected
to fail until Playwright/Chromium, the PEFT runtime packages, and
`PFE_REAL_LOCAL_MODEL` are configured.
