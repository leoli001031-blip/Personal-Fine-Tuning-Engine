#!/usr/bin/env python3
"""Run an optional browser-level live smoke with Playwright."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from dashboard_console_live_smoke import _check_dashboard_surface, _check_studio_surface
from first_run_smoke import _default_python, _repo_root
from server_live_smoke import _free_loopback_port, _prepare_workspace, _start_server, _stop_server, _wait_for_healthz


_STUDIO_RAW_ISSUE_CODES = (
    "needs_local_path",
    "real_local_inference_disabled",
    "runtime_dependencies_missing",
)


def _load_sync_playwright() -> object | None:
    try:
        from playwright.sync_api import sync_playwright

        return sync_playwright
    except Exception:
        return None


def _assert_no_browser_errors(errors: list[str]) -> None:
    relevant = [
        item
        for item in errors
        if "Failed to load resource" not in item
        and "net::ERR_ABORTED" not in item
        and "favicon" not in item.lower()
    ]
    if relevant:
        raise AssertionError("browser console/page errors were observed:\n" + "\n".join(relevant))


def _write_browser_local_model(workdir: Path) -> str:
    model_dir = workdir / "models" / "browser-studio-local"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GPT2LMHeadModel"],
                "bos_token_id": 0,
                "eos_token_id": 1,
                "model_type": "gpt2",
                "pad_token_id": 0,
                "vocab_size": 32,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return str(model_dir.resolve())


def _is_playwright_prereq_error(error: BaseException) -> bool:
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "playwright is not installed",
            "playwright chromium is not available",
            "executable doesn't exist",
            "playwright install",
            "browserType.launch".lower(),
        )
    )


def _assert_no_raw_studio_issue_codes(page: object, *, timeout_ms: int) -> None:
    page.wait_for_function(
        """(codes) => {
            const text = document.body?.innerText || "";
            return !codes.some((code) => text.includes(code));
        }""",
        arg=list(_STUDIO_RAW_ISSUE_CODES),
        timeout=timeout_ms,
    )


def _check_studio_click_flow(page: object, base_url: str, *, model_path: str, timeout_ms: int) -> dict[str, str]:
    page.goto(f"{base_url}/", wait_until="domcontentloaded", timeout=timeout_ms)
    page.wait_for_selector("#studioSummaryTitle", timeout=timeout_ms)
    page.wait_for_function(
        "() => document.querySelector('#apiUrlValue')?.textContent?.includes('/v1/chat/completions')",
        timeout=timeout_ms,
    )
    _assert_no_raw_studio_issue_codes(page, timeout_ms=timeout_ms)

    with page.expect_response(
        lambda response: response.url.endswith("/pfe/config/model") and response.status == 200,
        timeout=timeout_ms,
    ):
        page.fill("#modelPathInput", model_path)
        page.click("#saveModelPathButton")
    page.wait_for_function(
        """(modelPath) => {
            const value = document.querySelector('#modelValue')?.textContent || "";
            const input = document.querySelector('#modelPathInput')?.value || "";
            return value.includes(modelPath) || input === modelPath;
        }""",
        arg=model_path,
        timeout=timeout_ms,
    )
    _assert_no_raw_studio_issue_codes(page, timeout_ms=timeout_ms)

    with page.expect_response(
        lambda response: response.url.endswith("/pfe/config/real-local") and response.status == 200,
        timeout=timeout_ms,
    ):
        page.click("#realLocalToggleButton")
    page.wait_for_function(
        "() => document.querySelector('#realLocalToggleButton')?.textContent?.includes('暂停本地模型回复')",
        timeout=timeout_ms,
    )
    _assert_no_raw_studio_issue_codes(page, timeout_ms=timeout_ms)

    with page.expect_response(
        lambda response: response.url.endswith("/pfe/training/jobs") and response.status == 409,
        timeout=timeout_ms,
    ):
        page.click("#trainingPreflightButton")
    page.wait_for_function(
        "() => document.querySelector('#trainingValue')?.textContent?.trim() !== '未检查'",
        timeout=timeout_ms,
    )
    _assert_no_raw_studio_issue_codes(page, timeout_ms=timeout_ms)
    return page.evaluate(
        """() => ({
            studio_model_path: document.querySelector('#modelPathInput')?.value || "",
            studio_request_meta: document.querySelector('#requestMeta')?.textContent || "",
            studio_training_value: document.querySelector('#trainingValue')?.textContent || "",
            studio_training_meta: document.querySelector('#trainingMeta')?.textContent || ""
        })"""
    )


def _run_browser_checks(base_url: str, *, workdir: Path, headless: bool, timeout_ms: int) -> dict[str, str]:
    sync_playwright = _load_sync_playwright()
    if sync_playwright is None:
        raise RuntimeError("Playwright is not installed")

    errors: list[str] = []
    network_events: list[str] = []
    with sync_playwright() as playwright:  # type: ignore[operator]
        try:
            browser = playwright.chromium.launch(headless=headless)
        except Exception as exc:
            raise RuntimeError("Playwright Chromium is not available") from exc
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.on("console", lambda msg: errors.append(f"console.{msg.type}: {msg.text}") if msg.type in {"error"} else None)
            page.on("pageerror", lambda exc: errors.append(f"pageerror: {exc}"))
            page.on(
                "requestfailed",
                lambda request: network_events.append(
                    f"requestfailed {request.method} {request.url} {request.failure}"
                )
                if any(
                    path in request.url
                    for path in (
                        "/v1/chat/completions",
                        "/pfe/feedback",
                        "/pfe/status",
                        "/pfe/dashboard",
                        "/pfe/runtime",
                        "/pfe/workspaces",
                        "/pfe/models",
                        "/pfe/readiness",
                        "/pfe/config/model",
                        "/pfe/config/real-local",
                        "/pfe/training/jobs",
                        "/pfe/adapters",
                        "/pfe/eval/status",
                    )
                )
                else None,
            )
            page.on(
                "response",
                lambda response: network_events.append(f"response {response.status} {response.url}")
                if any(
                    path in response.url
                    for path in (
                        "/v1/chat/completions",
                        "/pfe/feedback",
                        "/pfe/status",
                        "/pfe/dashboard",
                        "/pfe/runtime",
                        "/pfe/workspaces",
                        "/pfe/models",
                        "/pfe/readiness",
                        "/pfe/config/model",
                        "/pfe/config/real-local",
                        "/pfe/training/jobs",
                        "/pfe/adapters",
                        "/pfe/eval/status",
                    )
                )
                else None,
            )

            try:
                page.goto(f"{base_url}/dashboard", wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_selector("#refreshBtn", timeout=timeout_ms)
                page.wait_for_function(
                    "() => document.querySelector('#totalAdapters')?.textContent?.trim() !== '-'",
                    timeout=timeout_ms,
                )
                page.wait_for_function(
                    "() => document.querySelector('#latestAdapter')?.textContent?.trim() !== '-'",
                    timeout=timeout_ms,
                )
                with page.expect_response(
                    lambda response: response.url.endswith("/pfe/dashboard/metrics") and response.status == 200,
                    timeout=timeout_ms,
                ):
                    page.click("#refreshBtn")

                studio_summary = _check_studio_click_flow(
                    page,
                    base_url,
                    model_path=_write_browser_local_model(workdir),
                    timeout_ms=timeout_ms,
                )
                _assert_no_browser_errors(errors)
                return studio_summary
            except Exception as exc:
                diagnostics = page.evaluate(
                    """() => ({
                        url: location.href,
                        summaryTitle: document.querySelector('#studioSummaryTitle')?.textContent || '',
                        apiUrl: document.querySelector('#apiUrlValue')?.textContent || '',
                        modelValue: document.querySelector('#modelValue')?.textContent || '',
                        requestMeta: document.querySelector('#requestMeta')?.textContent || '',
                        trainingValue: document.querySelector('#trainingValue')?.textContent || '',
                        trainingMeta: document.querySelector('#trainingMeta')?.textContent || '',
                        bodyText: document.body?.innerText?.slice(0, 1200) || ''
                    })"""
                )
                raise AssertionError(
                    "browser UI smoke failed\n"
                    f"diagnostics: {diagnostics}\n"
                    f"network_events: {network_events[-20:]}\n"
                    f"errors: {errors[-20:]}"
                ) from exc
        finally:
            browser.close()


def _run_smoke(args: argparse.Namespace, workdir: Path) -> dict[str, str]:
    setup = _prepare_workspace(args, workdir)
    port = args.port if args.port else _free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    process = _start_server(args, workdir, port)
    try:
        healthz = _wait_for_healthz(base_url, process, timeout_seconds=args.server_timeout)
        if healthz["body"].get("status") != "ok":  # type: ignore[union-attr]
            raise AssertionError(f"unexpected healthz payload: {healthz}")
        _check_dashboard_surface(base_url, request_timeout=args.request_timeout)
        _check_studio_surface(base_url, request_timeout=args.request_timeout)
        studio_summary = _run_browser_checks(
            base_url,
            workdir=workdir,
            headless=not args.show_browser,
            timeout_ms=args.browser_timeout_ms,
        )
    finally:
        stdout, stderr = _stop_server(process)

    if process.returncode not in {0, -15, -9, None}:
        raise AssertionError(
            f"server exited unexpectedly after browser smoke (code={process.returncode})\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    return {
        **setup,
        **studio_summary,
        "base_url": base_url,
        "port": str(port),
    }


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Optional browser-level smoke for dashboard and Studio. "
            "Install Playwright and its Chromium browser to run it."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root))
    parser.add_argument("--workspace", default="browser_ui_live")
    parser.add_argument("--port", type=int, default=0, help="Port to bind. Defaults to a free loopback port.")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--server-timeout", type=float, default=15.0)
    parser.add_argument("--request-timeout", type=float, default=5.0)
    parser.add_argument("--browser-timeout-ms", type=int, default=45000)
    parser.add_argument("--strict", action="store_true", help="Fail instead of skipping when Playwright is unavailable.")
    parser.add_argument("--show-browser", action="store_true")
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()

    if _load_sync_playwright() is None:
        print("BROWSER UI LIVE SMOKE SKIPPED")
        print("reason: Playwright is not installed")
        print("hint: install the e2e extras and run `python -m playwright install chromium`")
        return 2 if args.strict else 0

    tempdir = None
    if args.workdir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="pfe-browser-ui-live-smoke-")
        workdir = Path(tempdir.name)
    else:
        workdir = args.workdir.resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    print(f"workdir: {workdir}")
    print(f"python:  {args.python}")
    print()
    try:
        try:
            summary = _run_smoke(args, workdir)
        except RuntimeError as exc:
            if not _is_playwright_prereq_error(exc):
                raise
            print("BROWSER UI LIVE SMOKE SKIPPED")
            print(f"reason: {exc}")
            print("hint: install the e2e extras and run `python -m playwright install chromium`")
            return 2 if args.strict else 0
        print("BROWSER UI LIVE SMOKE PASSED")
        print(f"workspace: {summary['workspace']}")
        print(f"version:   {summary['version']}")
        print(f"base_url:  {summary['base_url']}")
        print(f"studio_model_path:    {summary['studio_model_path']}")
        print(f"studio_training:      {summary['studio_training_value']}")
        return 0
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
