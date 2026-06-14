#!/usr/bin/env python3
"""Run an optional browser-level live smoke with Playwright."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from dashboard_console_live_smoke import _check_console_surface, _check_dashboard_surface
from first_run_smoke import _default_python, _repo_root
from server_live_smoke import _free_loopback_port, _prepare_workspace, _start_server, _stop_server, _wait_for_healthz


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


def _run_browser_checks(base_url: str, *, headless: bool, timeout_ms: int) -> None:
    sync_playwright = _load_sync_playwright()
    if sync_playwright is None:
        raise RuntimeError("Playwright is not installed")

    errors: list[str] = []
    network_events: list[str] = []
    with sync_playwright() as playwright:  # type: ignore[operator]
        browser = playwright.chromium.launch(headless=headless)
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
                    for path in ("/v1/chat/completions", "/pfe/feedback", "/pfe/status", "/pfe/dashboard")
                )
                else None,
            )
            page.on(
                "response",
                lambda response: network_events.append(f"response {response.status} {response.url}")
                if any(
                    path in response.url
                    for path in ("/v1/chat/completions", "/pfe/feedback", "/pfe/status", "/pfe/dashboard")
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

                page.goto(f"{base_url}/", wait_until="domcontentloaded", timeout=timeout_ms)
                page.wait_for_selector("#chatForm", timeout=timeout_ms)
                page.fill("#messageInput", "Give me one browser-level beta validation step.")
                page.click("#sendBtn")
                page.wait_for_selector(".bubble.assistant .feedback-btn.accept", timeout=timeout_ms)
                page.locator(".bubble.assistant .feedback-btn.accept").last.click(timeout=timeout_ms)
                page.wait_for_function(
                    "() => document.body.innerText.includes('feedback accept')",
                    timeout=timeout_ms,
                )
                _assert_no_browser_errors(errors)
            except Exception as exc:
                diagnostics = page.evaluate(
                    """() => ({
                        url: location.href,
                        footerHint: document.querySelector('#footerHint')?.textContent || '',
                        sendDisabled: Boolean(document.querySelector('#sendBtn')?.disabled),
                        assistantCount: document.querySelectorAll('.bubble.assistant').length,
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
        _check_console_surface(base_url, request_timeout=args.request_timeout)
        _run_browser_checks(base_url, headless=not args.show_browser, timeout_ms=args.browser_timeout_ms)
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
        "base_url": base_url,
        "port": str(port),
    }


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Optional browser-level smoke for dashboard and chat console. "
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
        summary = _run_smoke(args, workdir)
        print("BROWSER UI LIVE SMOKE PASSED")
        print(f"workspace: {summary['workspace']}")
        print(f"version:   {summary['version']}")
        print(f"base_url:  {summary['base_url']}")
        return 0
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
