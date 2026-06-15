from __future__ import annotations

import threading
import time
import urllib.request
import webbrowser
from typing import Callable, Protocol


class HttpGet(Protocol):
    def __call__(self, url: str, timeout: float) -> object:
        ...


def browser_host(host: str) -> str:
    value = str(host or "").strip() or "127.0.0.1"
    if value in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    if ":" in value and not value.startswith("["):
        return f"[{value}]"
    return value


def studio_url(host: str, port: int, path: str = "/studio") -> str:
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"http://{browser_host(host)}:{int(port)}{normalized_path}"


def open_when_ready(
    url: str,
    health_url: str,
    *,
    timeout_seconds: float = 10.0,
    interval_seconds: float = 0.25,
    http_get: HttpGet = urllib.request.urlopen,
    open_browser: Callable[[str], object] = webbrowser.open,
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        try:
            response = http_get(health_url, timeout=min(max(interval_seconds, 0.01), 1.0))
            close = getattr(response, "close", None)
            if callable(close):
                close()
            open_browser(url)
            return True
        except Exception:
            if time.monotonic() >= deadline:
                return False
            time.sleep(max(0.0, interval_seconds))


def schedule_open_studio(host: str, port: int, *, timeout_seconds: float = 10.0) -> tuple[str, threading.Thread]:
    url = studio_url(host, port)
    health_url = studio_url(host, port, "/healthz")
    thread = threading.Thread(
        target=open_when_ready,
        kwargs={"url": url, "health_url": health_url, "timeout_seconds": timeout_seconds},
        daemon=True,
    )
    thread.start()
    return url, thread


__all__ = [
    "browser_host",
    "open_when_ready",
    "schedule_open_studio",
    "studio_url",
]
