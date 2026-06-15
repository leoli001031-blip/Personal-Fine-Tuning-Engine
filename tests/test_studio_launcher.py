from __future__ import annotations

import importlib

from pfe_server.studio_launcher import open_when_ready, studio_url


class FakeResponse:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_studio_url_prefers_loopback_for_wildcard_hosts() -> None:
    assert studio_url("127.0.0.1", 8921) == "http://127.0.0.1:8921/studio"
    assert studio_url("0.0.0.0", 8921) == "http://127.0.0.1:8921/studio"
    assert studio_url("::", 8921) == "http://127.0.0.1:8921/studio"


def test_open_when_ready_waits_for_health_before_opening_browser() -> None:
    attempts: list[str] = []
    opened: list[str] = []
    response = FakeResponse()

    def http_get(url: str, timeout: float) -> object:
        del timeout
        attempts.append(url)
        if len(attempts) == 1:
            raise OSError("not ready")
        return response

    result = open_when_ready(
        "http://127.0.0.1:8921/studio",
        "http://127.0.0.1:8921/healthz",
        timeout_seconds=1.0,
        interval_seconds=0.0,
        http_get=http_get,
        open_browser=opened.append,
    )

    assert result is True
    assert attempts == ["http://127.0.0.1:8921/healthz", "http://127.0.0.1:8921/healthz"]
    assert response.closed is True
    assert opened == ["http://127.0.0.1:8921/studio"]


def test_open_when_ready_does_not_open_browser_when_health_times_out() -> None:
    opened: list[str] = []

    def http_get(url: str, timeout: float) -> object:
        del url, timeout
        raise OSError("not ready")

    result = open_when_ready(
        "http://127.0.0.1:8921/studio",
        "http://127.0.0.1:8921/healthz",
        timeout_seconds=0.0,
        interval_seconds=0.0,
        http_get=http_get,
        open_browser=opened.append,
    )

    assert result is False
    assert opened == []


def test_pfe_server_main_schedules_studio_browser_open_by_default(monkeypatch, capsys) -> None:
    main_module = importlib.import_module("pfe_server.__main__")
    scheduled: list[tuple[str, int]] = []
    served: list[dict[str, object]] = []

    def fake_schedule(host: str, port: int) -> tuple[str, object]:
        scheduled.append((host, port))
        return f"http://{host}:{port}/studio", object()

    def fake_serve(**kwargs: object) -> str:
        served.append(kwargs)
        return "served"

    monkeypatch.setattr(main_module, "schedule_open_studio", fake_schedule)
    monkeypatch.setattr(main_module, "serve", fake_serve)

    assert main_module.main(["--port", "9234", "--workspace", "client-a"]) == 0

    output = capsys.readouterr().out
    assert "Opening PFE Studio at http://127.0.0.1:9234/studio" in output
    assert "served" in output
    assert scheduled == [("127.0.0.1", 9234)]
    assert served[0]["workspace"] == "client-a"
    assert served[0]["dry_run"] is False


def test_pfe_server_main_no_open_keeps_browser_closed(monkeypatch, capsys) -> None:
    main_module = importlib.import_module("pfe_server.__main__")
    scheduled: list[tuple[str, int]] = []

    monkeypatch.setattr(main_module, "schedule_open_studio", lambda host, port: scheduled.append((host, port)))
    monkeypatch.setattr(main_module, "serve", lambda **kwargs: "served")

    assert main_module.main(["--no-open", "--port", "9234"]) == 0

    output = capsys.readouterr().out
    assert "Opening PFE Studio" not in output
    assert "served" in output
    assert scheduled == []
