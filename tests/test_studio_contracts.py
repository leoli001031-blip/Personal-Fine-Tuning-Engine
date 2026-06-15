from pfe_server.studio_contracts import build_runtime_payload, runtime_host_port


def test_runtime_host_port_prefers_request_host_header() -> None:
    assert runtime_host_port({"host": "127.0.0.1:9012"}, {}) == ("127.0.0.1", 9012)
    assert runtime_host_port({"host": "localhost:not-a-port"}, {}) == ("localhost", 8921)


def test_runtime_payload_exposes_openai_chat_handoff_contract() -> None:
    payload = build_runtime_payload(
        headers={"host": "127.0.0.1:9012"},
        runtime_probe={},
        workspace="client-a",
        provider="core",
        allow_remote_access=False,
        privacy_mode="strict_local",
        auth_mode="api_key_required",
        started_at=100.0,
        now=105.4321,
    )

    assert payload["workspace"] == "client-a"
    assert payload["web_url"] == "http://127.0.0.1:9012/"
    assert payload["api_url"] == "http://127.0.0.1:9012/v1/chat/completions"
    assert payload["access_scope"] == "仅本机"
    assert payload["api_key_required"] is True
    assert payload["uptime_seconds"] == 5.432
    assert payload["api"]["kind"] == "openai_chat_completions"
    assert payload["api"]["method"] == "POST"
    assert payload["api"]["model_parameter"] == "local"
    assert payload["api"]["model_aliases"] == ["local", "local-default", "base"]
    assert payload["api"]["auth_header"] == "Authorization: Bearer $PFE_API_KEY"
    assert payload["api"]["request_body"]["model"] == "local"
