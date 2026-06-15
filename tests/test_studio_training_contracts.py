from pfe_server.studio_training_contracts import (
    build_legacy_training_trigger_preflight_payload,
    build_training_preflight_payload,
    training_request_from_body,
)


def test_training_request_defaults_method_and_removes_control_fields() -> None:
    request = training_request_from_body(
        {
            "method": "",
            "confirm": False,
            "auto_trigger": True,
            "base_model": "models/base",
            "epochs": 2,
        },
        {"confirm": "true"},
    )

    assert request.method == "sft"
    assert request.raw_method == ""
    assert request.confirmed is True
    assert request.training_config == {"base_model": "models/base", "epochs": 2}
    assert request.payload() == {
        "method": "sft",
        "training_config": {"base_model": "models/base", "epochs": 2},
        "confirmed": True,
    }


def test_training_request_reports_unsupported_method() -> None:
    request = training_request_from_body({"method": "lora", "confirm": True})

    assert request.method == "lora"
    assert request.unsupported_method == "lora"


def test_training_preflight_payload_reports_blockers_and_warnings() -> None:
    request = training_request_from_body({"method": "dpo", "epochs": 1}, confirmed=False)
    payload = build_training_preflight_payload(
        request=request,
        readiness={
            "summary": {"ready": False},
            "configuration": {"base_model": "configured-base"},
            "model": {"source": {"ok": False, "state": "missing_base_model"}},
            "runtime": {"dependencies": {"ok": False}},
            "inference": {"real_local_enabled": False},
            "version": {"current": None},
        },
        workspace="client-a",
        base_model="fallback-base",
    )

    assert payload["ready"] is False
    assert payload["requires_confirmation"] is True
    assert payload["confirm_api"] == "POST /pfe/training/jobs"
    assert payload["base_model"] == "configured-base"
    assert payload["blocked_by"] == ["missing_base_model"]
    assert payload["warnings"] == [
        "runtime_dependencies_missing",
        "real_local_inference_disabled",
    ]
    assert payload["next_actions"][0]["id"] == "choose_local_model"
    assert payload["preview"] == {
        "method": "dpo",
        "training_config": {"epochs": 1},
        "will_create_job": False,
        "will_start_background_training": False,
    }


def test_legacy_training_trigger_preflight_keeps_compatibility_contract() -> None:
    request = training_request_from_body(
        {"method": "sft", "base_model": "request-base"},
        confirmed=True,
    )
    payload = build_legacy_training_trigger_preflight_payload(
        request=request,
        workspace="client-a",
        base_model="fallback-base",
    )

    assert payload["ready"] is True
    assert payload["requires_confirmation"] is False
    assert payload["confirm_api"] == "POST /pfe/training/trigger"
    assert payload["base_model"] == "request-base"
    assert payload["warnings"] == ["legacy_trigger_bypasses_studio_preflight"]
    assert payload["preview"]["legacy_endpoint"] is True
    assert payload["preview"]["will_create_job"] is True
    assert payload["preview"]["will_start_background_training"] is True
