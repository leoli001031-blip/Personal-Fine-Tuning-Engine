#!/usr/bin/env python3
"""Smoke-test Phase 4 training handoff and real-train skip conditions."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pfe_server.app import build_serve_plan, smoke_test_request


async def _request(
    app: Any,
    path: str,
    *,
    method: str = "GET",
    body: dict[str, Any] | None = None,
    allow_status: set[int] | None = None,
) -> dict[str, Any]:
    result = await smoke_test_request(app, path=path, method=method, body=body)
    if result["status_code"] != 200 and (not allow_status or result["status_code"] not in allow_status):
        raise AssertionError(f"{method} {path} failed: {result}")
    body_payload = dict(result.get("body") or {})
    body_payload["_status_code"] = result["status_code"]
    return body_payload


async def _run() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pfe-phase4-train-") as tempdir:
        previous_home = os.environ.get("PFE_HOME")
        os.environ["PFE_HOME"] = str(Path(tempdir) / ".pfe")
        try:
            source_path = Path(tempdir) / "phase4-training.txt"
            source_path.write_text(
                (
                    "Phase4 training should export eligible real-corpus candidates to the existing "
                    "SFT sample store. Real LoRA training is attempted only when a local trainable "
                    "model path is explicitly configured for this smoke."
                ),
                encoding="utf-8",
            )
            plan = build_serve_plan(workspace="phase4_train_smoke", dry_run=True)
            app = plan.app
            await _request(app, "/pfe/phase4/sources", method="POST", body={"path": str(source_path)})
            await _request(app, "/pfe/phase4/training-candidates", method="POST", body={"limit": 6, "export": True})
            sample_export = await _request(
                app,
                "/pfe/phase4/training-candidates/export",
                method="POST",
                body={"target": "samples_db"},
            )
            train_model = os.environ.get("PFE_PHASE4_REAL_TRAIN_MODEL", "").strip()
            if not train_model:
                adapter = await _request(app, "/pfe/phase4/candidate-adapter", method="POST")
                return {
                    "ok": True,
                    "real_training": "skipped",
                    "skip_reason": "PFE_PHASE4_REAL_TRAIN_MODEL is not set",
                    "saved_training_samples": sample_export["saved_samples"],
                    "mock_fallback": True,
                    "candidate_adapter_version": adapter["adapter_version"],
                    "candidate_adapter_state": adapter["state"],
                    "training_endpoint": "/pfe/training/jobs",
                }
            if not Path(train_model).expanduser().exists():
                adapter = await _request(app, "/pfe/phase4/candidate-adapter", method="POST", body={"base_model": train_model})
                return {
                    "ok": True,
                    "real_training": "skipped",
                    "skip_reason": f"PFE_PHASE4_REAL_TRAIN_MODEL does not exist: {train_model}",
                    "saved_training_samples": sample_export["saved_samples"],
                    "mock_fallback": True,
                    "candidate_adapter_version": adapter["adapter_version"],
                    "candidate_adapter_state": adapter["state"],
                    "training_endpoint": "/pfe/training/jobs",
                }

            preflight = await _request(
                app,
                "/pfe/training/jobs",
                method="POST",
                body={"method": "sft", "base_model": train_model},
                allow_status={409},
            )
            ready = bool(dict(preflight.get("preflight") or preflight).get("ready"))
            if not ready:
                adapter = await _request(app, "/pfe/phase4/candidate-adapter", method="POST", body={"base_model": train_model})
                return {
                    "ok": True,
                    "real_training": "skipped",
                    "skip_reason": "training preflight is not ready",
                    "preflight": preflight.get("preflight") or preflight,
                    "saved_training_samples": sample_export["saved_samples"],
                    "mock_fallback": True,
                    "candidate_adapter_version": adapter["adapter_version"],
                    "candidate_adapter_state": adapter["state"],
                    "training_endpoint": "/pfe/training/jobs",
                }

            job = await _request(
                app,
                "/pfe/training/jobs",
                method="POST",
                body={"method": "sft", "base_model": train_model, "confirm": True},
                allow_status={202},
            )
            return {
                "ok": True,
                "real_training": "started",
                "saved_training_samples": sample_export["saved_samples"],
                "job": job,
                "training_endpoint": "/pfe/training/jobs",
            }
        finally:
            if previous_home is None:
                os.environ.pop("PFE_HOME", None)
            else:
                os.environ["PFE_HOME"] = previous_home


def main() -> int:
    print(json.dumps(asyncio.run(_run()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
