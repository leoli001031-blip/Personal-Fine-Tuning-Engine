#!/usr/bin/env python3
"""Smoke-test the Phase 3 signal inbox and candidate plan API."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pfe_server.app import build_serve_plan, smoke_test_request


async def _request(app: Any, path: str, *, method: str = "GET", body: dict[str, Any] | None = None) -> dict[str, Any]:
    result = await smoke_test_request(app, path=path, method=method, body=body)
    if result["status_code"] != 200:
        raise AssertionError(f"{method} {path} failed: {result}")
    return dict(result["body"])


async def _run() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pfe-phase3-smoke-") as tempdir:
        previous_home = os.environ.get("PFE_HOME")
        os.environ["PFE_HOME"] = str(Path(tempdir) / ".pfe")
        try:
            plan = build_serve_plan(workspace="phase3_smoke", dry_run=True)
            app = plan.app
            start = await _request(app, "/pfe/phase3")
            signal = await _request(
                app,
                "/pfe/phase3/signals",
                method="POST",
                body={
                    "signal_type": "edit",
                    "user_input": "请整理合同交付条款：乙方需 7 日内交付。",
                    "model_output": "这条款完全没问题。",
                    "corrected_output": "摘要：乙方需 7 日内交付。风险：违约金和验收口径需人工确认。本输出不是法律结论。",
                    "confidence": 0.9,
                },
            )
            candidates = await _request(app, "/pfe/phase3/training-candidates")
            candidate_plan = await _request(
                app,
                "/pfe/phase3/candidate-plan",
                method="POST",
                body={"persona_id": "ops-analyst", "scenario_id": "contract-risk-summary"},
            )
            return {
                "ok": True,
                "workspace": start["workspace"],
                "persona": start["personas"][0]["persona_id"],
                "scenario": start["scenarios"][0]["scenario_id"],
                "signal_id": signal["signal"]["signal_id"],
                "signal_type": signal["signal"]["signal_type"],
                "eligible_for_training": signal["signal"]["eligible_for_training"],
                "candidate_sample_count": candidates["count"],
                "plan_id": candidate_plan["plan_id"],
                "plan_state": candidate_plan["candidate_adapter"]["state"],
                "eval_gate": candidate_plan["eval_gate"]["current_state"],
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
