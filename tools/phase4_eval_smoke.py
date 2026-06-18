#!/usr/bin/env python3
"""Smoke-test Phase 4 base/local evaluation report generation."""

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
    with tempfile.TemporaryDirectory(prefix="pfe-phase4-eval-") as tempdir:
        previous_home = os.environ.get("PFE_HOME")
        os.environ["PFE_HOME"] = str(Path(tempdir) / ".pfe")
        try:
            source_path = Path(tempdir) / "phase4-eval.txt"
            source_path.write_text(
                (
                    "The Phase4 evaluation should compare base and local responses on the same "
                    "holdout prompts. Useful metrics include citation hit rate, summary coverage, "
                    "unsupported assertions, and refusal behavior when evidence is insufficient."
                ),
                encoding="utf-8",
            )
            plan = build_serve_plan(workspace="phase4_eval_smoke", dry_run=True)
            app = plan.app
            await _request(
                app,
                "/pfe/phase4/sources",
                method="POST",
                body={"path": str(source_path), "title": "Phase4 eval notes"},
            )
            await _request(app, "/pfe/phase4/training-candidates", method="POST", body={"limit": 6, "export": True})
            adapter = await _request(app, "/pfe/phase4/candidate-adapter", method="POST")
            report = await _request(
                app,
                "/pfe/phase4/eval",
                method="POST",
                body={"adapter_version": adapter["adapter_version"], "attach_to_adapter": True},
            )
            return {
                "ok": True,
                "adapter_version": adapter["adapter_version"],
                "real_model_calls": report["real_model_calls"],
                "gate_status": report["eval_gate"]["status"],
                "comparison": report["comparison"],
                "recommendation": report["recommendation"],
                "citation_delta": report["scores"]["local_delta"]["citation_hit_rate"],
                "summary_delta": report["scores"]["local_delta"]["summary_coverage"],
                "unsupported_assertion_delta": report["scores"]["local_delta"]["unsupported_assertions"],
                "holdout_count": report["holdout_count"],
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
