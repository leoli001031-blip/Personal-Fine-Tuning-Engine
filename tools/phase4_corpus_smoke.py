#!/usr/bin/env python3
"""Smoke-test Phase 4 corpus ingestion and training candidate export."""

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
    with tempfile.TemporaryDirectory(prefix="pfe-phase4-corpus-") as tempdir:
        previous_home = os.environ.get("PFE_HOME")
        os.environ["PFE_HOME"] = str(Path(tempdir) / ".pfe")
        try:
            source_path = Path(tempdir) / "phase4-research.md"
            source_path.write_text(
                (
                    "# Research notes\n\n"
                    "Phase4 collects real source material, chunks it with provenance, and generates "
                    "citation-grounded samples. The assistant should summarize only supplied material "
                    "and ask for more material or human confirmation when evidence is insufficient."
                ),
                encoding="utf-8",
            )
            plan = build_serve_plan(workspace="phase4_smoke", dry_run=True)
            app = plan.app
            start = await _request(app, "/pfe/phase4")
            source = await _request(
                app,
                "/pfe/phase4/sources",
                method="POST",
                body={"path": str(source_path), "title": "Phase4 research notes", "license_status": "local_user_provided"},
            )
            chunks = await _request(app, "/pfe/phase4/chunks")
            candidates = await _request(
                app,
                "/pfe/phase4/training-candidates",
                method="POST",
                body={"limit": 6, "export": True},
            )
            sample_export = await _request(
                app,
                "/pfe/phase4/training-candidates/export",
                method="POST",
                body={"target": "samples_db"},
            )
            return {
                "ok": True,
                "workspace": start["workspace"],
                "source_id": source["source"]["source_id"],
                "source_type": source["source"]["source_type"],
                "chunk_count": len(chunks["chunks"]),
                "candidate_count": candidates["count"],
                "eligible_count": candidates["eligible_count"],
                "candidate_export": candidates.get("export", {}).get("path"),
                "saved_training_samples": sample_export["saved_samples"],
                "split_counts": sample_export["split_counts"],
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
