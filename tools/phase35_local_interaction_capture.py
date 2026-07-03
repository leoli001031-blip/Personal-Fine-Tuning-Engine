#!/usr/bin/env python3
"""Generate Phase35 lightweight local interaction capture evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
PFE_CORE = ROOT / "pfe-core"
PFE_CLI = ROOT / "pfe-cli"
for path in (PFE_CORE, PFE_CLI):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pfe_core.phase35_local_interaction_capture import (
    build_phase35_capture_batch,
    build_phase35_comparison_summary,
    build_phase35_interaction_record,
    build_phase35_phase34_review,
    build_phase35_readiness,
    build_phase35_review_queue,
    load_phase35_state,
    phase35_store_path,
    render_phase35_agent_response,
    save_phase35_state,
    write_jsonl,
)


PHASE34_DIR = Path("docs/demo/phase34-simulated-real-user-acceptance-judge")
PHASE35_DIR = Path("docs/demo/phase35-local-interaction-capture")
_LOCAL_ABS_PATH_RE = re.compile(r"/Users/lichenhao/[^\s\"'，。；;、)）\]]+")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _redact_evidence_tree(path: Path) -> None:
    for item in path.rglob("*"):
        if not item.is_file() or item.suffix not in {".json", ".jsonl", ".md", ".txt"}:
            continue
        text = item.read_text(encoding="utf-8")
        redacted = _LOCAL_ABS_PATH_RE.sub("[LOCAL_PATH]", text)
        if redacted != text:
            item.write_text(redacted, encoding="utf-8")


def _simulated_records(count: int) -> list[dict[str, Any]]:
    goals = [
        "帮我整理当前工作区并判断下一步。",
        "现在情况如何，用短状态告诉我。",
        "你刚才跑偏了，回到证明 PFE 是否真的有用。",
        "准备一个下一阶段追求目标提示词。",
        "检查本地模型服务是否还在跑。",
        "不要碰 videos 和本地 Hermes 配置。",
    ]
    records: list[dict[str, Any]] = []
    for index in range(1, count + 1):
        goal = goals[(index - 1) % len(goals)]
        response = render_phase35_agent_response(user_goal=goal, model_variant="adapter")
        records.append(
            build_phase35_interaction_record(
                workspace="phase35-demo",
                user_goal=goal,
                assistant_response=str(response["assistant_response"]),
                feedback_action="correction" if index % 3 == 0 else "accept",
                user_feedback="模拟本地使用者反馈：这条只用于 Phase35 evidence，不是实际用户反馈。",
                model_variant="adapter",
                operator_id="",
                confirmed_actual_user_feedback=False,
                consent_for_training_candidate_review=False,
                not_scripted_or_curated=False,
                simulated_local_interaction=True,
            )
        )
    return records


def _run_cli_smoke(evidence_cli_dir: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        env = dict(**os_environ_without_pycache_noise(), PFE_HOME=str(Path(tmp) / ".pfe"))
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pfe_cli.main",
                "phase35",
                "interact",
                "--workspace",
                "phase35-cli-smoke",
                "--user-goal",
                "帮我判断下一步。",
                "--feedback-action",
                "accept",
                "--user-feedback",
                "CLI smoke simulated only.",
                "--simulated-local-interaction",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
    payload = {
        "kind": "phase35_cli_smoke",
        "command": "pfe phase35 interact --simulated-local-interaction",
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "passed": proc.returncode == 0 and "Phase35 local interaction captured" in proc.stdout,
        "actual_user_feedback_count": 0,
        "created_at": _utcnow_iso(),
    }
    _write_json(evidence_cli_dir / "cli_smoke.json", payload)
    (evidence_cli_dir / "cli_smoke_output.txt").write_text(proc.stdout + proc.stderr, encoding="utf-8")
    return payload


def os_environ_without_pycache_noise() -> dict[str, str]:
    import os

    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{PFE_CORE}:{PFE_CLI}" + (f":{existing}" if existing else "")
    return env


def _write_runbook(path: Path) -> None:
    path.write_text(
        """# Phase35 Runbook

Phase35 adds a lightweight local interaction capture lane before any Hermes integration.

## Simulated Evidence

```bash
.venv/bin/python tools/phase35_local_interaction_capture.py --clean-evidence
```

## Local CLI Capture

```bash
pfe phase35 interact \\
  --workspace personal-agent \\
  --user-goal "帮我整理当前工作区并判断下一步" \\
  --feedback-action correction \\
  --user-feedback "这次回答还是太泛，先跑真实检查。"
```

To mark a real local interaction as reviewable actual feedback, the operator must explicitly add:

```bash
--operator-id local-user \\
--confirm-actual-user-feedback \\
--consent-for-training-candidate-review \\
--not-scripted-or-curated
```

Even then, Phase35 only stores the record as pending review. Training remains blocked until Phase36 review approves it.
""",
        encoding="utf-8",
    )


def _write_final_decision(path: Path, summary: Mapping[str, Any]) -> None:
    path.write_text(
        f"""# Phase35 Final Decision

## Decision

- Final recommendation: {summary.get("final_recommendation")}
- Hermes integration used: false
- Actual training run: false
- Auto training allowed: false
- Auto promotion allowed: false
- Actual user feedback count in committed evidence: 0

## Capture

- Simulated local interaction count: {summary.get("simulated_local_interaction_count")}
- Pending review count: {summary.get("pending_review_count")}
- Training status: {summary.get("training_status")}
- Training blocked reason: {summary.get("training_blocked_reason")}

## Interpretation

Phase35 proves the lighter path is viable: PFE can capture local interactions into a durable review queue without depending on Hermes. Committed evidence stays simulated-only; real local use must be explicitly attested by the operator and still requires Phase36 review before training.
""",
        encoding="utf-8",
    )


def _write_next_goal(path: Path) -> None:
    path.write_text(
        """目标：开发并验证 PFE Phase36：本地交互 review queue + approved actual feedback candidate generation。

请在 /Users/lichenhao/Desktop/PFE 中完成：

1. 读取 Phase35 local interaction store。
2. 提供 review decision：approve_for_candidate / exclude / quarantine。
3. 只有 operator attested actual local interactions 可进入候选。
4. simulated_local_interaction 和缺 consent 的记录必须继续排除。
5. 生成 SFT/DPO candidates，但不自动训练。
6. 保存 evidence、focused tests、runbook、final decision。
7. 不接 Hermes，不修改 videos，不自动 promote。
""",
        encoding="utf-8",
    )


def generate_phase35_evidence(args: argparse.Namespace) -> dict[str, Any]:
    if args.clean_evidence:
        _clean_dir(PHASE35_DIR)
    for subdir in ("evidence", "evidence-capture", "evidence-review", "evidence-cli"):
        (PHASE35_DIR / subdir).mkdir(parents=True, exist_ok=True)
    evidence_dir = PHASE35_DIR / "evidence"
    capture_dir = PHASE35_DIR / "evidence-capture"
    review_dir = PHASE35_DIR / "evidence-review"
    cli_dir = PHASE35_DIR / "evidence-cli"

    phase34_summary = _read_json(PHASE34_DIR / "comparison_summary.json")
    phase34_review = build_phase35_phase34_review(phase34_summary=phase34_summary)
    records = _simulated_records(args.simulated_count)
    capture_batch = build_phase35_capture_batch(records)
    state_path = phase35_store_path(PHASE35_DIR / "evidence", "phase35-demo")
    state = load_phase35_state(state_path)
    # Simulated evidence is intentionally not appended as pending actual review.
    state["capture_batches"] = [dict(capture_batch)]
    save_phase35_state(state_path, state)
    review_queue = build_phase35_review_queue(state)
    readiness = build_phase35_readiness(state)
    cli_smoke = _run_cli_smoke(cli_dir)
    summary = build_phase35_comparison_summary(
        phase34_review=phase34_review,
        capture_batch=capture_batch,
        state=state,
        readiness=readiness,
    )
    summary = {
        **summary,
        "cli_smoke": cli_smoke,
        "committed_evidence_actual_user_feedback_count": 0,
    }

    _write_json(evidence_dir / "phase34_review.json", phase34_review)
    _write_json(capture_dir / "simulated_capture_batch.json", capture_batch)
    write_jsonl(capture_dir / "simulated_local_interactions.jsonl", records)
    _write_json(review_dir / "review_queue.json", review_queue)
    _write_json(review_dir / "training_readiness.json", readiness)
    _write_json(PHASE35_DIR / "comparison_summary.json", summary)
    _write_json(evidence_dir / "comparison_summary.json", summary)
    _write_runbook(PHASE35_DIR / "phase35-runbook.md")
    _write_final_decision(PHASE35_DIR / "phase35-final-decision.md", summary)
    _write_next_goal(PHASE35_DIR / "next-pursuit-goal.md")
    _redact_evidence_tree(PHASE35_DIR)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-evidence", action="store_true")
    parser.add_argument("--simulated-count", type=int, default=6)
    args = parser.parse_args()
    summary = generate_phase35_evidence(args)
    print(
        json.dumps(
            {
                "kind": summary["kind"],
                "status": summary["status"],
                "simulated_local_interaction_count": summary["simulated_local_interaction_count"],
                "pending_review_count": summary["pending_review_count"],
                "training_status": summary["training_status"],
                "final_recommendation": summary["final_recommendation"],
                "cli_smoke": summary["cli_smoke"],
                "hermes_integration_used": summary["hermes_integration_used"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
