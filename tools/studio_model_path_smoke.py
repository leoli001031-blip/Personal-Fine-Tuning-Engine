#!/usr/bin/env python3
"""Smoke-test the Studio model path and API handoff flow."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from first_run_smoke import _default_python, _repo_root, _require


def _add_repo_paths(repo_root: Path) -> None:
    for child in ("pfe-server", "pfe-core"):
        path = str(repo_root / child)
        if path not in sys.path:
            sys.path.insert(0, path)


def _write_minimal_local_model(workdir: Path) -> Path:
    model_dir = workdir / "models" / "studio-local"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["GPT2LMHeadModel"],
                "bos_token_id": 0,
                "eos_token_id": 1,
                "model_type": "gpt2",
                "pad_token_id": 0,
                "vocab_size": 32,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return model_dir.resolve()


def _request(app: Any, path: str, **kwargs: Any) -> dict[str, Any]:
    from pfe_server.app import smoke_test_request

    return asyncio.run(smoke_test_request(app, path=path, **kwargs))


def _expect_status(result: dict[str, Any], status_code: int, *, label: str) -> dict[str, Any]:
    if result["status_code"] != status_code:
        raise AssertionError(f"{label} returned {result['status_code']}:\n{result.get('text')}")
    body = result.get("body")
    return body if isinstance(body, dict) else {}


def _run_smoke(args: argparse.Namespace, workdir: Path) -> dict[str, str]:
    _add_repo_paths(args.repo_root)

    old_home = os.environ.get("PFE_HOME")
    old_workspace = os.environ.get("PFE_WORKSPACE")
    old_real_local = os.environ.get("PFE_ENABLE_REAL_LOCAL_INFERENCE")
    pfe_home = workdir / ".pfe"
    model_dir = _write_minimal_local_model(workdir)
    try:
        os.environ["PFE_HOME"] = str(pfe_home)
        os.environ["PFE_WORKSPACE"] = args.workspace
        os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)

        from pfe_server.app import build_serve_plan

        plan = build_serve_plan(host=args.host, port=args.port, workspace=args.workspace, dry_run=True)
        app = plan.app
        host_header = {"host": f"{args.host}:{args.port}"}

        studio = _request(app, "/studio", headers=host_header)
        if studio["status_code"] != 200:
            raise AssertionError(f"studio page returned {studio['status_code']}")
        _require(studio["text"], "PFE / 本地模型工作台", label="studio html")
        _require(studio["text"], "当前工作单", label="studio html")
        _require(studio["text"], "zc / PFE Studio", label="studio html")
        _require(studio["text"], "/pfe/static/studio.css", label="studio html")
        _require(studio["text"], "/pfe/static/studio.js", label="studio html")
        studio_css = _request(app, "/pfe/static/studio.css", headers=host_header)
        if studio_css["status_code"] != 200:
            raise AssertionError(f"studio css returned {studio_css['status_code']}")
        studio_js = _request(app, "/pfe/static/studio.js", headers=host_header)
        if studio_js["status_code"] != 200:
            raise AssertionError(f"studio js returned {studio_js['status_code']}")
        css_text = str(studio_css.get("text") or "")
        js_text = str(studio_js.get("text") or "")
        _require(css_text, ".work-facts", label="studio css")
        _require(css_text, ".brand-signature", label="studio css")
        _require(studio["text"], "复制 API", label="studio html")
        _require(studio["text"], "模型文件夹", label="studio html")
        _require(studio["text"], "保存模型", label="studio html")
        _require(studio["text"], "使用本地模型回复", label="studio html")
        _require(js_text, "暂停本地模型回复", label="studio js")
        _require(js_text, "先选择模型文件夹", label="studio js")
        _require(js_text, "点“使用本地模型回复”后生效", label="studio js")
        _require(js_text, "本机推理依赖未安装", label="studio js")
        _require(js_text, "演示回复", label="studio js")
        _require(studio["text"], "测试接入", label="studio html")
        _require(studio["text"], "未测试", label="studio html")
        _require(js_text, "/pfe/handoff/test", label="studio js")
        _require(studio["text"], "聊天 API", label="studio html")
        _require(studio["text"], "反馈 API", label="studio html")
        _require(studio["text"], "模型参数", label="studio html")
        _require(studio["text"], "调用示例", label="studio html")
        _require(studio["text"], "复制调用示例", label="studio html")
        _require(studio["text"], "版本生成", label="studio html")
        _require(studio["text"], "检查条件", label="studio html")
        _require(js_text, "还没有模型版本", label="studio js")
        _require(studio["text"], "最近任务", label="studio html")
        _require(studio["text"], "停止生成", label="studio html")
        _require(studio["text"], "重新生成", label="studio html")
        _require(css_text, "version-evidence", label="studio css")
        _require(js_text, "eval_summary", label="studio js")
        _require(js_text, "decision", label="studio js")
        _require(js_text, "/pfe/eval/status", label="studio js")
        _require(js_text, "评估", label="studio js")
        _require(js_text, "会成为 API 回复使用的模型版本。", label="studio js")
        _require(js_text, "归档后仍可在历史中查看，需要时可以回退。", label="studio js")
        _require(js_text, "会创建一个后台任务，完成后出现在版本列表。", label="studio js")
        _require(js_text, "/pfe/config/model", label="studio js")
        _require(js_text, "/pfe/config/real-local", label="studio js")
        _require(js_text, "/pfe/workspaces", label="studio js")
        _require(js_text, "/pfe/training/jobs", label="studio js")

        workspace_name = "studio-client"
        initial_workspaces = _expect_status(
            _request(app, "/pfe/workspaces", headers=host_header),
            200,
            label="workspaces before create",
        )
        if initial_workspaces.get("current") != args.workspace:
            raise AssertionError(f"unexpected initial workspace: {initial_workspaces}")

        workspace_save = _expect_status(
            _request(
                app,
                "/pfe/workspaces",
                method="POST",
                body={"name": workspace_name},
                headers=host_header,
            ),
            200,
            label="workspace create",
        )
        if workspace_save.get("current") != workspace_name:
            raise AssertionError(f"workspace create did not switch current workspace: {workspace_save}")

        runtime = _expect_status(
            _request(app, "/pfe/runtime", headers=host_header),
            200,
            label="runtime",
        )
        expected_api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
        expected_feedback_url = f"http://{args.host}:{args.port}/pfe/feedback"
        if runtime.get("api_url") != expected_api_url:
            raise AssertionError(f"unexpected api_url: {runtime}")
        api_contract = runtime.get("api") if isinstance(runtime.get("api"), dict) else {}
        if api_contract.get("chat_completions_url") != expected_api_url:
            raise AssertionError(f"unexpected api handoff contract: {runtime}")
        if api_contract.get("feedback_url") != expected_feedback_url:
            raise AssertionError(f"unexpected feedback handoff contract: {runtime}")
        if api_contract.get("response_id_fields") != ["session_id", "request_id"]:
            raise AssertionError(f"handoff contract did not expose response ids: {runtime}")
        if "accept" not in set(api_contract.get("feedback_actions") or []):
            raise AssertionError(f"handoff contract did not expose feedback actions: {runtime}")
        if api_contract.get("model_parameter") != "base":
            raise AssertionError(f"unexpected api model parameter: {runtime}")
        if "local" not in set(api_contract.get("model_aliases") or []):
            raise AssertionError(f"handoff contract did not expose local model alias: {runtime}")
        if api_contract.get("request_body", {}).get("model") != "base":
            raise AssertionError(f"unexpected api request example: {runtime}")
        if runtime.get("workspace") != workspace_name:
            raise AssertionError(f"runtime did not switch workspace: {runtime}")

        jobs_before = _expect_status(
            _request(app, "/pfe/training/jobs", headers=host_header),
            200,
            label="training jobs before preflight",
        )
        if jobs_before.get("items") != [] or jobs_before.get("latest") is not None:
            raise AssertionError(f"unexpected initial training jobs: {jobs_before}")

        before = _expect_status(
            _request(app, "/pfe/readiness", headers=host_header),
            200,
            label="readiness before model save",
        )
        before_blockers = set(before.get("summary", {}).get("blockers") or [])
        if "needs_local_path" not in before_blockers:
            raise AssertionError(f"readiness did not ask for a local model path: {before}")

        saved = _expect_status(
            _request(
                app,
                "/pfe/config/model",
                method="PUT",
                body={"base_model": str(model_dir)},
                headers=host_header,
            ),
            200,
            label="model config save",
        )
        if saved.get("selected") != str(model_dir):
            raise AssertionError(f"saved selected model did not match local path: {saved}")
        if saved.get("effective_scope") != "next_chat_request" or saved.get("reload_required") is not False:
            raise AssertionError(f"unexpected model config apply contract: {saved}")

        models = _expect_status(_request(app, "/pfe/models", headers=host_header), 200, label="models")
        if models.get("selected") != str(model_dir):
            raise AssertionError(f"models endpoint did not retain selected local path: {models}")

        readiness = _expect_status(
            _request(app, "/pfe/readiness", headers=host_header),
            200,
            label="readiness after model save",
        )
        model_source = readiness.get("model", {}).get("source", {})
        if model_source.get("state") != "ready" or model_source.get("path") != str(model_dir):
            raise AssertionError(f"readiness did not recognize local model path: {readiness}")
        after_blockers = set(readiness.get("summary", {}).get("blockers") or [])
        if "needs_local_path" in after_blockers:
            raise AssertionError(f"readiness still reports missing local model path: {readiness}")

        toggled = _expect_status(
            _request(
                app,
                "/pfe/config/real-local",
                method="PUT",
                body={"enabled": True},
                headers=host_header,
            ),
            200,
            label="real-local toggle",
        )
        if toggled.get("enabled") is not True or toggled.get("reload_required") is not False:
            raise AssertionError(f"real-local toggle did not enable current service process: {toggled}")
        readiness = toggled.get("readiness") if isinstance(toggled.get("readiness"), dict) else {}
        after_toggle_blockers = set(readiness.get("summary", {}).get("blockers") or [])
        if "real_local_inference_disabled" in after_toggle_blockers:
            raise AssertionError(f"readiness still reports disabled real-local inference: {readiness}")

        training_preflight = _expect_status(
            _request(
                app,
                "/pfe/training/jobs",
                method="POST",
                body={"method": "sft", "epochs": 1},
                headers=host_header,
            ),
            409,
            label="training preflight",
        )
        if training_preflight.get("code") != "confirmation_required":
            raise AssertionError(f"training preflight did not require confirmation: {training_preflight}")
        if training_preflight.get("kind") != "pfe_training_preflight_required":
            raise AssertionError(f"training preflight did not expose typed response kind: {training_preflight}")
        request = training_preflight.get("request") if isinstance(training_preflight.get("request"), dict) else {}
        if request.get("method") != "sft" or request.get("confirmed") is not False:
            raise AssertionError(f"training preflight did not expose typed request: {training_preflight}")
        preflight = training_preflight.get("preflight") if isinstance(training_preflight.get("preflight"), dict) else {}
        if preflight.get("kind") != "pfe_training_preflight":
            raise AssertionError(f"training preflight did not expose typed preflight kind: {training_preflight}")
        if preflight.get("base_model") != str(model_dir):
            raise AssertionError(f"training preflight did not use saved model path: {training_preflight}")
        if preflight.get("requires_confirmation") is not True:
            raise AssertionError(f"training preflight did not expose confirmation contract: {training_preflight}")
        if "job_id" in training_preflight:
            raise AssertionError(f"training preflight unexpectedly created a job: {training_preflight}")
        jobs_after_preflight = _expect_status(
            _request(app, "/pfe/training/jobs", headers=host_header),
            200,
            label="training jobs after preflight",
        )
        if jobs_after_preflight.get("items") != [] or jobs_after_preflight.get("active") is not None:
            raise AssertionError(f"training preflight unexpectedly changed job list: {jobs_after_preflight}")

        chat = _expect_status(
            _request(
                app,
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "hello from studio smoke"}],
                    "metadata": {"source": "studio_model_path_smoke"},
                },
                headers=host_header,
            ),
            200,
            label="chat completion",
        )
        inference = chat.get("metadata", {}).get("inference", {})
        if inference.get("resolved_base_model") != str(model_dir):
            raise AssertionError(f"chat did not use the saved local model path: {chat}")
        if inference.get("real_local_enabled") is not True:
            raise AssertionError(f"chat did not enter the real-local attempt path: {chat}")

        handoff_test = _expect_status(
            _request(
                app,
                "/pfe/handoff/test",
                method="POST",
                body={"message": "hello from studio handoff test", "action": "accept"},
                headers=host_header,
            ),
            200,
            label="handoff test",
        )
        if handoff_test.get("ok") is not True:
            raise AssertionError(f"handoff test did not pass: {handoff_test}")
        if handoff_test.get("contract", {}).get("feedback_url") != expected_feedback_url:
            raise AssertionError(f"handoff test did not expose feedback url: {handoff_test}")
        if not handoff_test.get("chat", {}).get("request_id"):
            raise AssertionError(f"handoff test did not return request id: {handoff_test}")
        if handoff_test.get("feedback", {}).get("signal_type") != "accept":
            raise AssertionError(f"handoff test did not record accept feedback: {handoff_test}")

        return {
            "workspace": workspace_name,
            "pfe_home": str(pfe_home),
            "base_model": str(model_dir),
            "api_url": str(runtime.get("api_url") or ""),
            "api_model": str(api_contract.get("model_parameter") or ""),
            "inference_mode": str(readiness.get("inference", {}).get("mode") or ""),
            "real_local_enabled": str(readiness.get("inference", {}).get("real_local_enabled") or ""),
            "training_preflight": str(preflight.get("ready")),
            "training_jobs": str(jobs_after_preflight.get("count")),
            "chat_served_by": str(chat.get("served_by") or ""),
            "chat_resolved_model": str(inference.get("resolved_base_model") or ""),
            "handoff_test": str(handoff_test.get("ok")),
        }
    finally:
        if old_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = old_home
        if old_workspace is None:
            os.environ.pop("PFE_WORKSPACE", None)
        else:
            os.environ["PFE_WORKSPACE"] = old_workspace
        if old_real_local is None:
            os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
        else:
            os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = old_real_local


def main() -> int:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the PFE Studio user path: serve Studio, save a local model path, "
            "verify readiness, then call the OpenAI-compatible chat API with model=local."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=_default_python(repo_root), help="Kept for Makefile/test parity.")
    parser.add_argument("--workspace", default="studio_model_path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8921)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-workdir", action="store_true")
    args = parser.parse_args()
    args.repo_root = args.repo_root.resolve()
    del args.python

    tempdir = None
    if args.workdir is None:
        tempdir = tempfile.TemporaryDirectory(prefix="pfe-studio-model-path-")
        workdir = Path(tempdir.name)
    else:
        workdir = args.workdir.resolve()
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    print(f"workdir: {workdir}")
    print()
    try:
        summary = _run_smoke(args, workdir)
        print("STUDIO MODEL PATH SMOKE PASSED")
        print(f"workspace:           {summary['workspace']}")
        print(f"pfe_home:            {summary['pfe_home']}")
        print(f"base_model:          {summary['base_model']}")
        print(f"api_url:             {summary['api_url']}")
        print(f"api_model:           {summary['api_model']}")
        print(f"inference_mode:      {summary['inference_mode']}")
        print(f"real_local_enabled:  {summary['real_local_enabled']}")
        print(f"training_preflight:  {summary['training_preflight']}")
        print(f"training_jobs:       {summary['training_jobs']}")
        print(f"chat_served_by:      {summary['chat_served_by']}")
        print(f"chat_resolved_model: {summary['chat_resolved_model']}")
        return 0
    finally:
        if tempdir is not None and not args.keep_workdir:
            tempdir.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
