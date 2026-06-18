from __future__ import annotations

import asyncio
import importlib
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.config import PFEConfig
from pfe_server.app import build_serve_plan, create_app, smoke_test_request

server_app = importlib.import_module("pfe_server.app")

class ServerHttpSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous_home = os.environ.get("PFE_HOME")
        self.pfe_home = Path(self.tempdir.name) / ".pfe"
        os.environ["PFE_HOME"] = str(self.pfe_home)
        self.plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
        self.app = self.plan.app

    def tearDown(self) -> None:
        if self.previous_home is None:
            os.environ.pop("PFE_HOME", None)
        else:
            os.environ["PFE_HOME"] = self.previous_home
        self.tempdir.cleanup()

    def _smoke(self, path: str, **kwargs):
        return asyncio.run(smoke_test_request(self.app, path=path, **kwargs))

    def _studio_assets(self) -> tuple[str, str]:
        css = self._smoke("/pfe/static/studio.css")
        js = self._smoke("/pfe/static/studio.js")
        self.assertEqual(css["status_code"], 200)
        self.assertEqual(js["status_code"], 200)
        self.assertIn("text/css", css["headers"].get("content-type", ""))
        self.assertIn("javascript", js["headers"].get("content-type", ""))
        return css["text"], js["text"]

    def _server_adapter_store(self) -> AdapterStore:
        self.app.state.pfe_services.workspace = "user_default"
        return AdapterStore(home=self.pfe_home, workspace="user_default")

    def _create_pending_adapter(self, store: AdapterStore, base_model: str = "base") -> str:
        created = store.create_training_version(
            base_model=base_model,
            training_config={"backend": "mock_local", "train_type": "sft"},
        )
        version = str(created["version"])
        store.mark_pending_eval(version, num_samples=3, metrics={"loss": 0.1})
        return version

    def test_healthz_returns_ok(self) -> None:
        result = self._smoke("/healthz")
        self.assertEqual(result["status_code"], 200)
        self.assertEqual(result["body"]["status"], "ok")
        self.assertIn("content-type", result["headers"])

    def test_root_serves_studio_frontend(self) -> None:
        result = self._smoke("/")
        self.assertEqual(result["status_code"], 200)
        self.assertIn("text/html", result["headers"].get("content-type", ""))
        self.assertIn("PFE / 本地模型工作台", result["text"])
        self.assertIn("本地版本可验证、可接入。", result["text"])
        self.assertIn("当前工作单", result["text"])
        self.assertIn("效果证据", result["text"])
        self.assertIn("zc / PFE Studio", result["text"])
        self.assertIn("/pfe/static/studio.css", result["text"])
        self.assertIn("/pfe/static/studio.js", result["text"])
        _css, js = self._studio_assets()
        self.assertIn("/pfe/runtime", js)
        self.assertIn("/pfe/models", js)
        self.assertIn("/pfe/handoff", js)

    def test_chat_alias_serves_legacy_operations_frontend(self) -> None:
        result = self._smoke("/chat")
        self.assertEqual(result["status_code"], 200)
        self.assertIn("text/html", result["headers"].get("content-type", ""))
        self.assertIn("Worker Daemon", result["text"])
        self.assertIn("Train Queue", result["text"])
        self.assertIn("/pfe/auto-train/run-worker-runner", result["text"])

        alias = self._smoke("/pfe/chat")
        self.assertEqual(alias["status_code"], 200)
        self.assertIn("Worker Daemon", alias["text"])

    def test_studio_frontend_serves_user_facing_control_surface(self) -> None:
        result = self._smoke("/studio")
        self.assertEqual(result["status_code"], 200)
        self.assertIn("text/html", result["headers"].get("content-type", ""))
        self.assertIn("PFE / 本地模型工作台", result["text"])
        self.assertIn("本地版本可验证、可接入。", result["text"])
        self.assertIn("当前工作单", result["text"])
        self.assertIn("DEMO-PHASE2-042", result["text"])
        self.assertIn("zc / PFE Studio", result["text"])
        self.assertIn("/pfe/static/studio.css", result["text"])
        self.assertIn("/pfe/static/studio.js", result["text"])
        css, js = self._studio_assets()
        self.assertIn(".work-facts", css)
        self.assertIn(".proof-panel", css)
        self.assertIn(".brand-signature", css)
        self.assertIn("/pfe/runtime", js)
        self.assertIn("/pfe/models", js)
        self.assertIn("/pfe/adapters", js)
        self.assertIn("/pfe/readiness", js)
        self.assertIn("/pfe/workspaces", js)
        self.assertIn("/pfe/handoff", js)
        self.assertIn("当前回复", result["text"])
        self.assertIn("复制 API", result["text"])
        self.assertIn("复制 API 地址", result["text"])
        self.assertIn("接入信息", result["text"])
        self.assertIn("复制接入信息", result["text"])
        self.assertIn("测试接入", result["text"])
        self.assertIn("未测试", result["text"])
        self.assertIn("聊天 API", result["text"])
        self.assertIn("反馈 API", result["text"])
        self.assertIn("模型参数", result["text"])
        self.assertIn("调用示例", result["text"])
        self.assertIn("复制调用示例", result["text"])
        self.assertIn("版本生成", result["text"])
        self.assertIn("基座", result["text"])
        self.assertIn("当前", result["text"])
        self.assertIn("待评估", result["text"])
        self.assertIn("已加载", result["text"])
        self.assertIn("检查条件", result["text"])
        self.assertIn("还没有模型版本", js)
        self.assertIn("最近任务", result["text"])
        self.assertIn("停止生成", result["text"])
        self.assertIn("重新生成", result["text"])
        self.assertIn("version-evidence", css)
        self.assertIn("eval_summary", js)
        self.assertIn("decision", js)
        self.assertIn("promotion_gate", js)
        self.assertIn("上线闸门：等待评估通过", js)
        self.assertIn("/pfe/eval/status", js)
        self.assertIn("评估", js)
        self.assertIn("会成为 API 回复使用的模型版本。", js)
        self.assertIn("归档后仍可在历史中查看，需要时可以回退。", js)
        self.assertIn("会创建一个后台任务，完成后出现在版本列表。", js)
        self.assertIn("/pfe/training/jobs", js)
        self.assertIn("工作区", result["text"])
        self.assertIn("创建并切换", result["text"])
        self.assertIn("模型文件夹", result["text"])
        self.assertIn("保存模型", result["text"])
        self.assertIn("本地回复", result["text"])
        self.assertIn("暂停回复", js)
        self.assertIn("先选择模型文件夹", js)
        self.assertIn("点“本地回复”后生效", js)
        self.assertIn("本机推理依赖未安装", js)
        self.assertIn("演示回复", js)
        self.assertNotIn("开启真实本地模型", result["text"] + js)
        self.assertIn("/pfe/handoff/test", js)
        self.assertIn("/pfe/config/real-local", js)

        alias = self._smoke("/pfe/studio")
        self.assertEqual(alias["status_code"], 200)
        self.assertIn("PFE / 本地模型工作台", alias["text"])

    def test_studio_handoff_surface_exposes_copyable_user_contract(self) -> None:
        config = PFEConfig()
        config.model.base_model = "Qwen/Qwen3-4B"
        config.save(home=self.pfe_home)
        store = self._server_adapter_store()
        version = self._create_pending_adapter(store, base_model="Qwen/Qwen3-4B")
        store.attach_eval_report(version, {"recommendation": "deploy", "comparison": "improved", "scores": {}})
        store.promote(version)

        result = self._smoke("/pfe/handoff", headers={"host": "127.0.0.1:9012"})

        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertEqual(body["kind"], "pfe_studio_handoff")
        self.assertEqual(body["workspace"], "user_default")
        self.assertEqual(body["urls"]["web"], "http://127.0.0.1:9012/")
        self.assertEqual(body["urls"]["api"], "http://127.0.0.1:9012/v1/chat/completions")
        self.assertEqual(body["urls"]["feedback"], "http://127.0.0.1:9012/pfe/feedback")
        self.assertEqual(body["model"]["selected"], "Qwen/Qwen3-4B")
        self.assertEqual(body["model"]["api_parameter"], "base")
        self.assertIn("local-default", body["model"]["aliases"])
        self.assertEqual(body["version"]["current"], version)
        self.assertEqual(body["version"]["latest"], version)
        self.assertEqual(body["version"]["count"], 1)
        self.assertEqual(body["closed_loop"]["chat"]["url"], "http://127.0.0.1:9012/v1/chat/completions")
        self.assertEqual(body["closed_loop"]["feedback"]["url"], "http://127.0.0.1:9012/pfe/feedback")
        self.assertEqual(body["closed_loop"]["required_response_fields"], ["session_id", "request_id"])
        self.assertIn("accept", body["closed_loop"]["feedback"]["actions"])
        self.assertIn("session_id", body["snippets"]["javascript"])
        self.assertIn("requests.post", body["snippets"]["python"])
        self.assertIn("Web: http://127.0.0.1:9012/", body["copy_text"])
        self.assertIn("Chat API: http://127.0.0.1:9012/v1/chat/completions", body["copy_text"])
        self.assertIn("Feedback API: http://127.0.0.1:9012/pfe/feedback", body["copy_text"])
        self.assertIn("Keep per answer: session_id, request_id", body["copy_text"])
        self.assertIn("Model parameter: base", body["copy_text"])
        self.assertIn(f"Current version: {version}", body["copy_text"])

    def test_studio_handoff_test_runs_chat_and_feedback_loop(self) -> None:
        result = self._smoke(
            "/pfe/handoff/test",
            method="POST",
            body={"message": "hello from handoff test", "action": "accept"},
            headers={"host": "127.0.0.1:9012"},
        )

        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertEqual(body["kind"], "pfe_handoff_test")
        self.assertTrue(body["ok"])
        self.assertEqual(body["contract"]["chat_url"], "http://127.0.0.1:9012/v1/chat/completions")
        self.assertEqual(body["contract"]["feedback_url"], "http://127.0.0.1:9012/pfe/feedback")
        self.assertEqual(body["contract"]["response_id_fields"], ["session_id", "request_id"])
        self.assertTrue(body["chat"]["ok"])
        self.assertTrue(body["chat"]["session_id"])
        self.assertTrue(body["chat"]["request_id"])
        self.assertTrue(body["feedback"]["ok"])
        self.assertEqual(body["feedback"]["signal_type"], "accept")
        self.assertEqual(body["feedback"]["pending_found"], True)

    def test_studio_runtime_models_and_adapters_surfaces(self) -> None:
        config = PFEConfig()
        config.model.base_model = "Qwen/Qwen2.5-3B-Instruct"
        config.save(home=self.pfe_home)

        runtime = self._smoke("/pfe/runtime", headers={"host": "127.0.0.1:9012"})
        self.assertEqual(runtime["status_code"], 200)
        self.assertEqual(runtime["body"]["workspace"], str(self.pfe_home))
        self.assertEqual(runtime["body"]["web_url"], "http://127.0.0.1:9012/")
        self.assertEqual(runtime["body"]["api_url"], "http://127.0.0.1:9012/v1/chat/completions")
        self.assertEqual(runtime["body"]["studio_url"], "http://127.0.0.1:9012/studio")
        self.assertEqual(runtime["body"]["api"]["kind"], "openai_chat_completions")
        self.assertEqual(runtime["body"]["api"]["method"], "POST")
        self.assertEqual(runtime["body"]["api"]["chat_completions_url"], "http://127.0.0.1:9012/v1/chat/completions")
        self.assertEqual(runtime["body"]["api"]["feedback_url"], "http://127.0.0.1:9012/pfe/feedback")
        self.assertEqual(runtime["body"]["api"]["model_parameter"], "base")
        self.assertIn("local-default", runtime["body"]["api"]["model_aliases"])
        self.assertEqual(runtime["body"]["api"]["response_id_fields"], ["session_id", "request_id"])
        self.assertIn("accept", runtime["body"]["api"]["feedback_actions"])
        self.assertEqual(runtime["body"]["api"]["request_body"]["model"], "base")
        self.assertEqual(runtime["body"]["api"]["feedback_body"]["action"], "accept")
        self.assertEqual(runtime["body"]["access_scope"], "仅本机")
        self.assertIn("privacy_mode", runtime["body"])

        models = self._smoke("/pfe/models")
        self.assertEqual(models["status_code"], 200)
        self.assertEqual(models["body"]["selected"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(models["body"]["mode"], "configurable")
        self.assertGreaterEqual(models["body"]["count"], 1)
        selected = models["body"]["candidates"][0]
        self.assertEqual(selected["id"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertIsNone(selected["local_path"])
        self.assertIsNone(selected["exists"])

        adapters = self._smoke("/pfe/adapters")
        self.assertEqual(adapters["status_code"], 200)
        self.assertIn("versions", adapters["body"])
        self.assertIn("count", adapters["body"])
        self.assertIn("pending", adapters["body"])
        self.assertIn("latest_version", adapters["body"])
        self.assertIn("base_model", adapters["body"])
        self.assertIn("latest_adapter", adapters["body"])
        self.assertIn("pending_eval_adapter", adapters["body"])
        self.assertIn("adapter_loaded", adapters["body"])

        workspaces = self._smoke("/pfe/workspaces")
        self.assertEqual(workspaces["status_code"], 200)
        self.assertEqual(workspaces["body"]["current"], str(self.pfe_home))
        self.assertIn("items", workspaces["body"])
        self.assertIn("create_api", workspaces["body"])

    def test_studio_workspaces_create_and_switch_current_process(self) -> None:
        previous_workspace = os.environ.get("PFE_WORKSPACE")
        try:
            os.environ.pop("PFE_WORKSPACE", None)
            created = self._smoke(
                "/pfe/workspaces",
                method="POST",
                body={"name": "client-a"},
            )
            self.assertEqual(created["status_code"], 200)
            self.assertTrue(created["body"]["saved"])
            self.assertTrue(created["body"]["created"])
            self.assertTrue(created["body"]["switched"])
            self.assertEqual(created["body"]["previous"], str(self.pfe_home))
            self.assertEqual(created["body"]["current"], "client-a")
            self.assertEqual(created["body"]["runtime"]["workspace"], "client-a")
            self.assertEqual(os.environ.get("PFE_WORKSPACE"), "client-a")
            self.assertTrue((self.pfe_home / "adapters" / "client-a").is_dir())
            self.assertTrue((self.pfe_home / "workspaces" / "client-a").is_dir())
            self.assertEqual(created["body"]["workspaces"]["current"], "client-a")
            self.assertEqual(created["body"]["readiness"]["runtime"]["service"]["workspace"], "client-a")

            listed = self._smoke("/pfe/workspaces")
            self.assertEqual(listed["status_code"], 200)
            self.assertEqual(listed["body"]["current"], "client-a")
            self.assertIn("client-a", {item["id"] for item in listed["body"]["items"]})

            same = self._smoke(
                "/pfe/workspaces",
                method="POST",
                body={"name": "client-a"},
            )
            self.assertEqual(same["status_code"], 200)
            self.assertFalse(same["body"]["created"])
            self.assertFalse(same["body"]["changed"])

            invalid = self._smoke(
                "/pfe/workspaces",
                method="POST",
                body={"name": "../bad"},
            )
            self.assertEqual(invalid["status_code"], 422)
            self.assertFalse(invalid["body"]["saved"])
            self.assertEqual(invalid["body"]["validation"]["issues"][0]["code"], "workspace_invalid_chars")
        finally:
            self.app.state.pfe_services.workspace = str(self.pfe_home)
            if previous_workspace is None:
                os.environ.pop("PFE_WORKSPACE", None)
            else:
                os.environ["PFE_WORKSPACE"] = previous_workspace

    def test_default_app_reads_workspace_from_environment(self) -> None:
        previous_workspace = os.environ.get("PFE_WORKSPACE")
        try:
            os.environ["PFE_WORKSPACE"] = "env-start"
            app = create_app()
            runtime = asyncio.run(smoke_test_request(app, "/pfe/runtime"))
            self.assertEqual(runtime["status_code"], 200)
            self.assertEqual(runtime["body"]["workspace"], "env-start")
        finally:
            if previous_workspace is None:
                os.environ.pop("PFE_WORKSPACE", None)
            else:
                os.environ["PFE_WORKSPACE"] = previous_workspace

    def test_studio_readiness_reports_template_mode_until_real_local_is_enabled(self) -> None:
        previous_real_local = os.environ.get("PFE_ENABLE_REAL_LOCAL_INFERENCE")
        try:
            os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
            config = PFEConfig()
            config.model.base_model = "Qwen/Qwen2.5-3B-Instruct"
            config.save(home=self.pfe_home)

            readiness = self._smoke("/pfe/readiness", headers={"host": "127.0.0.1:9012"})
            self.assertEqual(readiness["status_code"], 200)
            body = readiness["body"]
            self.assertEqual(body["summary"]["label"], "需确认")
            self.assertEqual(body["inference"]["mode"], "template")
            self.assertEqual(body["inference"]["mode_label"], "模板回复")
            self.assertFalse(body["inference"]["real_local_enabled"])
            self.assertFalse(body["model"]["source"]["ok"])
            self.assertEqual(body["model"]["source"]["state"], "needs_local_path")
            self.assertEqual(body["runtime"]["service"]["api_url"], "http://127.0.0.1:9012/v1/chat/completions")
            self.assertIn("runtime_dependencies", body["checks"])
            self.assertIn(
                "enable_real_local_inference",
                {item["id"] for item in body["next_actions"]},
            )
        finally:
            if previous_real_local is None:
                os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
            else:
                os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = previous_real_local

    def test_studio_readiness_recognizes_existing_local_model_source(self) -> None:
        previous_real_local = os.environ.get("PFE_ENABLE_REAL_LOCAL_INFERENCE")
        try:
            os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"
            local_model = self.pfe_home / "models" / "tiny-local"
            local_model.mkdir(parents=True)
            config = PFEConfig()
            config.model.base_model = str(local_model)
            config.save(home=self.pfe_home)

            readiness = self._smoke("/pfe/readiness")
            self.assertEqual(readiness["status_code"], 200)
            body = readiness["body"]
            self.assertTrue(body["inference"]["real_local_enabled"])
            self.assertEqual(body["model"]["source"]["state"], "ready")
            self.assertTrue(body["model"]["source"]["ok"])
            self.assertEqual(body["model"]["source"]["path"], str(local_model))
            self.assertIn(body["inference"]["mode"], {"real_local", "template"})
            self.assertIn("runtime", body)
            self.assertIn("dependencies", body["runtime"])
        finally:
            if previous_real_local is None:
                os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
            else:
                os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = previous_real_local

    def test_studio_model_config_update_saves_base_model(self) -> None:
        config = PFEConfig()
        config.model.base_model = "Qwen/Qwen2.5-3B-Instruct"
        config.save(home=self.pfe_home)

        preview = self._smoke(
            "/pfe/config/model",
            method="PUT",
            body={"base_model": "Qwen/Qwen3-4B"},
            query_params={"validate_only": "true"},
        )
        self.assertEqual(preview["status_code"], 200)
        self.assertFalse(preview["body"]["saved"])
        self.assertTrue(preview["body"]["validate_only"])
        self.assertTrue(preview["body"]["changed"])
        self.assertEqual(preview["body"]["effective_scope"], "not_applied")
        self.assertFalse(preview["body"]["reload_required"])
        self.assertEqual(preview["body"]["selected"], "Qwen/Qwen3-4B")
        self.assertEqual(PFEConfig.load(home=self.pfe_home).model.base_model, "Qwen/Qwen2.5-3B-Instruct")

        saved = self._smoke(
            "/pfe/config/model",
            method="PUT",
            body={"base_model": "Qwen/Qwen3-4B"},
        )
        self.assertEqual(saved["status_code"], 200)
        self.assertTrue(saved["body"]["saved"])
        self.assertTrue(saved["body"]["changed"])
        self.assertEqual(saved["body"]["effective_scope"], "next_chat_request")
        self.assertFalse(saved["body"]["reload_required"])
        self.assertEqual(saved["body"]["applies_to_models"], ["local", "local-default", "base"])
        self.assertEqual(saved["body"]["previous"], "Qwen/Qwen2.5-3B-Instruct")
        self.assertEqual(saved["body"]["models"]["selected"], "Qwen/Qwen3-4B")
        self.assertEqual(PFEConfig.load(home=self.pfe_home).model.base_model, "Qwen/Qwen3-4B")

        readiness = self._smoke("/pfe/readiness")
        self.assertEqual(readiness["status_code"], 200)
        self.assertEqual(readiness["body"]["configuration"]["base_model"], "Qwen/Qwen3-4B")
        self.assertEqual(readiness["body"]["configuration"]["effective_scope"], "next_chat_request")
        self.assertFalse(readiness["body"]["configuration"]["reload_required"])
        self.assertEqual(readiness["body"]["configuration"]["applies_to_models"], ["local", "local-default", "base"])
        self.assertTrue(readiness["body"]["checks"]["model_configuration"]["ok"])

        saved_again = self._smoke(
            "/pfe/config/model",
            method="PUT",
            body={"base_model": "Qwen/Qwen3-4B"},
        )
        self.assertEqual(saved_again["status_code"], 200)
        self.assertFalse(saved_again["body"]["changed"])
        self.assertFalse(saved_again["body"]["reload_required"])

        local_model = self.pfe_home / "models" / "Qwen3-4B"
        local_model.mkdir(parents=True)
        local_saved = self._smoke(
            "/pfe/config/model",
            method="PUT",
            body={"base_model": str(local_model)},
        )
        self.assertEqual(local_saved["status_code"], 200)
        self.assertTrue(local_saved["body"]["changed"])
        self.assertFalse(local_saved["body"]["reload_required"])
        self.assertEqual(local_saved["body"]["models"]["selected"], str(local_model))

        local_readiness = self._smoke("/pfe/readiness")
        self.assertEqual(local_readiness["status_code"], 200)
        self.assertEqual(local_readiness["body"]["model"]["source"]["state"], "ready")
        self.assertEqual(local_readiness["body"]["model"]["source"]["path"], str(local_model))

    def test_studio_model_config_update_rejects_missing_local_model_path(self) -> None:
        result = self._smoke(
            "/pfe/config/model",
            method="PUT",
            body={"base_model": str(self.pfe_home / "missing-model")},
        )
        self.assertEqual(result["status_code"], 422)
        self.assertFalse(result["body"]["saved"])
        self.assertEqual(result["body"]["validation"]["issues"][0]["code"], "model_path_not_found")

    def test_studio_real_local_toggle_updates_current_process_readiness(self) -> None:
        previous_real_local = os.environ.get("PFE_ENABLE_REAL_LOCAL_INFERENCE")
        try:
            os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
            local_model = self.pfe_home / "models" / "tiny-local"
            local_model.mkdir(parents=True)
            config = PFEConfig()
            config.model.base_model = str(local_model)
            config.save(home=self.pfe_home)

            before = self._smoke("/pfe/readiness")
            self.assertEqual(before["status_code"], 200)
            self.assertFalse(before["body"]["inference"]["real_local_enabled"])
            self.assertIn("real_local_inference_disabled", before["body"]["summary"]["blockers"])

            enabled = self._smoke(
                "/pfe/config/real-local",
                method="PUT",
                body={"enabled": True},
            )
            self.assertEqual(enabled["status_code"], 200)
            self.assertTrue(enabled["body"]["saved"])
            self.assertFalse(enabled["body"]["previous"])
            self.assertTrue(enabled["body"]["enabled"])
            self.assertTrue(enabled["body"]["changed"])
            self.assertFalse(enabled["body"]["persisted"])
            self.assertEqual(enabled["body"]["effective_scope"], "current_process_next_request")
            self.assertFalse(enabled["body"]["reload_required"])
            self.assertTrue(enabled["body"]["readiness"]["inference"]["real_local_enabled"])
            self.assertNotIn("real_local_inference_disabled", enabled["body"]["readiness"]["summary"]["blockers"])
            self.assertTrue(enabled["body"]["readiness"]["checks"]["real_local_flag"]["ok"])

            chat = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )
            self.assertEqual(chat["status_code"], 200)
            self.assertTrue(chat["body"]["metadata"]["inference"]["real_local_enabled"])
            self.assertEqual(chat["body"]["metadata"]["inference"]["resolved_base_model"], str(local_model))

            disabled = self._smoke(
                "/pfe/config/real-local",
                method="PUT",
                body={"enabled": False},
            )
            self.assertEqual(disabled["status_code"], 200)
            self.assertFalse(disabled["body"]["enabled"])
            self.assertFalse(disabled["body"]["readiness"]["inference"]["real_local_enabled"])
        finally:
            if previous_real_local is None:
                os.environ.pop("PFE_ENABLE_REAL_LOCAL_INFERENCE", None)
            else:
                os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = previous_real_local

    def test_studio_adapter_promote_requires_confirmation_and_uses_requested_version(self) -> None:
        store = self._server_adapter_store()
        version = self._create_pending_adapter(store)

        missing_confirmation = self._smoke(
            f"/pfe/adapters/{version}/promote",
            method="POST",
        )
        self.assertEqual(missing_confirmation["status_code"], 409)
        self.assertEqual(missing_confirmation["body"]["code"], "confirmation_required")
        self.assertIsNone(store.current_latest_version())

        blocked = self._smoke(
            f"/pfe/adapters/{version}/promote",
            method="POST",
            body={"confirm": True},
        )
        self.assertEqual(blocked["status_code"], 409)
        self.assertEqual(blocked["body"]["code"], "promotion_eval_required")
        self.assertEqual(blocked["body"]["promotion_gate"]["reason"], "eval_required")
        self.assertIsNone(store.current_latest_version())

        store.attach_eval_report(
            version,
            {
                "recommendation": "deploy",
                "comparison": "improved",
                "scores": {"quality_preservation": 1.0},
            },
        )
        promoted = self._smoke(
            f"/pfe/adapters/{version}/promote",
            method="POST",
            body={"confirm": True},
        )
        self.assertEqual(promoted["status_code"], 200)
        self.assertTrue(promoted["body"]["success"])
        self.assertEqual(promoted["body"]["action"], "promote")
        self.assertEqual(promoted["body"]["version"], version)
        self.assertEqual(promoted["body"]["current_version"], version)
        self.assertEqual(store.current_latest_version(), version)
        self.assertEqual(promoted["body"]["adapters"]["current"]["version"], version)

    def test_studio_adapter_rollback_restores_archived_previous_version(self) -> None:
        store = self._server_adapter_store()
        first = self._create_pending_adapter(store, base_model="base-a")
        store.attach_eval_report(first, {"recommendation": "deploy", "comparison": "improved", "scores": {}})
        store.promote(first)
        second = self._create_pending_adapter(store, base_model="base-b")
        store.attach_eval_report(second, {"recommendation": "deploy", "comparison": "improved", "scores": {}})
        store.promote(second)
        self.assertEqual(store.current_latest_version(), second)
        self.assertEqual(
            next(row for row in store.list_version_records(limit=10) if row["version"] == first)["state"],
            "archived",
        )

        rolled_back = self._smoke(
            f"/pfe/adapters/{first}/rollback",
            method="POST",
            body={"confirm": True},
        )
        self.assertEqual(rolled_back["status_code"], 200)
        self.assertTrue(rolled_back["body"]["success"])
        self.assertEqual(rolled_back["body"]["action"], "rollback")
        self.assertEqual(rolled_back["body"]["previous_version"], second)
        self.assertEqual(rolled_back["body"]["current_version"], first)
        self.assertEqual(store.current_latest_version(), first)

    def test_studio_adapter_archive_requires_confirmation_and_blocks_current(self) -> None:
        store = self._server_adapter_store()
        current = self._create_pending_adapter(store, base_model="base-a")
        store.attach_eval_report(current, {"recommendation": "deploy", "comparison": "improved", "scores": {}})
        store.promote(current)
        candidate = self._create_pending_adapter(store, base_model="base-b")

        missing_confirmation = self._smoke(
            f"/pfe/adapters/{candidate}/archive",
            method="POST",
        )
        self.assertEqual(missing_confirmation["status_code"], 409)
        self.assertEqual(missing_confirmation["body"]["code"], "confirmation_required")

        archived = self._smoke(
            f"/pfe/adapters/{candidate}/archive",
            method="POST",
            body={"confirm": True},
        )
        self.assertEqual(archived["status_code"], 200)
        self.assertTrue(archived["body"]["success"])
        self.assertEqual(archived["body"]["action"], "archive")
        self.assertEqual(
            next(row for row in store.list_version_records(limit=10) if row["version"] == candidate)["state"],
            "archived",
        )

        blocked = self._smoke(
            f"/pfe/adapters/{current}/archive",
            method="POST",
            body={"confirm": True},
        )
        self.assertEqual(blocked["status_code"], 409)
        self.assertEqual(blocked["body"]["code"], "adapter_action_failed")

    def test_studio_adapters_include_user_readable_evaluation_evidence(self) -> None:
        store = self._server_adapter_store()
        unevaluated = self._create_pending_adapter(store, base_model="base-pending")
        pending_result = self._smoke("/pfe/adapters")
        pending_item = next(row for row in pending_result["body"]["versions"] if row["version"] == unevaluated)
        self.assertEqual(pending_result["body"]["pending_eval_adapter"]["version"], unevaluated)
        self.assertFalse(pending_item["can_promote"])
        self.assertEqual(pending_item["promotion_gate"]["reason"], "eval_required")
        self.assertEqual(pending_item["decision"]["primary_action"], "eval")

        version = self._create_pending_adapter(
            store,
            base_model="base-eval",
        )
        store.attach_eval_report(
            version,
            {
                "recommendation": "deploy",
                "comparison": "improved",
                "scores": {
                    "style_match": 0.91,
                    "preference_alignment": 0.88,
                    "quality_preservation": 0.94,
                },
                "summary": "candidate improved personalization without quality loss",
            },
        )

        result = self._smoke("/pfe/adapters")
        self.assertEqual(result["status_code"], 200)
        item = next(row for row in result["body"]["versions"] if row["version"] == version)
        self.assertEqual(item["user_state"], "待确认")
        self.assertEqual(item["training_summary"]["num_samples"], 3)
        self.assertIn("训练样本 3", item["training_summary"]["summary_line"])
        self.assertEqual(item["eval_summary"]["label"], "评估通过")
        self.assertEqual(item["eval_summary"]["recommendation"], "deploy")
        self.assertEqual(item["eval_summary"]["comparison"], "improved")
        self.assertEqual(item["eval_summary"]["scores"]["style_match"], 0.91)
        self.assertIn("评估结论：评估通过", item["eval_summary"]["summary_line"])
        self.assertEqual(item["decision"]["label"], "建议设为当前")
        self.assertEqual(item["decision"]["primary_action"], "promote")
        self.assertTrue(item["can_promote"])
        self.assertTrue(item["can_eval"])
        self.assertEqual(item["action_api"]["eval"], "/pfe/eval")

        failed = self._create_pending_adapter(store, base_model="base-failed")
        store.attach_eval_report(
            failed,
            {
                "recommendation": "keep_previous",
                "comparison": "degraded",
                "scores": {"quality_preservation": 0.42},
            },
        )
        result = self._smoke("/pfe/adapters")
        failed_item = next(row for row in result["body"]["versions"] if row["version"] == failed)
        self.assertEqual(failed_item["user_state"], "有问题")
        self.assertEqual(failed_item["eval_summary"]["label"], "有问题")
        self.assertEqual(failed_item["decision"]["label"], "建议保留旧版")
        self.assertEqual(failed_item["decision"]["primary_action"], "archive")
        self.assertFalse(failed_item["can_promote"])
        self.assertTrue(failed_item["can_eval"])

    def test_studio_eval_requires_confirmation_and_updates_version_evidence(self) -> None:
        store = self._server_adapter_store()
        version = self._create_pending_adapter(store, base_model="base-eval-run")

        missing_confirmation = self._smoke(
            "/pfe/eval",
            method="POST",
            body={"version": version},
        )
        self.assertEqual(missing_confirmation["status_code"], 409)
        self.assertEqual(missing_confirmation["body"]["code"], "confirmation_required")
        self.assertEqual(missing_confirmation["body"]["status_url"], "/pfe/eval/status")

        pipeline = self.app.state.pfe_services.pipeline
        original_evaluate = getattr(pipeline, "evaluate", None)

        def fake_evaluate(**kwargs):
            self.assertEqual(kwargs["adapter"], version)
            report = {
                "recommendation": "deploy",
                "comparison": "improved",
                "scores": {"style_match": 0.93, "preference_alignment": 0.87},
            }
            store.attach_eval_report(version, report)
            return "EVAL COMPLETE"

        pipeline.evaluate = fake_evaluate
        try:
            started = self._smoke(
                "/pfe/eval",
                method="POST",
                body={"version": version, "confirm": True, "num_samples": 3},
            )
            self.assertEqual(started["status_code"], 202)
            self.assertEqual(started["body"]["state"], "running")
            self.assertEqual(started["body"]["version"], version)
            self.assertEqual(started["body"]["status_url"], "/pfe/eval/status")
            running_item = next(row for row in started["body"]["adapters"]["versions"] if row["version"] == version)
            self.assertTrue(running_item["eval_running"])
            self.assertFalse(running_item["can_eval"])
            self.assertEqual(running_item["eval_summary"]["label"], "评估中")

            status = {}
            for _ in range(20):
                result = self._smoke("/pfe/eval/status")
                self.assertEqual(result["status_code"], 200)
                status = result["body"]
                if status.get("state") == "completed":
                    break
                time.sleep(0.05)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["version"], version)
            self.assertEqual(status["recommendation"], "deploy")
            completed_item = next(row for row in status["adapters"]["versions"] if row["version"] == version)
            self.assertEqual(completed_item["eval_summary"]["label"], "评估通过")
            self.assertEqual(completed_item["decision"]["label"], "建议设为当前")
            self.assertTrue(completed_item["can_eval"])
        finally:
            if original_evaluate is not None:
                pipeline.evaluate = original_evaluate

    def test_studio_eval_blocks_when_persisted_running_state_exists(self) -> None:
        store = self._server_adapter_store()
        version = self._create_pending_adapter(store, base_model="base-eval-running")
        workspace = self.app.state.pfe_services.workspace
        server_app._eval_overall_state.pop(workspace, None)
        server_app._save_json_state(
            server_app._eval_state_path(workspace),
            server_app.build_eval_running_state(
                version=version,
                requested_version=version,
                job_id="eval-running-from-disk",
            ),
        )

        result = self._smoke(
            "/pfe/eval",
            method="POST",
            body={"version": version, "confirm": True},
        )

        self.assertEqual(result["status_code"], 409)
        self.assertEqual(result["body"]["code"], "eval_already_running")

    def test_training_jobs_require_confirmation_and_return_preflight_without_side_effects(self) -> None:
        local_model = self.pfe_home / "models" / "tiny-local"
        local_model.mkdir(parents=True)
        config = PFEConfig()
        config.model.base_model = str(local_model)
        config.save(home=self.pfe_home)
        existing_jobs = dict(server_app._training_jobs_state)

        before = self._smoke("/pfe/training/jobs")
        self.assertEqual(before["status_code"], 200)
        self.assertEqual(before["body"]["workspace"], str(self.pfe_home))
        self.assertEqual(before["body"]["items"], [])
        self.assertIsNone(before["body"]["latest"])
        self.assertIsNone(before["body"]["active"])

        result = self._smoke(
            "/pfe/training/jobs",
            method="POST",
            body={"method": "sft", "epochs": 1, "confirm": False},
        )

        self.assertEqual(result["status_code"], 409)
        self.assertEqual(result["body"]["kind"], "pfe_training_preflight_required")
        self.assertEqual(result["body"]["code"], "confirmation_required")
        self.assertEqual(result["body"]["request"]["method"], "sft")
        self.assertEqual(result["body"]["request"]["training_config"], {"epochs": 1})
        self.assertFalse(result["body"]["request"]["confirmed"])
        self.assertNotIn("job_id", result["body"])
        self.assertEqual(server_app._training_jobs_state, existing_jobs)
        self.assertFalse((self.pfe_home / "training_jobs.json").exists())
        preflight = result["body"]["preflight"]
        self.assertEqual(preflight["kind"], "pfe_training_preflight")
        self.assertEqual(preflight["request"]["method"], "sft")
        self.assertTrue(preflight["ready"])
        self.assertTrue(preflight["requires_confirmation"])
        self.assertEqual(preflight["confirm_api"], "POST /pfe/training/jobs")
        self.assertEqual(preflight["method"], "sft")
        self.assertEqual(preflight["base_model"], str(local_model))
        self.assertEqual(preflight["blocked_by"], [])
        self.assertEqual(preflight["preview"]["training_config"], {"epochs": 1})
        self.assertFalse(preflight["preview"]["will_create_job"])
        self.assertFalse(preflight["preview"]["will_start_background_training"])

        after = self._smoke("/pfe/training/jobs")
        self.assertEqual(after["status_code"], 200)
        self.assertEqual(after["body"]["items"], [])
        self.assertIsNone(after["body"]["latest"])

    def test_training_jobs_reject_unsupported_method_without_sft_fallback(self) -> None:
        local_model = self.pfe_home / "models" / "tiny-local"
        local_model.mkdir(parents=True)
        config = PFEConfig()
        config.model.base_model = str(local_model)
        config.save(home=self.pfe_home)

        result = self._smoke(
            "/pfe/training/jobs",
            method="POST",
            body={"method": "bogus", "epochs": 1, "confirm": True},
        )

        self.assertEqual(result["status_code"], 400)
        self.assertEqual(result["body"]["kind"], "pfe_training_request_error")
        self.assertEqual(result["body"]["code"], "unsupported_training_method")
        self.assertEqual(result["body"]["request"]["method"], "bogus")
        self.assertEqual(result["body"]["request"]["training_config"], {"epochs": 1})
        self.assertTrue(result["body"]["request"]["confirmed"])
        self.assertEqual(result["body"]["supported_methods"], ["sft", "dpo"])
        self.assertFalse((self.pfe_home / "training_jobs.json").exists())

    def test_training_jobs_confirmed_request_is_observable_by_job_list(self) -> None:
        local_model = self.pfe_home / "models" / "tiny-local"
        local_model.mkdir(parents=True)
        config = PFEConfig()
        config.model.base_model = str(local_model)
        config.save(home=self.pfe_home)
        pipeline = self.app.state.pfe_services.pipeline
        original_train = getattr(pipeline, "train", None)
        pipeline.train = lambda: "TRAINING COMPLETE 20260615-999"
        try:
            started = self._smoke(
                "/pfe/training/jobs",
                method="POST",
                body={"method": "sft", "epochs": 1, "confirm": True},
            )
            self.assertEqual(started["status_code"], 202)
            self.assertEqual(started["body"]["kind"], "pfe_training_job_started")
            job_id = started["body"]["job_id"]
            self.assertEqual(started["body"]["request"]["method"], "sft")
            self.assertEqual(started["body"]["request"]["training_config"], {"epochs": 1})
            self.assertTrue(started["body"]["request"]["confirmed"])
            self.assertEqual(started["body"]["job"]["job_id"], job_id)
            self.assertEqual(started["body"]["job"]["workspace"], str(self.pfe_home))
            self.assertEqual(started["body"]["job"]["events"][0]["type"], "queued")
            self.assertEqual(started["body"]["job"]["events_url"], f"/pfe/training/jobs/{job_id}/events")
            self.assertEqual(started["body"]["jobs"]["latest"]["job_id"], job_id)
            self.assertEqual(started["body"]["status_url"], f"/pfe/training/jobs/{job_id}")

            job = {}
            for _ in range(20):
                job_result = self._smoke(f"/pfe/training/jobs/{job_id}")
                self.assertEqual(job_result["status_code"], 200)
                job = job_result["body"]
                if job.get("status") == "completed":
                    break
                time.sleep(0.05)
            self.assertEqual(job["status"], "completed")
            self.assertEqual(job["adapter_version"], "20260615-999")
            self.assertEqual(job["status_url"], f"/pfe/training/jobs/{job_id}")
            self.assertEqual(job["events_url"], f"/pfe/training/jobs/{job_id}/events")
            self.assertGreaterEqual(job["event_count"], 3)
            self.assertEqual(job["latest_event"]["type"], "completed")

            events = self._smoke(f"/pfe/training/jobs/{job_id}/events")
            self.assertEqual(events["status_code"], 200)
            self.assertEqual(events["body"]["job_id"], job_id)
            self.assertEqual(events["body"]["latest"]["type"], "completed")
            self.assertEqual(events["body"]["job"]["latest_event"]["type"], "completed")
            self.assertIn(
                "started",
                {item["type"] for item in events["body"]["items"]},
            )

            jobs = self._smoke("/pfe/training/jobs")
            self.assertEqual(jobs["status_code"], 200)
            self.assertEqual(jobs["body"]["latest"]["job_id"], job_id)
            self.assertEqual(jobs["body"]["latest"]["status"], "completed")
            self.assertEqual(jobs["body"]["latest"]["latest_event"]["type"], "completed")
            self.assertEqual(jobs["body"]["state"]["job_id"], job_id)
            self.assertEqual(jobs["body"]["state"]["adapter_version"], "20260615-999")
        finally:
            if original_train is not None:
                pipeline.train = original_train

    def test_training_jobs_confirmed_request_blocks_when_preflight_fails(self) -> None:
        config = PFEConfig()
        config.model.base_model = "Qwen/Qwen2.5-3B-Instruct"
        config.save(home=self.pfe_home)

        result = self._smoke(
            "/pfe/training/jobs",
            method="POST",
            body={"method": "sft", "confirm": True},
        )

        self.assertEqual(result["status_code"], 409)
        self.assertEqual(result["body"]["kind"], "pfe_training_preflight_failed")
        self.assertEqual(result["body"]["code"], "training_preflight_failed")
        self.assertEqual(result["body"]["request"]["method"], "sft")
        self.assertIn("needs_local_path", result["body"]["preflight"]["blocked_by"])

    def test_training_jobs_confirmed_request_blocks_when_active_job_exists(self) -> None:
        local_model = self.pfe_home / "models" / "tiny-local"
        local_model.mkdir(parents=True)
        config = PFEConfig()
        config.model.base_model = str(local_model)
        config.save(home=self.pfe_home)
        job_id = "job-active-start-block"
        now = "2026-06-15T00:00:00Z"
        job = {
            "job_id": job_id,
            "workspace": str(self.pfe_home),
            "status": "running",
            "method": "sft",
            "adapter_version": None,
            "checkpoints": [],
            "events": [],
            "training_config": {"epochs": 1},
            "created_at": now,
            "updated_at": now,
        }
        server_app._append_training_job_event(
            job,
            event_type="started",
            status="running",
            message="training job started",
        )
        server_app._training_jobs_state[job_id] = job
        server_app._save_json_state(server_app._training_jobs_path(str(self.pfe_home)), {job_id: job})

        result = self._smoke(
            "/pfe/training/jobs",
            method="POST",
            body={"method": "sft", "epochs": 1, "confirm": True},
        )

        self.assertEqual(result["status_code"], 409)
        self.assertEqual(result["body"]["code"], "training_job_already_active")
        self.assertEqual(result["body"]["active_job"]["job_id"], job_id)
        self.assertEqual(result["body"]["jobs"]["active"]["job_id"], job_id)

    def test_training_job_cancel_requires_confirmation_and_cancels_queued_job(self) -> None:
        job_id = "job-queued-cancel"
        now = "2026-06-15T00:00:00Z"
        job = {
            "job_id": job_id,
            "workspace": str(self.pfe_home),
            "status": "queued",
            "method": "sft",
            "adapter_version": None,
            "checkpoints": [],
            "events": [],
            "training_config": {"epochs": 1},
            "created_at": now,
            "updated_at": now,
        }
        server_app._append_training_job_event(
            job,
            event_type="queued",
            status="queued",
            message="training job queued",
        )
        server_app._training_jobs_state[job_id] = job
        server_app._save_json_state(server_app._training_jobs_path(str(self.pfe_home)), {job_id: job})

        missing_confirmation = self._smoke(f"/pfe/training/jobs/{job_id}/cancel", method="POST")
        self.assertEqual(missing_confirmation["status_code"], 409)
        self.assertEqual(missing_confirmation["body"]["code"], "confirmation_required")

        cancelled = self._smoke(
            f"/pfe/training/jobs/{job_id}/cancel",
            method="POST",
            body={"confirm": True},
        )
        self.assertEqual(cancelled["status_code"], 200)
        self.assertTrue(cancelled["body"]["success"])
        self.assertEqual(cancelled["body"]["action"], "cancelled")
        self.assertEqual(cancelled["body"]["job"]["status"], "cancelled")
        self.assertEqual(cancelled["body"]["job"]["latest_event"]["type"], "cancelled")

        events = self._smoke(f"/pfe/training/jobs/{job_id}/events")
        self.assertEqual(events["status_code"], 200)
        self.assertEqual(events["body"]["latest"]["type"], "cancelled")

    def test_training_job_cancel_records_request_for_running_job_without_claiming_interrupt(self) -> None:
        job_id = "job-running-cancel"
        now = "2026-06-15T00:00:00Z"
        job = {
            "job_id": job_id,
            "workspace": str(self.pfe_home),
            "status": "running",
            "method": "sft",
            "adapter_version": None,
            "checkpoints": [],
            "events": [],
            "training_config": {"epochs": 1},
            "created_at": now,
            "updated_at": now,
        }
        server_app._append_training_job_event(
            job,
            event_type="started",
            status="running",
            message="training job started",
        )
        server_app._training_jobs_state[job_id] = job
        server_app._save_json_state(server_app._training_jobs_path(str(self.pfe_home)), {job_id: job})

        cancelled = self._smoke(
            f"/pfe/training/jobs/{job_id}/cancel",
            method="POST",
            body={"confirm": True},
        )
        self.assertEqual(cancelled["status_code"], 200)
        self.assertEqual(cancelled["body"]["action"], "cancel_requested")
        self.assertEqual(cancelled["body"]["job"]["status"], "running")
        self.assertTrue(cancelled["body"]["job"]["cancellation_requested"])
        self.assertEqual(cancelled["body"]["job"]["latest_event"]["type"], "cancel_requested")
        self.assertIn("无法被强行中断", cancelled["body"]["message"])
        self.assertTrue(job["cancellation_requested"])
        self.assertEqual(job["events"][-1]["type"], "cancel_requested")

    def test_training_job_cancel_blocks_terminal_job(self) -> None:
        job_id = "job-completed-cancel"
        now = "2026-06-15T00:00:00Z"
        job = {
            "job_id": job_id,
            "workspace": str(self.pfe_home),
            "status": "completed",
            "method": "sft",
            "adapter_version": "20260615-001",
            "checkpoints": [],
            "events": [],
            "training_config": {"epochs": 1},
            "created_at": now,
            "updated_at": now,
        }
        server_app._append_training_job_event(
            job,
            event_type="completed",
            status="completed",
            message="training job completed",
        )
        server_app._training_jobs_state[job_id] = job
        server_app._save_json_state(server_app._training_jobs_path(str(self.pfe_home)), {job_id: job})

        result = self._smoke(
            f"/pfe/training/jobs/{job_id}/cancel",
            method="POST",
            body={"confirm": True},
        )
        self.assertEqual(result["status_code"], 409)
        self.assertEqual(result["body"]["code"], "job_not_cancellable")

    def test_training_job_retry_requires_confirmation_and_reuses_original_config(self) -> None:
        local_model = self.pfe_home / "models" / "tiny-local"
        local_model.mkdir(parents=True)
        config = PFEConfig()
        config.model.base_model = str(local_model)
        config.save(home=self.pfe_home)

        job_id = "job-failed-retry"
        now = "2026-06-15T00:00:00Z"
        original_job = {
            "job_id": job_id,
            "workspace": str(self.pfe_home),
            "status": "failed",
            "method": "sft",
            "adapter_version": None,
            "checkpoints": [],
            "events": [],
            "training_config": {"epochs": 2, "learning_rate": 0.001},
            "created_at": now,
            "updated_at": now,
            "error": "trainer failed",
        }
        server_app._append_training_job_event(
            original_job,
            event_type="failed",
            status="failed",
            message="training job failed",
        )
        server_app._training_jobs_state[job_id] = original_job
        server_app._save_json_state(server_app._training_jobs_path(str(self.pfe_home)), {job_id: original_job})

        missing_confirmation = self._smoke(f"/pfe/training/jobs/{job_id}/retry", method="POST")
        self.assertEqual(missing_confirmation["status_code"], 409)
        self.assertEqual(missing_confirmation["body"]["kind"], "pfe_training_preflight_required")
        self.assertEqual(missing_confirmation["body"]["code"], "confirmation_required")
        self.assertEqual(missing_confirmation["body"]["retry_of"], job_id)
        self.assertEqual(missing_confirmation["body"]["request"]["method"], "sft")
        self.assertFalse(missing_confirmation["body"]["request"]["confirmed"])
        self.assertEqual(
            missing_confirmation["body"]["confirm_api"],
            f"POST /pfe/training/jobs/{job_id}/retry",
        )
        self.assertEqual(missing_confirmation["body"]["preflight"]["base_model"], str(local_model))
        self.assertEqual(
            missing_confirmation["body"]["preflight"]["preview"]["training_config"],
            {"epochs": 2, "learning_rate": 0.001},
        )

        pipeline = self.app.state.pfe_services.pipeline
        original_train = getattr(pipeline, "train", None)
        pipeline.train = lambda: "TRAINING COMPLETE 20260615-998"
        try:
            retried = self._smoke(
                f"/pfe/training/jobs/{job_id}/retry",
                method="POST",
                body={"confirm": True},
            )
            self.assertEqual(retried["status_code"], 202)
            self.assertEqual(retried["body"]["kind"], "pfe_training_job_started")
            self.assertEqual(retried["body"]["action"], "retry_started")
            self.assertEqual(retried["body"]["retry_of"], job_id)
            self.assertEqual(retried["body"]["request"]["method"], "sft")
            self.assertEqual(
                retried["body"]["request"]["training_config"],
                {"epochs": 2, "learning_rate": 0.001},
            )
            new_job_id = retried["body"]["job_id"]
            self.assertNotEqual(new_job_id, job_id)
            new_job = retried["body"]["job"]
            self.assertEqual(new_job["retry_of"], job_id)
            self.assertEqual(new_job["training_config"], {"epochs": 2, "learning_rate": 0.001})
            self.assertEqual(new_job["retry_url"], f"/pfe/training/jobs/{new_job_id}/retry")

            completed = {}
            for _ in range(20):
                job_result = self._smoke(f"/pfe/training/jobs/{new_job_id}")
                self.assertEqual(job_result["status_code"], 200)
                completed = job_result["body"]
                if completed.get("status") == "completed" and "auto_eval" in completed:
                    break
                time.sleep(0.05)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["retry_of"], job_id)
            self.assertEqual(completed["adapter_version"], "20260615-998")
            self.assertEqual(completed["auto_eval"]["state"], "failed_to_start")

            original_events = self._smoke(f"/pfe/training/jobs/{job_id}/events")
            self.assertEqual(original_events["status_code"], 200)
            self.assertIn("retry_requested", {item["type"] for item in original_events["body"]["items"]})
        finally:
            if original_train is not None:
                pipeline.train = original_train

    def test_training_job_retry_blocks_non_terminal_job(self) -> None:
        job_id = "job-running-retry"
        now = "2026-06-15T00:00:00Z"
        job = {
            "job_id": job_id,
            "workspace": str(self.pfe_home),
            "status": "running",
            "method": "sft",
            "adapter_version": None,
            "checkpoints": [],
            "events": [],
            "training_config": {"epochs": 1},
            "created_at": now,
            "updated_at": now,
        }
        server_app._append_training_job_event(
            job,
            event_type="started",
            status="running",
            message="training job started",
        )
        server_app._training_jobs_state[job_id] = job
        server_app._save_json_state(server_app._training_jobs_path(str(self.pfe_home)), {job_id: job})

        result = self._smoke(
            f"/pfe/training/jobs/{job_id}/retry",
            method="POST",
            body={"confirm": True},
        )
        self.assertEqual(result["status_code"], 409)
        self.assertEqual(result["body"]["code"], "job_not_retryable")

    def test_training_job_retry_rejects_missing_or_unsupported_method(self) -> None:
        now = "2026-06-15T00:00:00Z"
        cases = (
            ("job-missing-method", None, 409, "training_method_missing"),
            ("job-bad-method", "bogus", 400, "unsupported_training_method"),
        )
        for job_id, method, expected_status, expected_code in cases:
            job = {
                "job_id": job_id,
                "workspace": str(self.pfe_home),
                "status": "failed",
                "adapter_version": None,
                "checkpoints": [],
                "events": [],
                "training_config": {"epochs": 1},
                "created_at": now,
                "updated_at": now,
            }
            if method is not None:
                job["method"] = method
            server_app._append_training_job_event(
                job,
                event_type="failed",
                status="failed",
                message="training job failed",
            )
            server_app._training_jobs_state[job_id] = job
            server_app._save_json_state(server_app._training_jobs_path(str(self.pfe_home)), {job_id: job})

            result = self._smoke(
                f"/pfe/training/jobs/{job_id}/retry",
                method="POST",
                body={"confirm": True},
            )

            self.assertEqual(result["status_code"], expected_status)
            self.assertEqual(result["body"]["code"], expected_code)
            if method is not None:
                self.assertEqual(result["body"]["kind"], "pfe_training_request_error")
                self.assertEqual(result["body"]["request"]["method"], method)

    def test_training_trigger_legacy_endpoint_uses_studio_job_contract(self) -> None:
        pipeline = self.app.state.pfe_services.pipeline
        original_train = getattr(pipeline, "train", None)
        pipeline.train = lambda: "TRAINING COMPLETE 20260615-997"
        try:
            triggered = self._smoke(
                "/pfe/training/trigger",
                method="POST",
                body={"reason": "legacy_manual", "epochs": 1},
            )
            self.assertEqual(triggered["status_code"], 202)
            self.assertEqual(triggered["body"]["kind"], "pfe_training_job_started")
            self.assertEqual(triggered["body"]["legacy_endpoint"], "/pfe/training/trigger")
            self.assertEqual(triggered["body"]["reason"], "legacy_manual")
            self.assertEqual(triggered["body"]["request"]["method"], "sft")
            self.assertEqual(triggered["body"]["request"]["training_config"], {"epochs": 1})
            self.assertEqual(triggered["body"]["preflight"]["confirm_api"], "POST /pfe/training/trigger")
            self.assertFalse(triggered["body"]["preflight"]["requires_confirmation"])
            self.assertIn(
                "legacy_trigger_bypasses_studio_preflight",
                triggered["body"]["preflight"]["warnings"],
            )
            job_id = triggered["body"]["job_id"]

            completed = {}
            for _ in range(20):
                job_result = self._smoke(f"/pfe/training/jobs/{job_id}")
                self.assertEqual(job_result["status_code"], 200)
                completed = job_result["body"]
                if completed.get("status") == "completed":
                    break
                time.sleep(0.05)
            self.assertEqual(completed["status"], "completed")
            self.assertEqual(completed["adapter_version"], "20260615-997")
            self.assertEqual(completed["reason"], "legacy_manual")

            jobs = self._smoke("/pfe/training/jobs")
            self.assertEqual(jobs["status_code"], 200)
            self.assertEqual(jobs["body"]["latest"]["job_id"], job_id)
            self.assertEqual(jobs["body"]["latest"]["reason"], "legacy_manual")

            status = self._smoke("/pfe/training/status")
            self.assertEqual(status["status_code"], 200)
            self.assertEqual(status["body"]["state"], "completed")
            self.assertEqual(status["body"]["job_id"], job_id)
            self.assertEqual(status["body"]["adapter_version"], "20260615-997")
            self.assertEqual(status["body"]["reason"], "legacy_manual")
        finally:
            if original_train is not None:
                pipeline.train = original_train

    def test_training_trigger_rejects_unsupported_method_without_sft_fallback(self) -> None:
        result = self._smoke(
            "/pfe/training/trigger",
            method="POST",
            body={"method": "bogus", "reason": "legacy_manual"},
        )

        self.assertEqual(result["status_code"], 400)
        self.assertEqual(result["body"]["kind"], "pfe_training_request_error")
        self.assertEqual(result["body"]["code"], "unsupported_training_method")
        self.assertEqual(result["body"]["legacy_endpoint"], "/pfe/training/trigger")
        self.assertEqual(result["body"]["request"]["method"], "bogus")
        self.assertFalse((self.pfe_home / "training_jobs.json").exists())

    def test_status_returns_runtime_snapshot(self) -> None:
        result = self._smoke("/pfe/status", query_params={"detail": "full"})
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertTrue(body["strict_local"])
        self.assertIn("provider", body)
        self.assertIn("runtime", body)
        self.assertIn("sample_counts", body)
        self.assertEqual(result["request"]["query_params"], {"detail": "full"})
        self.assertIn("inference", body["metadata"])
        self.assertIn("export", body["metadata"])
        self.assertIn("trainer", body["metadata"])
        self.assertIn("lifecycle", body["metadata"])
        self.assertIn("server_runtime", body["metadata"])
        self.assertIn("artifact_format", body["metadata"]["export"])
        self.assertIn("recommended_backend", body["metadata"]["export"])
        self.assertIn("requires_export_step", body["metadata"]["export"])
        self.assertIn("export_artifact_path", body["metadata"]["export"])
        self.assertIn("export_artifact_valid", body["metadata"]["export"])
        self.assertIn("export_artifact_size_bytes", body["metadata"]["export"])
        self.assertIn("artifact_directory", body["metadata"]["trainer"])
        self.assertIn("output_dir", body["metadata"]["trainer"])
        self.assertIn("recommended_backend", body["metadata"]["trainer"])
        self.assertIn("requires_export_step", body["metadata"]["trainer"])
        self.assertIn("export_artifact_summary", body["metadata"]["trainer"])
        self.assertIn("placeholder_files", body["metadata"]["export"])
        self.assertIn("materialized", body["metadata"]["export"])
        self.assertIn("write_state", body["metadata"]["export"])
        self.assertIn(body["metadata"]["export"]["write_state"], {"materialized", "pending"})
        self.assertIn("probe_paths", body["metadata"]["server_runtime"])
        self.assertIn("/pfe/status", {item["path"] for item in body["metadata"]["server_runtime"]["probe_paths"]})
        self.assertIn("launch_mode", body["metadata"]["server_runtime"])
        self.assertIn(body["metadata"]["server_runtime"]["launch_mode"], {"dry_run", "uvicorn.run"})
        self.assertIn("probe_status", body["metadata"]["server_runtime"])
        self.assertIn(body["metadata"]["server_runtime"]["probe_status"]["state"], {"ok", "degraded", "skipped", "deferred"})
        self.assertIn("last_serve_check", body["metadata"]["server_runtime"])
        self.assertIn("serve_summary", body["metadata"]["server_runtime"])
        self.assertIn("launch_state", body["metadata"]["server_runtime"])
        self.assertIn("probe_summary", body["metadata"]["server_runtime"]["serve_summary"])
        self.assertIn("checked_paths", body["metadata"]["server_runtime"]["serve_summary"])
        self.assertIn("train", body["metadata"]["lifecycle"])
        self.assertIn("eval", body["metadata"]["lifecycle"])
        self.assertIn("promotion", body["metadata"]["lifecycle"])
        self.assertIn("serve", body["metadata"]["lifecycle"])
        self.assertIn(body["metadata"]["lifecycle"]["train"]["state"], {"idle", "ready"})
        self.assertIn(body["metadata"]["lifecycle"]["eval"]["state"], {"ready", "waiting_for_holdout"})
        self.assertIn("latest_adapter_version", body["metadata"]["lifecycle"]["promotion"])
        self.assertIn("last_check", body["metadata"]["lifecycle"]["serve"])
        self.assertIn("launch_mode", body["metadata"]["lifecycle"]["serve"])

    def test_signal_accepts_local_management_event(self) -> None:
        result = self._smoke(
            "/pfe/signal",
            method="POST",
            body={
                "event_id": "evt-http-smoke-1",
                "request_id": "req-http-smoke-1",
                "session_id": "sess-http-smoke-1",
                "source_event_id": "evt-source-1",
                "source_event_ids": ["evt-source-1", "evt-http-smoke-1"],
                "event_type": "accept",
                "user_input": "我今天有点焦虑",
                "model_output": "我们先把任务拆成三个最小步骤。",
                "user_action": {"type": "accept"},
                "metadata": {"scenario": "life-coach"},
            },
        )
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertTrue(body["stored"])
        self.assertEqual(body["request_id"], "req-http-smoke-1")
        self.assertEqual(result["request"]["method"], "POST")
        self.assertIn("content-type", result["headers"])

    def test_serve_plan_exposes_command_and_runner_info(self) -> None:
        plan = build_serve_plan(workspace=str(self.pfe_home))
        self.assertEqual(plan.runtime.app_target, "pfe_server.app:app")
        self.assertEqual(plan.runtime.host, "127.0.0.1")
        self.assertEqual(plan.runtime.port, 8921)
        self.assertEqual(plan.runner["target"], "pfe_server.app:app")
        self.assertIn("uvicorn", plan.command)
        self.assertTrue(plan.runtime.dry_run)
        self.assertIsInstance(plan.runtime.uvicorn_available, bool)
        self.assertIn("kind", plan.runner)
        self.assertIn("notes", plan.runtime.model_dump())
        self.assertIn("dry_run", plan.runtime.model_dump())
        self.assertIn("command", plan.runtime.model_dump())

    def test_serve_plan_false_includes_runtime_probe(self) -> None:
        plan = build_serve_plan(workspace=str(self.pfe_home), dry_run=False)
        self.assertFalse(plan.runtime.dry_run)
        self.assertEqual(plan.runner["target"], "pfe_server.app:app")
        self.assertIn(plan.runner["kind"], {"uvicorn.run", "dry_run"})
        self.assertIn("launch_mode", plan.runtime_probe)
        self.assertIn(plan.runtime_probe["launch_mode"], {"dry_run", "uvicorn.run"})
        self.assertIn("probe_status", plan.runtime_probe)
        self.assertIn(plan.runtime_probe["probe_status"]["state"], {"ok", "degraded", "deferred"})
        self.assertIn("last_serve_check", plan.runtime_probe)
        self.assertIn(plan.runtime_probe["last_serve_check"].get("path", "/pfe/status"), {"/pfe/status", None})
        self.assertIn("serve_summary", plan.runtime_probe)
        self.assertIn("launch_state", plan.runtime_probe)
        self.assertIn("probe_summary", plan.runtime_probe["serve_summary"])
        self.assertIn("checked_paths", plan.runtime_probe["serve_summary"])
        self.assertIn("before", plan.runtime_probe["launch_state"])
        self.assertIn("after", plan.runtime_probe["launch_state"])
        self.assertIn("/pfe/status", {item["path"] for item in plan.runtime_probe["probe_paths"]})
        self.assertIn("command", plan.runtime_probe)
        self.assertIn("runner", plan.runtime_probe)
        self.assertIn("probe_status", plan.runtime_probe["serve_summary"])
        self.assertIn("probe_state", plan.runtime_probe["serve_summary"])

    def test_feedback_accepts_explicit_user_feedback(self) -> None:
        """Test that feedback endpoint accepts explicit user feedback signals."""
        # Test accept feedback
        result = self._smoke(
            "/pfe/feedback",
            method="POST",
            body={
                "session_id": "test-session",
                "request_id": "test-request-1",
                "action": "accept",
                "user_message": "Hello",
                "assistant_message": "Hi there!",
                "response_time_seconds": 3.0,
            },
        )
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertTrue(body["success"])
        self.assertEqual(body["signal_type"], "accept")
        # Explicit accept feedback should follow the normalized ChatCollector semantics.
        self.assertEqual(body["confidence"], 0.9)
        self.assertEqual(body["session_id"], "test-session")
        self.assertEqual(body["request_id"], "test-request-1")
        self.assertIn("pipeline_ingest", body["metadata"])
        self.assertEqual(body["metadata"]["pipeline_ingest"]["request_id"], "test-request-1")
        self.assertEqual(body["metadata"]["pipeline_ingest"]["session_id"], "test-session")
        self.assertIn("curation_state", body["metadata"]["pipeline_ingest"]["metadata"])
        self.assertIn("auto_train", body["metadata"]["pipeline_ingest"]["metadata"])

    def test_feedback_reject_signal(self) -> None:
        """Test reject feedback signal extraction."""
        result = self._smoke(
            "/pfe/feedback",
            method="POST",
            body={
                "session_id": "test-session",
                "request_id": "test-request-2",
                "action": "delete",
                "user_message": "Hello",
                "assistant_message": "Hi there!",
            },
        )
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertEqual(body["signal_type"], "reject")
        self.assertEqual(body["confidence"], 0.95)

    def test_feedback_edit_signal(self) -> None:
        """Test edit feedback signal extraction with edit distance calculation."""
        result = self._smoke(
            "/pfe/feedback",
            method="POST",
            body={
                "session_id": "test-session",
                "request_id": "test-request-3",
                "action": "edit",
                "user_message": "Hello",
                "assistant_message": "Hi there!",
                "edited_text": "Hello there!",
            },
        )
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertEqual(body["signal_type"], "edit")
        self.assertIn("metadata", body)

    def test_feedback_regenerate_signal(self) -> None:
        """Test regenerate feedback signal extraction."""
        result = self._smoke(
            "/pfe/feedback",
            method="POST",
            body={
                "session_id": "test-session",
                "request_id": "test-request-4",
                "action": "regenerate",
                "user_message": "Hello",
                "assistant_message": "Hi there!",
            },
        )
        self.assertEqual(result["status_code"], 200)
        body = result["body"]
        self.assertEqual(body["signal_type"], "regenerate")
        self.assertEqual(body["confidence"], 0.85)

    def test_chat_then_feedback_round_trip_preserves_ids_for_closed_loop(self) -> None:
        chat_result = self._smoke(
            "/v1/chat/completions",
            method="POST",
            body={
                "model": "local",
                "adapter_version": "latest",
                "messages": [
                    {
                        "role": "user",
                        "content": "My name is Alex. Please answer with short bullet points.",
                    }
                ],
            },
        )
        self.assertEqual(chat_result["status_code"], 200)
        chat_body = chat_result["body"]
        self.assertIn("session_id", chat_body)
        self.assertIn("request_id", chat_body)
        self.assertTrue(chat_body["session_id"])
        self.assertTrue(chat_body["request_id"])
        self.assertEqual(
            chat_body["metadata"]["signal_collection"]["session_id"],
            chat_body["session_id"],
        )
        self.assertEqual(
            chat_body["metadata"]["signal_collection"]["request_id"],
            chat_body["request_id"],
        )
        self.assertTrue(chat_body["metadata"]["signal_collection"]["interaction_stored"])

        feedback_result = self._smoke(
            "/pfe/feedback",
            method="POST",
            body={
                "session_id": chat_body["session_id"],
                "request_id": chat_body["request_id"],
                "action": "accept",
                "response_time_seconds": 2.5,
            },
        )
        self.assertEqual(feedback_result["status_code"], 200)
        feedback_body = feedback_result["body"]
        self.assertTrue(feedback_body["success"])
        self.assertEqual(feedback_body["session_id"], chat_body["session_id"])
        self.assertEqual(feedback_body["request_id"], chat_body["request_id"])
        self.assertEqual(feedback_body["signal_type"], "accept")
        self.assertGreaterEqual(feedback_body["metadata"]["signals_extracted"], 0)

    def test_chat_completion_uses_initialized_base_model_for_local_model(self) -> None:
        config = PFEConfig()
        config.model.base_model = "/models/init-server-default"
        config.save(home=self.pfe_home)
        captured_base_models: list[str] = []

        def capture_init(engine, config):  # type: ignore[no-untyped-def]
            captured_base_models.append(config.base_model)
            engine.config = config

        with patch("pfe_core.pipeline.InferenceEngine.__init__", new=capture_init), patch(
            "pfe_core.pipeline.InferenceEngine.generate",
            return_value="server configured reply",
        ), patch(
            "pfe_core.pipeline.InferenceEngine.status",
            return_value={"served_by": "mock"},
        ):
            result = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        self.assertEqual(result["status_code"], 200)
        self.assertEqual(captured_base_models, ["/models/init-server-default"])
        self.assertEqual(result["body"]["choices"][0]["message"]["content"], "server configured reply")

    def test_chat_completion_requires_api_key_for_remote_clients_when_enabled(self) -> None:
        previous_allow_remote_access = self.app.state.pfe_services.security.allow_remote_access
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.allow_remote_access = True
            self.app.state.pfe_services.security.auth_mode = "local_optional"
            os.environ.pop("PFE_API_KEY", None)

            remote_without_key = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_key["status_code"], 503)
            self.assertEqual(remote_without_key["body"]["code"], "api_key_not_configured")

            os.environ["PFE_API_KEY"] = "secret-remote-key"

            remote_without_header = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_header["status_code"], 401)
            self.assertEqual(remote_without_header["body"]["code"], "unauthorized")

            remote_with_header = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_with_header["status_code"], 200)
            self.assertIn("session_id", remote_with_header["body"])
            self.assertIn("request_id", remote_with_header["body"])
        finally:
            self.app.state.pfe_services.security.allow_remote_access = previous_allow_remote_access
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_chat_completion_requires_api_key_when_auth_mode_is_strict(self) -> None:
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.auth_mode = "api_key_required"
            os.environ["PFE_API_KEY"] = "secret-local-key"

            local_without_key = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                },
            )
            self.assertEqual(local_without_key["status_code"], 401)
            self.assertEqual(local_without_key["body"]["code"], "unauthorized")

            local_with_key = self._smoke(
                "/v1/chat/completions",
                method="POST",
                body={
                    "model": "local",
                    "messages": [{"role": "user", "content": "ping"}],
                },
                headers={"x-api-key": "secret-local-key"},
            )
            self.assertEqual(local_with_key["status_code"], 200)
            self.assertIn("session_id", local_with_key["body"])
            self.assertIn("request_id", local_with_key["body"])
        finally:
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_feedback_requires_api_key_for_remote_clients_when_enabled(self) -> None:
        previous_allow_remote_access = self.app.state.pfe_services.security.allow_remote_access
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.allow_remote_access = True
            self.app.state.pfe_services.security.auth_mode = "local_optional"
            os.environ.pop("PFE_API_KEY", None)

            remote_without_key = self._smoke(
                "/pfe/feedback",
                method="POST",
                body={
                    "session_id": "sess-remote",
                    "request_id": "req-remote",
                    "action": "accept",
                },
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_key["status_code"], 503)
            self.assertEqual(remote_without_key["body"]["code"], "api_key_not_configured")
            self.assertIn("hint", remote_without_key["body"])

            os.environ["PFE_API_KEY"] = "secret-remote-key"

            remote_without_header = self._smoke(
                "/pfe/feedback",
                method="POST",
                body={
                    "session_id": "sess-remote",
                    "request_id": "req-remote",
                    "action": "accept",
                },
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_header["status_code"], 401)
            self.assertEqual(remote_without_header["body"]["code"], "unauthorized")
            self.assertIn("hint", remote_without_header["body"])

            remote_with_header = self._smoke(
                "/pfe/feedback",
                method="POST",
                body={
                    "session_id": "sess-remote",
                    "request_id": "req-remote",
                    "action": "accept",
                },
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_with_header["status_code"], 200)
            self.assertTrue(remote_with_header["body"]["success"])
        finally:
            self.app.state.pfe_services.security.allow_remote_access = previous_allow_remote_access
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_feedback_requires_api_key_when_auth_mode_is_strict(self) -> None:
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.auth_mode = "api_key_required"
            os.environ["PFE_API_KEY"] = "secret-local-key"

            local_without_key = self._smoke(
                "/pfe/feedback",
                method="POST",
                body={
                    "session_id": "sess-local",
                    "request_id": "req-local",
                    "action": "accept",
                },
            )
            self.assertEqual(local_without_key["status_code"], 401)
            self.assertEqual(local_without_key["body"]["code"], "unauthorized")
            self.assertIn("hint", local_without_key["body"])

            local_with_key = self._smoke(
                "/pfe/feedback",
                method="POST",
                body={
                    "session_id": "sess-local",
                    "request_id": "req-local",
                    "action": "accept",
                },
                headers={"x-api-key": "secret-local-key"},
            )
            self.assertEqual(local_with_key["status_code"], 200)
            self.assertTrue(local_with_key["body"]["success"])
        finally:
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_status_requires_api_key_for_remote_clients_when_enabled(self) -> None:
        previous_allow_remote_access = self.app.state.pfe_services.security.allow_remote_access
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.allow_remote_access = True
            self.app.state.pfe_services.security.auth_mode = "local_optional"
            os.environ.pop("PFE_API_KEY", None)

            remote_without_key = self._smoke(
                "/pfe/status",
                query_params={"detail": "full"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_key["status_code"], 503)
            self.assertEqual(remote_without_key["body"]["code"], "api_key_not_configured")
            self.assertIn("hint", remote_without_key["body"])

            os.environ["PFE_API_KEY"] = "secret-remote-key"

            remote_without_header = self._smoke(
                "/pfe/status",
                query_params={"detail": "full"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_header["status_code"], 401)
            self.assertEqual(remote_without_header["body"]["code"], "unauthorized")
            self.assertIn("hint", remote_without_header["body"])

            remote_with_header = self._smoke(
                "/pfe/status",
                query_params={"detail": "full"},
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_with_header["status_code"], 200)
            self.assertIn("strict_local", remote_with_header["body"])
        finally:
            self.app.state.pfe_services.security.allow_remote_access = previous_allow_remote_access
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_studio_management_surfaces_require_api_key_for_remote_clients_when_enabled(self) -> None:
        previous_allow_remote_access = self.app.state.pfe_services.security.allow_remote_access
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.allow_remote_access = True
            self.app.state.pfe_services.security.auth_mode = "local_optional"
            os.environ.pop("PFE_API_KEY", None)

            for path in ("/pfe/runtime", "/pfe/workspaces", "/pfe/models", "/pfe/adapters", "/pfe/readiness", "/pfe/training/jobs", "/pfe/eval/status"):
                remote_without_key = self._smoke(path, client_host="10.0.0.8")
                self.assertEqual(remote_without_key["status_code"], 503)
                self.assertEqual(remote_without_key["body"]["code"], "api_key_not_configured")
            remote_events_without_key = self._smoke(
                "/pfe/training/jobs/demo/events",
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_events_without_key["status_code"], 503)
            self.assertEqual(remote_events_without_key["body"]["code"], "api_key_not_configured")
            remote_cancel_without_key = self._smoke(
                "/pfe/training/jobs/demo/cancel",
                method="POST",
                body={"confirm": True},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_cancel_without_key["status_code"], 503)
            self.assertEqual(remote_cancel_without_key["body"]["code"], "api_key_not_configured")
            remote_retry_without_key = self._smoke(
                "/pfe/training/jobs/demo/retry",
                method="POST",
                body={"confirm": True},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_retry_without_key["status_code"], 503)
            self.assertEqual(remote_retry_without_key["body"]["code"], "api_key_not_configured")
            for path in (
                "/pfe/adapters/demo/promote",
                "/pfe/adapters/demo/rollback",
                "/pfe/adapters/demo/archive",
            ):
                remote_action_without_key = self._smoke(
                    path,
                    method="POST",
                    body={"confirm": True},
                    client_host="10.0.0.8",
                )
                self.assertEqual(remote_action_without_key["status_code"], 503)
                self.assertEqual(remote_action_without_key["body"]["code"], "api_key_not_configured")
            remote_put_without_key = self._smoke(
                "/pfe/config/model",
                method="PUT",
                body={"base_model": "Qwen/Qwen3-4B"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_put_without_key["status_code"], 503)
            self.assertEqual(remote_put_without_key["body"]["code"], "api_key_not_configured")
            remote_real_local_without_key = self._smoke(
                "/pfe/config/real-local",
                method="PUT",
                body={"enabled": True},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_real_local_without_key["status_code"], 503)
            self.assertEqual(remote_real_local_without_key["body"]["code"], "api_key_not_configured")
            remote_workspace_without_key = self._smoke(
                "/pfe/workspaces",
                method="POST",
                body={"name": "remote-client"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_workspace_without_key["status_code"], 503)
            self.assertEqual(remote_workspace_without_key["body"]["code"], "api_key_not_configured")
            remote_training_without_key = self._smoke(
                "/pfe/training/jobs",
                method="POST",
                body={"method": "sft"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_training_without_key["status_code"], 503)
            self.assertEqual(remote_training_without_key["body"]["code"], "api_key_not_configured")
            remote_eval_without_key = self._smoke(
                "/pfe/eval",
                method="POST",
                body={"version": "demo", "confirm": True},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_eval_without_key["status_code"], 503)
            self.assertEqual(remote_eval_without_key["body"]["code"], "api_key_not_configured")

            os.environ["PFE_API_KEY"] = "secret-remote-key"

            for path in ("/pfe/runtime", "/pfe/workspaces", "/pfe/models", "/pfe/adapters", "/pfe/readiness", "/pfe/training/jobs", "/pfe/eval/status"):
                remote_with_header = self._smoke(
                    path,
                    headers={"Authorization": "Bearer secret-remote-key"},
                    client_host="10.0.0.8",
                )
                self.assertEqual(remote_with_header["status_code"], 200)
            remote_events_with_header = self._smoke(
                "/pfe/training/jobs/demo/events",
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_events_with_header["status_code"], 404)
            self.assertEqual(remote_events_with_header["body"]["code"], "not_found")
            remote_cancel_with_header = self._smoke(
                "/pfe/training/jobs/demo/cancel",
                method="POST",
                body={"confirm": True},
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_cancel_with_header["status_code"], 404)
            self.assertEqual(remote_cancel_with_header["body"]["code"], "not_found")
            remote_retry_with_header = self._smoke(
                "/pfe/training/jobs/demo/retry",
                method="POST",
                body={"confirm": True},
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_retry_with_header["status_code"], 404)
            self.assertEqual(remote_retry_with_header["body"]["code"], "not_found")
            remote_put_with_header = self._smoke(
                "/pfe/config/model",
                method="PUT",
                body={"base_model": "Qwen/Qwen3-4B"},
                query_params={"validate_only": "true"},
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_put_with_header["status_code"], 200)
            remote_real_local_with_header = self._smoke(
                "/pfe/config/real-local",
                method="PUT",
                body={"enabled": False},
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_real_local_with_header["status_code"], 200)
            remote_workspace_with_header = self._smoke(
                "/pfe/workspaces",
                method="POST",
                body={"name": "remote-client"},
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_workspace_with_header["status_code"], 200)
            remote_training_with_header = self._smoke(
                "/pfe/training/jobs",
                method="POST",
                body={"method": "sft"},
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_training_with_header["status_code"], 409)
            self.assertEqual(remote_training_with_header["body"]["code"], "confirmation_required")
            remote_eval_with_header = self._smoke(
                "/pfe/eval",
                method="POST",
                body={"version": "demo", "confirm": True},
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_eval_with_header["status_code"], 404)
            self.assertEqual(remote_eval_with_header["body"]["code"], "not_found")
        finally:
            self.app.state.pfe_services.security.allow_remote_access = previous_allow_remote_access
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            self.app.state.pfe_services.workspace = str(self.pfe_home)
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

    def test_signals_requires_api_key_for_remote_clients_when_enabled(self) -> None:
        previous_allow_remote_access = self.app.state.pfe_services.security.allow_remote_access
        previous_auth_mode = self.app.state.pfe_services.security.auth_mode
        previous_api_key = os.environ.get("PFE_API_KEY")
        try:
            self.app.state.pfe_services.security.allow_remote_access = True
            self.app.state.pfe_services.security.auth_mode = "local_optional"
            os.environ.pop("PFE_API_KEY", None)

            remote_without_key = self._smoke(
                "/pfe/signals",
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_key["status_code"], 503)
            self.assertEqual(remote_without_key["body"]["code"], "api_key_not_configured")
            self.assertIn("hint", remote_without_key["body"])

            os.environ["PFE_API_KEY"] = "secret-remote-key"

            remote_without_header = self._smoke(
                "/pfe/signals",
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_without_header["status_code"], 401)
            self.assertEqual(remote_without_header["body"]["code"], "unauthorized")
            self.assertIn("hint", remote_without_header["body"])

            remote_with_header = self._smoke(
                "/pfe/signals",
                headers={"Authorization": "Bearer secret-remote-key"},
                client_host="10.0.0.8",
            )
            self.assertEqual(remote_with_header["status_code"], 200)
            self.assertIn("signals", remote_with_header["body"])
        finally:
            self.app.state.pfe_services.security.allow_remote_access = previous_allow_remote_access
            self.app.state.pfe_services.security.auth_mode = previous_auth_mode
            if previous_api_key is None:
                os.environ.pop("PFE_API_KEY", None)
            else:
                os.environ["PFE_API_KEY"] = previous_api_key

if __name__ == "__main__":
    unittest.main()
