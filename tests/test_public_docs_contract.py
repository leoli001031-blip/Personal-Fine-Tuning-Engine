from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_readme_is_studio_first_for_default_user_path() -> None:
    readme = _read("README.md")

    assert "[Studio Workflow](#studio-workflow)" in readme
    assert "The default user surface is PFE Studio in the browser" in readme
    assert ".venv/bin/python -m pfe_server --port 8921 --workspace user_default" in readme
    assert "select model -> copy API/web URL -> manage versions" in readme
    assert "The main control plane remains the CLI" not in readme


def test_chinese_readme_is_studio_first_for_default_user_path() -> None:
    readme = _read("README.zh-CN.md")

    assert "[Studio 主路径](#studio-主路径)" in readme
    assert "默认用户入口是浏览器里的 PFE Studio" in readme
    assert ".venv/bin/python -m pfe_server --port 8921 --workspace user_default" in readme
    assert "选择模型 -> 复制 API/网页地址 -> 管理版本" in readme
    assert "主控制面仍然是 CLI" not in readme


def test_release_docs_record_remote_gate_as_closed() -> None:
    closeout = _read("docs/reference/phase2-closeout.md")
    notes = _read("docs/reference/release-notes-phase2-rc.md")
    evidence = _read("docs/reference/release-readiness-evidence.md")

    assert "远端 GitHub Actions strict release gate 已在 `main` 通过" in closeout
    assert "当前 release gate 阻塞项为 0" in notes
    assert "actions/runs/27518991700" in evidence
    assert "commit: `0c08d2791edce2ac6c48ce1f432a1eb6716fca8d`" in evidence
    assert "剩余 release-ready 缺口是远端 GitHub Actions 结果" not in notes
    assert "远端 CI 证据" not in closeout.split("## 6. 最终判断", 1)[-1]
