from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
for package_dir in ("pfe-core", "pfe-cli", "pfe-server"):
    package_path = str(ROOT / package_dir)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)


@pytest.fixture()
def pfe_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / ".pfe"
    monkeypatch.setenv("PFE_HOME", str(home))
    return home


@pytest.fixture(scope="session")
def pfe_packages_available() -> bool:
    """Check that PFE packages can be imported in the current interpreter.

    E2E tests will auto-detect the project venv when needed, but this
    fixture lets tests fail fast with a clear message if something is wrong.
    """
    try:
        import pfe_core  # noqa: F401
        import pfe_server  # noqa: F401
        return True
    except ImportError as exc:
        pytest.skip(
            f"PFE packages not available in current interpreter ({sys.executable}): {exc}"
        )

