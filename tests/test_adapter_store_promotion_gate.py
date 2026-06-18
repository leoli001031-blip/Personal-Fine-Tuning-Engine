from __future__ import annotations

from pathlib import Path

import pytest

from pfe_core.adapter_store.store import AdapterStore
from pfe_core.errors import AdapterError


def _pending_adapter(store: AdapterStore) -> str:
    created = store.create_training_version(
        base_model="base",
        training_config={"backend": "mock_local", "train_type": "sft"},
    )
    version = str(created["version"])
    store.mark_pending_eval(version, num_samples=1, metrics={"loss": 0.1})
    return version


def test_promote_blocks_pending_adapter_without_deploy_eval(tmp_path: Path) -> None:
    store = AdapterStore(home=tmp_path / ".pfe")
    version = _pending_adapter(store)

    with pytest.raises(AdapterError, match="evaluation recommendation=deploy"):
        store.promote(version)

    assert store.current_latest_version() is None


def test_promote_allows_pending_adapter_after_deploy_eval(tmp_path: Path) -> None:
    store = AdapterStore(home=tmp_path / ".pfe")
    version = _pending_adapter(store)
    store.attach_eval_report(
        version,
        {
            "recommendation": "deploy",
            "comparison": "improved",
            "scores": {},
            "studio_eval_suite": {"passed": True, "results": []},
        },
    )

    assert f"Promoted {version}" in store.promote(version)
    assert store.current_latest_version() == version


def test_promote_blocks_failed_studio_eval_suite_even_with_deploy_recommendation(tmp_path: Path) -> None:
    store = AdapterStore(home=tmp_path / ".pfe")
    version = _pending_adapter(store)
    store.attach_eval_report(
        version,
        {
            "recommendation": "deploy",
            "comparison": "improved",
            "scores": {},
            "studio_eval_suite": {"passed": False, "failed_cases": ["refusal"]},
        },
    )

    with pytest.raises(AdapterError, match="Studio eval suite did not pass"):
        store.promote(version)

    assert store.current_latest_version() is None
