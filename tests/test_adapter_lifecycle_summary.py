from __future__ import annotations

from pfe_cli.adapter_lifecycle_summary import _format_lifecycle_summary


def test_lifecycle_summary_keeps_first_non_latest_version_line() -> None:
    result = "\n".join(
        [
            "  20260617-005  state=pending_eval  samples=2  format=peft_lora",
            "* 20260617-002  state=promoted  samples=6  format=gguf_merged",
        ]
    )

    lines = _format_lifecycle_summary(result)

    assert lines == [
        "Adapter versions",
        "latest: 20260617-002",
        "- 20260617-005 | lifecycle=pending_eval | samples=2 | format=peft_lora",
        "- 20260617-002 | lifecycle=promoted | latest=yes | samples=6 | format=gguf_merged",
    ]
