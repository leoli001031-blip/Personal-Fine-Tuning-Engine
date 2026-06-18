"""Adapter lifecycle summary formatting."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from .adapter_value_formatting import _format_value


def _format_lifecycle_summary(result: Any) -> list[str] | None:
    mapping = result if isinstance(result, dict) else None
    if mapping is None and isinstance(result, str):
        return _format_lifecycle_summary_from_text(result)

    if mapping is None:
        return None

    if "versions" in mapping and isinstance(mapping["versions"], Sequence):
        return _format_version_list(mapping["versions"])

    if "version" in mapping or "state" in mapping or "latest" in mapping:
        return _format_single_version(mapping)

    return None


def _format_lifecycle_summary_from_text(result: str) -> list[str] | None:
    stripped = result.strip()
    match = re.match(r"^Promoted\s+(.+?)\s+to\s+latest\.$", stripped)
    if match is not None:
        return [f"latest: {match.group(1)}", "lifecycle: promoted"]
    if stripped == "No adapter versions found.":
        return [stripped]

    version_lines: list[tuple[str, str, bool, str, str]] = []
    latest_version = None
    for raw_line in result.strip("\n").splitlines():
        line = raw_line.rstrip()
        match = re.match(
            r"^([* ])\s+([^\s]+)\s+state=([^\s]+)\s+samples=([^\s]+)\s+format=([^\s]+)$",
            line,
        )
        if match is None:
            continue
        marker, version, lifecycle, samples, artifact_format = match.groups()
        latest_flag = marker == "*"
        if latest_flag:
            latest_version = version
        version_lines.append((version, lifecycle, latest_flag, samples, artifact_format))
    if not version_lines:
        return None

    lines = ["Adapter versions"]
    if latest_version is not None:
        lines.append(f"latest: {latest_version}")
    for version, lifecycle, latest_flag, samples, artifact_format in version_lines:
        suffix = " | latest=yes" if latest_flag else ""
        lines.append(f"- {version} | lifecycle={lifecycle}{suffix} | samples={samples} | format={artifact_format}")
    return lines


def _format_version_list(versions: Sequence[Any]) -> list[str]:
    lines = ["Adapter versions"]
    latest = None
    for item in versions:
        item_map = item if isinstance(item, dict) else None
        if item_map is not None and item_map.get("latest"):
            latest = item_map.get("version")
            break
    if latest is not None:
        lines.append(f"latest: {_format_value(latest)}")
    for item in versions:
        item_map = item if isinstance(item, dict) else None
        if item_map is None:
            lines.append(f"- {_format_value(item)}")
            continue
        version = item_map.get("version", "n/a")
        lifecycle = item_map.get("state", item_map.get("status", "n/a"))
        latest_flag = item_map.get("latest", False)
        suffix = " | latest=yes" if latest_flag else ""
        lines.append(
            f"- {version} | lifecycle={_format_value(lifecycle)}{suffix} | "
            f"samples={_format_value(item_map.get('num_samples', 'n/a'))} | "
            f"format={_format_value(item_map.get('artifact_format', 'n/a'))}"
        )
    return lines


def _format_single_version(mapping: dict[str, Any]) -> list[str] | None:
    version = mapping.get("version")
    state = mapping.get("state", mapping.get("status"))
    latest = mapping.get("latest")
    lines = []
    if version is not None:
        lines.append(f"latest: {_format_value(version)}")
    if state is not None:
        lines.append(f"lifecycle: {_format_value(state)}")
    if latest is not None:
        lines.append(f"latest_pointer: {_format_value(latest)}")
    return lines or None


__all__ = ["_format_lifecycle_summary"]
