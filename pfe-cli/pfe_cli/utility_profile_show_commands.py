"""Profile display commands."""

from __future__ import annotations

import json
from typing import Optional

import typer


def register_profile_show_commands(profile_app: typer.Typer) -> None:
    @profile_app.command("show")
    def profile_show(
        user_id: str = typer.Option("default", "--user-id", help="User identifier."),
        json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show the current profile analysis snapshot for a user."""

        del workspace
        from pfe_core.profile_extractor import get_user_profile_store

        try:
            store = get_user_profile_store()
            profile = store.get_profile(user_id)

            if json_output:
                typer.echo(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))
                return

            typer.echo("\n".join(_profile_lines(user_id, profile)))
        except Exception as exc:
            typer.secho(f"Error loading profile: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)


def _profile_lines(user_id: str, profile) -> list[str]:
    lines = [f"User Profile: {user_id}", ""]

    if profile.style_preferences:
        lines.append("Style Preferences:")
        lines.extend(_preference_lines(profile.style_preferences))
        lines.append("")

    if profile.domain_preferences:
        lines.append("Domain Preferences:")
        lines.extend(_preference_lines(profile.domain_preferences))
        lines.append("")

    if profile.interaction_patterns:
        lines.append("Interaction Patterns:")
        lines.extend(_preference_lines(profile.interaction_patterns))
        lines.append("")

    if profile.profile_summary:
        lines.append(f"Profile Summary: {profile.profile_summary}")
    if profile.dominant_style:
        lines.append(f"Dominant Style: {profile.dominant_style}")
    if profile.dominant_domains:
        lines.append(f"Dominant Domains: {', '.join(profile.dominant_domains)}")

    lines.append(f"\nAnalysis Count: {profile.analysis_count}")
    if profile.last_analysis_at:
        lines.append(f"Last Analysis: {profile.last_analysis_at.isoformat()}")
    return lines


def _preference_lines(preferences) -> list[str]:
    lines = []
    for key, pref in sorted(
        preferences.items(),
        key=lambda item: item[1].score * item[1].confidence,
        reverse=True,
    )[:5]:
        lines.append(
            f"  - {key}: {pref.score:.2f} "
            f"(confidence: {pref.confidence:.2f}, freq: {pref.frequency})"
        )
    return lines


__all__ = ["register_profile_show_commands"]
