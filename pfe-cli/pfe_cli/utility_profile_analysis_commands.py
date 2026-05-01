"""Profile analysis, extraction, and drift commands."""

from __future__ import annotations

import json
from typing import Optional

import typer


def register_profile_analysis_commands(profile_app: typer.Typer) -> None:
    @profile_app.command("analyze")
    def profile_analyze(
        user_id: str = typer.Option("default", "--user-id", help="User identifier."),
        incremental: bool = typer.Option(True, "--incremental/--full", help="Incremental or full analysis."),
        json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Manually trigger rule-based profile analysis from stored signals."""

        del workspace
        from pfe_core.profile_extractor import extract_profile_for_user

        try:
            result = extract_profile_for_user(
                user_id=user_id,
                signals=None,
                incremental=incremental,
            )

            if json_output:
                typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
                return

            typer.echo(f"Profile analysis completed for user: {user_id}")
            typer.echo(f"Signals analyzed: {result.get('signals_analyzed', 0)}")

            if result.get("domains_found"):
                typer.echo(f"Domains found: {', '.join(result['domains_found'])}")
            if result.get("styles_found"):
                typer.echo(f"Styles found: {', '.join(result['styles_found'])}")
            if result.get("patterns_found"):
                typer.echo(f"Patterns found: {', '.join(result['patterns_found'])}")
            if result.get("profile_summary"):
                typer.echo(f"\nProfile Summary: {result['profile_summary']}")
        except Exception as exc:
            typer.secho(f"Error analyzing profile: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @profile_app.command("extract")
    def profile_extract(
        user_id: str = typer.Option("default", "--user-id", help="User identifier."),
        use_llm: bool = typer.Option(False, "--use-llm", help="Use LLM for structured extraction."),
        json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Extract or re-extract profile summary for a user."""

        del workspace
        from pfe_core.profile_extractor import ProfileExtractor

        try:
            extractor = ProfileExtractor(user_id=user_id)
            summary = extractor.generate_profile_summary(use_llm=use_llm)
            profile = extractor.profile

            if json_output:
                typer.echo(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2))
                return

            typer.echo(f"Profile extraction completed for user: {user_id}")
            typer.echo(f"Extracted by: {profile.extracted_by}")
            if profile.llm_extracted_at:
                typer.echo(f"LLM extracted at: {profile.llm_extracted_at.isoformat()}")
            typer.echo(f"Summary: {summary}")
        except Exception as exc:
            typer.secho(f"Error extracting profile: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)

    @profile_app.command("drift")
    def profile_drift(
        user_id: str = typer.Option("default", "--user-id", help="User identifier."),
        threshold: float = typer.Option(0.3, "--threshold", help="Drift detection threshold."),
        json_output: bool = typer.Option(False, "--json", help="Emit JSON output."),
        workspace: Optional[str] = typer.Option(None, "--workspace", help="Workspace label."),
    ) -> None:
        """Show preference drift detection report for a user."""

        del workspace
        from pfe_core.profile_extractor import get_user_profile_store

        try:
            store = get_user_profile_store()
            profile = store.get_profile(user_id)
            alerts = profile.detect_preference_drift(threshold=threshold)

            if json_output:
                typer.echo(json.dumps({"alerts": alerts}, ensure_ascii=False, indent=2))
                return

            typer.echo(f"Drift Report for user: {user_id}")
            if not alerts:
                typer.echo("No significant preference drift detected.")
                return

            typer.echo(f"Detected {len(alerts)} drift alert(s) (threshold={threshold}):")
            for alert in alerts:
                direction_icon = "+" if alert["drift_direction"] == "increase" else "-"
                typer.echo(
                    f"  - {alert['preference_key']}: {alert['old_avg']:.2f} -> {alert['new_avg']:.2f} "
                    f"({direction_icon}, severity={alert['severity']})"
                )
        except Exception as exc:
            typer.secho(f"Error detecting drift: {exc}", err=True, fg=typer.colors.RED)
            raise typer.Exit(code=1)


__all__ = ["register_profile_analysis_commands"]
