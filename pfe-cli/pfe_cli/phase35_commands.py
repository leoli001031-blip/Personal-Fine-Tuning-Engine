"""Phase35 lightweight local interaction capture CLI commands."""

from __future__ import annotations

from typing import Optional

import typer


def _phase35_home():
    from pfe_core.config import PFEConfig

    return PFEConfig.resolve_home()


def register_phase35_commands(phase35_app: typer.Typer) -> None:
    """Attach Phase35 local interaction commands."""

    @phase35_app.command("interact")
    def phase35_interact(
        workspace: str = typer.Option("user_default", "--workspace", help="Workspace label."),
        user_goal: str = typer.Option(..., "--user-goal", help="Local task goal from the user."),
        feedback_action: str = typer.Option("accept", "--feedback-action", help="accept, reject, edit, correction, continue, final_acceptance."),
        user_feedback: str = typer.Option("", "--user-feedback", help="User judgement or correction text."),
        edited_text: str = typer.Option("", "--edited-text", help="Edited target text for edit/correction feedback."),
        model_variant: str = typer.Option("adapter", "--model-variant", help="base or adapter profile replay."),
        operator_id: str = typer.Option("", "--operator-id", help="Required when confirming actual user feedback."),
        confirm_actual_user_feedback: bool = typer.Option(False, "--confirm-actual-user-feedback", help="Mark this local record as actual user feedback."),
        consent_for_training_candidate_review: bool = typer.Option(False, "--consent-for-training-candidate-review", help="Allow later review for training candidates."),
        not_scripted_or_curated: bool = typer.Option(False, "--not-scripted-or-curated", help="Assert this was not scripted or curated."),
        simulated_local_interaction: bool = typer.Option(False, "--simulated-local-interaction", help="Mark as simulated evidence only."),
    ) -> None:
        """Capture one local interaction into the Phase35 pending-review store."""

        from pfe_core.phase35_local_interaction_capture import (
            append_phase35_capture_batch,
            build_phase35_capture_batch,
            build_phase35_interaction_record,
            build_phase35_readiness,
            load_phase35_state,
            phase35_store_path,
            render_phase35_agent_response,
        )

        response = render_phase35_agent_response(user_goal=user_goal, model_variant=model_variant)
        record = build_phase35_interaction_record(
            workspace=workspace,
            user_goal=user_goal,
            assistant_response=str(response["assistant_response"]),
            feedback_action=feedback_action,
            user_feedback=user_feedback,
            edited_text=edited_text,
            model_variant=model_variant,
            operator_id=operator_id,
            confirmed_actual_user_feedback=confirm_actual_user_feedback,
            consent_for_training_candidate_review=consent_for_training_candidate_review,
            not_scripted_or_curated=not_scripted_or_curated,
            simulated_local_interaction=simulated_local_interaction,
        )
        batch = build_phase35_capture_batch([record])
        store = phase35_store_path(_phase35_home(), workspace)
        state = append_phase35_capture_batch(store, batch)
        readiness = build_phase35_readiness(state)

        typer.echo("Phase35 local interaction captured")
        typer.echo(f"Interaction ID: {record.get('interaction_id')}")
        typer.echo(f"Assistant Response: {record.get('assistant_response')}")
        typer.echo(f"Feedback Source: {record.get('feedback_source')}")
        typer.echo(f"Accepted Pending Review: {batch.get('accepted_pending_review_count')}")
        typer.echo(f"Non Training: {batch.get('non_training_count')}")
        typer.echo(f"Blocked: {batch.get('blocked_count')}")
        typer.echo(f"Quarantined: {batch.get('quarantined_count')}")
        typer.echo(f"Pending Review: {readiness.get('pending_review_count')}")
        typer.echo(f"Ready For Phase36 Review: {readiness.get('ready_for_phase36_review')}")
        typer.echo("Training: blocked (phase35_capture_only_phase36_review_required)")

    @phase35_app.command("status")
    def phase35_status(
        workspace: str = typer.Option("user_default", "--workspace", help="Workspace label."),
    ) -> None:
        """Show Phase35 local interaction readiness."""

        from pfe_core.phase35_local_interaction_capture import (
            build_phase35_readiness,
            load_phase35_state,
            phase35_store_path,
        )

        store = phase35_store_path(_phase35_home(), workspace)
        readiness = build_phase35_readiness(load_phase35_state(store))
        typer.echo("Phase35 Local Interaction Status")
        typer.echo(f"Interaction Count: {readiness.get('interaction_count')}")
        typer.echo(f"Pending Review: {readiness.get('pending_review_count')}")
        typer.echo(f"Attested Actual Pending Review: {readiness.get('attested_actual_pending_review_count')}")
        typer.echo(f"Ready For Phase36 Review: {readiness.get('ready_for_phase36_review')}")
        typer.echo(f"Training Status: {readiness.get('training_status')}")
        typer.echo(f"Next Action: {readiness.get('next_action')}")

    @phase35_app.command("review-queue")
    def phase35_review_queue(
        workspace: str = typer.Option("user_default", "--workspace", help="Workspace label."),
        limit: int = typer.Option(10, "--limit", help="Maximum pending records to show."),
    ) -> None:
        """List Phase35 interactions waiting for human review."""

        from pfe_core.phase35_local_interaction_capture import (
            build_phase35_review_queue,
            load_phase35_state,
            phase35_store_path,
        )

        store = phase35_store_path(_phase35_home(), workspace)
        queue = build_phase35_review_queue(load_phase35_state(store))
        typer.echo("Phase35 Review Queue")
        typer.echo(f"Pending Review: {queue.get('pending_review_count')}")
        for item in list(queue.get("pending_review") or [])[:limit]:
            typer.echo(f"- {item.get('interaction_id')} | {item.get('feedback_source')} | {item.get('user_goal')}")
