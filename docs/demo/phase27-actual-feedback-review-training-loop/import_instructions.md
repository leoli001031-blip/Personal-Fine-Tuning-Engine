# Phase27 Import Instructions

The template files are collection aids, not training data. Do not import them as
actual feedback until a real user has filled the correction/preference fields
and the attestation is complete.

Preferred import format is JSONL. CSV is supported for review handoff.

Required policy:

- `feedback_source` must be `actual_user_feedback`.
- `attestation.confirmed_actual_user_feedback` must be true.
- `attestation.not_scripted_or_curated` must be true.
- `attestation.consent_for_training_candidate_review` must be true.
- Keep source citation metadata intact.
- Do not output legal conclusions; only summarize contract material, risk notes, citations, and manual confirmation needs.
