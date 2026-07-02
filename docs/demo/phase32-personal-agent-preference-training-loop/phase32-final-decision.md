# Phase32 Final Decision

## Decision

- Recommendation: promote_after_manual_review
- Status: ready_for_manual_review
- Promotion allowed: True
- Auto promotion allowed: false
- Product benefit claim allowed: True
- Actual user feedback collected: false
- Historical Agent conversations used: true

## Review

- Decisions: 68
- Approved for training: 39
- Excluded: 3
- Quarantined: 26
- Taxonomy counts: `{"boundary_awareness": 68, "concise_status": 68, "correction_responsiveness": 23, "evidence_first": 68, "execution_first": 68, "local_context_awareness": 68, "persistence": 68}`

## Candidates

- SFT samples: 39
- DPO pairs: 39
- Hard negatives: 39
- Profile candidates: 39
- Memory candidates: 39
- Raw private text committed: False

## Training

- Real training: completed
- Selected model: Qwen/Qwen2.5-0.5B-Instruct
- Adapter path: [LOCAL_PATH]

## Scores

- Base: `{"boundary_awareness_rate": 0.0, "concise_status_rate": 0.375, "correction_responsiveness_rate": 0.75, "evidence_grounding_rate": 0.0, "execution_first_rate": 0.75, "follows_user_latest_intent_rate": 0.5, "hallucinated_completion_rate": 0.0, "overall_personalization_score": 0.396, "raw_private_text_leak_rate": 0.0, "unnecessary_explanation_rate": 0.0}`
- Adapter: `{"boundary_awareness_rate": 0.0, "concise_status_rate": 0.375, "correction_responsiveness_rate": 0.75, "evidence_grounding_rate": 0.125, "execution_first_rate": 0.875, "follows_user_latest_intent_rate": 0.5, "hallucinated_completion_rate": 0.0, "overall_personalization_score": 0.437, "raw_private_text_leak_rate": 0.0, "unnecessary_explanation_rate": 0.0}`

## Reasons

- no blocking reasons
