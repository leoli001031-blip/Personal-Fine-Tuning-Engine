# Phase23 Signal Routing Examples

## phase23-signal-accept-001

- Type: accept
- Lanes: memory, manual_review
- Eligible for training: False
- Reason: accept_not_enough_for_training

## phase23-signal-reject-001

- Type: reject
- Lanes: manual_review
- Eligible for training: False
- Reason: requires_positive_pair

## phase23-signal-edit-001

- Type: correction
- Lanes: memory, training_candidate
- Eligible for training: True
- Reason: high-information correction/edit preserves the runtime contract and source citation

## phase23-signal-correction-001

- Type: correction
- Lanes: memory, training_candidate
- Eligible for training: True
- Reason: high-information correction/edit preserves the runtime contract and source citation

## phase23-signal-preference-001

- Type: preference
- Lanes: profile
- Eligible for training: False
- Reason: preferences update profile first and require reinforcement before training

## phase23-signal-safety-001

- Type: safety_block
- Lanes: manual_review, excluded
- Eligible for training: False
- Reason: safety_block

## phase23-signal-external-law-001

- Type: correction
- Lanes: manual_review, excluded
- Eligible for training: False
- Reason: external_law_inducement

## phase23-signal-pii-001

- Type: correction
- Lanes: excluded
- Eligible for training: False
- Reason: detected_high_risk_pii

## phase23-signal-edit-missing-output-001

- Type: edit
- Lanes: manual_review, excluded
- Eligible for training: False
- Reason: edit_missing_corrected_output
