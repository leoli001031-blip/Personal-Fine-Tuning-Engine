# Phase85 Final Decision

- Status: `archive_incomplete_phase85_evidence`
- Recommendation: `repair_phase85_evidence`
- Evidence type: simulated benchmark with real local model calls
- Actual user feedback: 0
- Training or adapter benefit claim: not allowed
- Automatic promotion/deployment: not allowed

## Three-arm result

- `base_api_length_control_160`: native=0.0000, repair=0.0000, fallback=1.0000, latency_p95=3.5530s
- `persona_api_contract_v3_fresh`: native=0.1029, repair=0.1912, fallback=0.7059, latency_p95=2.9267s
- `persona_api_contract_v4`: native=0.1029, repair=0.1471, fallback=0.7500, latency_p95=1.4068s

- V4 target score: `0.6467`
- V4 gain vs base: `0.2234`
- V4 target category floor: `0.52`

## Failed strict gates

- `v4_target_score_at_least_0_80`
- `v4_target_not_below_v3`
- `v4_each_target_category_at_least_0_75`
- `v4_native_format_rate_at_least_0_75`
- `v4_fallback_rate_at_most_0_10`
- `v4_fallback_below_v3`
- `manual_review_found_no_semantic_failures`
