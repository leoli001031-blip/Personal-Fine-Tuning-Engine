"""Content-quality gates for SFT and preference candidates."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Mapping


def _normalized_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).lower()


def assess_preference_candidate_quality(
    pairs: Iterable[Mapping[str, Any]],
    *,
    min_unique_ratio: float = 0.7,
    max_exact_duplicate_rate: float = 0.15,
) -> dict[str, Any]:
    rows = [dict(pair) for pair in pairs]
    count = len(rows)
    chosen = [_normalized_text(row.get("chosen")) for row in rows]
    rejected = [_normalized_text(row.get("rejected")) for row in rows]
    prompts = [_normalized_text(row.get("instruction") or row.get("prompt")) for row in rows]

    def unique_ratio(values: list[str]) -> float:
        return round(len(set(values)) / len(values), 4) if values else 0.0

    pair_keys = list(zip(prompts, chosen, rejected))
    duplicate_count = sum(amount - 1 for amount in Counter(pair_keys).values() if amount > 1)
    duplicate_rate = round(duplicate_count / count, 4) if count else 1.0
    low_information_count = sum(
        not prompt or len(preferred) < 24 or len(dispreferred) < 16 or preferred == dispreferred
        for prompt, preferred, dispreferred in pair_keys
    )
    chosen_unique_ratio = unique_ratio(chosen)
    rejected_unique_ratio = unique_ratio(rejected)
    prompt_unique_ratio = unique_ratio(prompts)
    reasons: list[str] = []
    if not rows:
        reasons.append("no_preference_candidates")
    if any(not prompt for prompt in prompts):
        reasons.append("missing_training_prompt")
    if chosen_unique_ratio < min_unique_ratio:
        reasons.append("chosen_unique_ratio_below_threshold")
    if rejected_unique_ratio < min_unique_ratio:
        reasons.append("rejected_unique_ratio_below_threshold")
    if prompt_unique_ratio < min_unique_ratio:
        reasons.append("prompt_unique_ratio_below_threshold")
    if duplicate_rate > max_exact_duplicate_rate:
        reasons.append("exact_duplicate_rate_above_threshold")
    if low_information_count:
        reasons.append("low_information_preference_candidates")
    return {
        "kind": "pfe_preference_candidate_quality",
        "passed": not reasons,
        "candidate_count": count,
        "chosen_unique_ratio": chosen_unique_ratio,
        "rejected_unique_ratio": rejected_unique_ratio,
        "prompt_unique_ratio": prompt_unique_ratio,
        "exact_duplicate_count": duplicate_count,
        "exact_duplicate_rate": duplicate_rate,
        "low_information_count": low_information_count,
        "thresholds": {
            "min_unique_ratio": min_unique_ratio,
            "max_exact_duplicate_rate": max_exact_duplicate_rate,
        },
        "reasons": reasons,
    }
