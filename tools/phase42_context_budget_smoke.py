#!/usr/bin/env python3
"""Run a real tokenizer/model smoke around the configured 4K context boundary."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_ROOT = REPO_ROOT / "pfe-core"
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from pfe_core.inference.engine import InferenceConfig, InferenceEngine


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-model",
        type=Path,
        default=REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument("--context-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "docs"
            / "demo"
            / "phase42-trustworthy-training-runtime-hardening"
            / "evidence-hermes-streaming"
            / "context_budget_smoke.json"
        ),
    )
    args = parser.parse_args()
    base_model = args.base_model.expanduser().resolve()
    os.environ["PFE_ENABLE_REAL_LOCAL_INFERENCE"] = "1"
    os.environ["PFE_MAX_CONTEXT_TOKENS"] = str(max(256, args.context_tokens))
    os.environ["PFE_MAX_OUTPUT_TOKENS"] = "64"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=True)
    parts: list[str] = []
    token_count = 0
    index = 0
    target = max(4000, args.context_tokens + 100)
    while token_count < target:
        parts.append(f"证据片段{index}：本段用于验证上下文预算和左侧截断。")
        index += 1
        if index % 25 == 0:
            token_count = len(tokenizer("\n".join(parts), add_special_tokens=False)["input_ids"])
    prompt = "\n".join(parts) + "\n请只回答：预算检查完成。"

    engine = InferenceEngine(InferenceConfig(base_model=str(base_model)))
    response = engine.generate(
        [{"role": "user", "content": prompt}],
        max_tokens=max(1, args.max_tokens),
        temperature=0,
        metadata={"enable_real_local": True},
    )
    generation = dict(engine.status().get("generation") or {})
    budget = dict(generation.get("token_budget") or {})
    passed = (
        generation.get("served_by") == "local"
        and int(budget.get("original_prompt_tokens") or 0) >= 4000
        and int(budget.get("prompt_tokens") or 0) <= int(budget.get("effective_context_tokens") or 0)
        and budget.get("input_truncated") is True
        and int(budget.get("effective_context_tokens") or 0) == max(256, args.context_tokens)
    )
    payload = {
        "kind": "phase42_real_context_budget_smoke",
        "passed": passed,
        "base_model": str(base_model),
        "response": response,
        "finish_reason": generation.get("finish_reason"),
        "token_budget": budget,
        "claim_boundary": "This proves the configured 4K path only; it does not claim the model maximum context.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("Phase42 real context budget smoke failed")
    print("PHASE42 REAL CONTEXT BUDGET SMOKE PASSED")
    print(f"original_prompt_tokens: {budget.get('original_prompt_tokens')}")
    print(f"prompt_tokens: {budget.get('prompt_tokens')}")
    print(f"effective_context_tokens: {budget.get('effective_context_tokens')}")
    print(f"effective_max_new_tokens: {budget.get('effective_max_new_tokens')}")
    print(f"input_truncated: {budget.get('input_truncated')}")
    print(f"finish_reason: {generation.get('finish_reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
