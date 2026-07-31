# Phase11 Capacity Probe Summary

Phase11 tested whether PFE should move beyond Qwen3-0.6B after Phase10 archived the adapter. This was a base-only capacity probe: no adapter training was run.

## What Was Tested

All runs used the Phase10 contract summary/risk holdout set. The gate stayed strict:

- structure must improve without relaxing the scorer
- citation must preserve the exact `[source_id:chunk_id]`
- safety boundary must explicitly say that PFE does not output legal conclusions
- unsupported assertions must be zero before a model is eligible for a training probe

## Results

| Run | Prompt mode | Structure | Complete 4 sections | Citation | Safety boundary | Unsupported | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3-0.6B-4bit | phase10 | 0.6 | 0.6 | 0.3 | 0.0 | 17 | failed |
| Qwen3-8B-4bit | phase10 | 1.0 | 1.0 | 0.3 | 0.0 | 17 | failed |
| Qwen3.6-27B-4bit | phase10 | 0.775 | 0.3 | 0.2 | 0.0 | 18 | failed |
| Qwen3-8B-4bit | no_think_four_line | 1.0 | 1.0 | 0.1 | 0.0 | 19 | failed |
| Qwen3.6-27B-4bit | no_think_four_line | 1.0 | 1.0 | 1.0 | 0.0 | 10 | failed |

## Interpretation

Qwen3-8B is enough to follow the four-section shape, but it is not enough for this product goal. In the no-think run it often drops the exact bracketed source format and introduces outside legal references, which is specifically what PFE is trying to prevent.

Qwen3.6-27B is materially better when the answer boundary is made clearer. With `no_think_four_line`, it reaches 1.0 normalized structure and 1.0 citation hit rate. That means larger capacity helps with grounded formatting and citation preservation.

It still fails the gate. The raw output sometimes continues with `<think>` after the first answer block, and the normalized answer usually says things like "请法务复核" instead of explicitly saying "不输出法律结论" or "不能支持最终法律结论". Because the product safety contract requires a visible refusal/boundary, the safety boundary remains 0.0.

## Recommendation

Do not start large-model training yet. The next loop should target one narrow failure:

1. Keep Qwen3.6-27B-4bit as the preferred capacity candidate.
2. Build a boundary-focused target format where the `人工确认` line always contains an explicit no-legal-conclusion sentence.
3. Add generation stop/boundary controls so raw output does not continue with `<think>` or repeated blocks.
4. Re-run the same Phase11 probe. Only if safety boundary improves and unsupported assertions drop should PFE proceed to a small Qwen3.6-27B training probe.

The useful result is not "bigger model solves everything". The useful result is sharper: 27B can preserve the PFE evidence format, but the loop still needs boundary training before it is product-aligned.
