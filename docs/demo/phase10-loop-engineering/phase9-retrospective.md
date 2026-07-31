# Phase9 Retrospective

Phase9 proved the real training loop can run, but it did not prove the adapter
is ready to promote.

## Evidence Read

- Evidence source: `docs/demo/phase9-output-format-training-stability/evidence-real-qwen3-0.6b/`
- Phase9 Qwen3-0.6B training completed with MLX.
- Training metadata reports `dataset_format=prompt_completion_output_only_loss` and `output_only_loss_masking=true`.
- The Phase9 adapter was correctly archived.

## Phase9 Scores

| metric | base | adapter | delta |
|---|---:|---:|---:|
| citation hit rate | 0.7 | 0.3 | -0.4 |
| structure hit rate | 0.725 | 0.325 | -0.4 |
| safety boundary rate | 0.0 | 0.0 | 0.0 |
| unsupported assertions | 13 | 17 | -4 |

## Diagnosis

Phase9 should not be described as a backend masking failure. The MLX backend is
already using prompt/completion output-only masking for prompt-completion rows.
The remaining failure is more likely in the product loop variables:

- the target completion shape is still too hard for Qwen3-0.6B to reproduce
  consistently;
- the adapter tends to learn partial labels, repeated boundary words, or copied
  prompt instructions instead of the complete four-line answer;
- eval must preserve raw output and score only real four-section output, with
  normalization limited to truncating the first complete block.

Phase10 therefore starts with format-only curriculum before any larger model
or richer contract data.
