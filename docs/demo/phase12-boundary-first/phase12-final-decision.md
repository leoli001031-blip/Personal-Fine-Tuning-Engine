# Phase12 Final Decision

## Base Probe

- Best model: mlx-community/Qwen3.6-27B-4bit
- Best prompt mode: boundary_first_chat_no_think
- Structure hit rate: 1.0
- Citation hit rate: 1.0
- Safety boundary rate: 1.0
- Unsupported assertions: 0
- Think leak rate: 0.0
- External law reference rate: 0.0

## Training Probe

- Training run: true
- Real training: failed
- Error type: metal_out_of_memory
- Exit code: 134
- Adapter artifact created: false
- Eval real model calls: false
- Eval recommendation: archive

## Decision

Phase12 proves that `boundary_first_chat_no_think` makes Qwen3.6-27B obey the PFE output boundary in base inference, but the 12-step 27B MLX training probe is blocked by Metal out-of-memory before an adapter artifact/eval can be produced. Do not promote. Next step should either use the boundary-first chat prompt as the inference contract, try a smaller 8B adapter training probe, or build a lower-memory 27B training/eval runner.
