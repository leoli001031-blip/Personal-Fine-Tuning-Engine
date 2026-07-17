# Phase80 Runbook

```bash
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py prepare --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py train --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant base_0_5b_minimal --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant runtime_0_5b --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant phase79_high_lr_adapter --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant phase80_low_lr_adapter --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant phase79_high_lr_stop_control --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant base_4b_minimal --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py generate --variant runtime_4b --clean
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py full-regression
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py finalize
.venv/bin/python tools/phase80_small_model_failure_taxonomy.py validate
```

The holdout and hypotheses are frozen before low-LR training. The Phase79 high-LR adapter is a read-only historical comparison, never the new candidate. Stop control changes only max generation tokens and repetition penalty, while all scores use the unchanged Phase75 deterministic rubric. This phase cannot promote or change defaults.
