# Phase81 Runbook

```bash
.venv/bin/python tools/phase81_trainable_mid_model_selection.py prepare --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py train --steps 4 --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py generate --scope sanity --variant base_mid_4step_sanity --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py generate --scope sanity --variant adapter_mid_4step_sanity --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py sanity
.venv/bin/python tools/phase81_trainable_mid_model_selection.py train --steps 12 --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py generate --scope full --variant base_mid_length_control --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py generate --scope full --variant runtime_mid_length_control --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py generate --scope full --variant adapter_mid_12step_length_control --clean
.venv/bin/python tools/phase81_trainable_mid_model_selection.py full-regression
.venv/bin/python tools/phase81_trainable_mid_model_selection.py finalize
.venv/bin/python tools/phase81_trainable_mid_model_selection.py validate
```

The model revision, fresh holdout, curriculum, generation protocol, and success thresholds are frozen before training. The 12-step probe is blocked unless the 4-step training and seven-session sanity gate pass.
