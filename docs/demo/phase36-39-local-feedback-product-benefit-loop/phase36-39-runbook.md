# Phase36-39 Runbook

Generate the full local feedback product-benefit loop evidence:

```bash
.venv/bin/python tools/phase36_39_local_feedback_product_benefit_loop.py --clean-evidence
```

This run does not integrate Hermes and does not train 27B. It separates the actual feedback lane from the simulated lab lane. Without at least 12 approved actual local interactions, product-benefit claims remain lab-only.
