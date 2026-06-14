#!/usr/bin/env python3
"""Prepare a tiny local Hugging Face-style causal LM for release smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _default_output_dir() -> Path:
    return Path.home() / ".cache" / "pfe" / "release-models" / "tiny-gpt2-local"


def prepare_tiny_model(output_dir: Path) -> dict[str, str]:
    try:
        from transformers import GPT2Config, GPT2LMHeadModel
    except Exception as exc:
        raise RuntimeError("prepare_tiny_hf_model requires transformers with GPT2 support") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    config = GPT2Config(
        vocab_size=128,
        n_positions=48,
        n_ctx=48,
        n_embd=32,
        n_layer=1,
        n_head=1,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(str(output_dir), safe_serialization=True)

    marker_path = output_dir / "pfe_tiny_model_manifest.json"
    marker = {
        "kind": "pfe_tiny_hf_model",
        "architecture": "GPT2LMHeadModel",
        "purpose": "local release smoke",
        "output_dir": str(output_dir),
    }
    marker_path.write_text(json.dumps(marker, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "config": str(output_dir / "config.json"),
        "model": str(output_dir / "model.safetensors"),
        "manifest": str(marker_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a tiny local HF model for PFE real-local smoke tests.")
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    args = parser.parse_args()

    summary = prepare_tiny_model(args.output_dir.expanduser().resolve())
    print("TINY HF MODEL READY")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
