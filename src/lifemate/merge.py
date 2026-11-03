"""Merging the LoRA adapter into the base model.

vLLM serves a merged model, so the trained adapter is merged into the base weights and
written to a single directory that can be baked into the Docker image.

This runs on the Colab GPU (it loads the 4.66B-parameter base model), never locally.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def merge_adapter(checkpoint: str | Path, output_dir: str | Path,
                  model_path: str | None = None) -> Path:
    """Merge the adapter in ``checkpoint`` into its base model and write it to ``output_dir``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = Path(checkpoint)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)

    base = model_path
    if base is None:
        adapter_config = checkpoint / "adapter_config.json"
        if not adapter_config.is_file():
            raise FileNotFoundError(
                f"{checkpoint} has no adapter_config.json; pass --model-path to name the "
                f"base model explicitly"
            )
        with adapter_config.open(encoding="utf-8") as fh:
            base = json.load(fh).get("base_model_name_or_path")
    if not base:
        raise ValueError("no base model to merge into")

    print(f"loading base model: {base}", flush=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base,
            dtype=torch.bfloat16,
            device_map="cpu",
        )
    except TypeError:
        # transformers < 5.0 names the argument torch_dtype.
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
        )
    print(f"applying adapter: {checkpoint}", flush=True)
    merged = PeftModel.from_pretrained(model, str(checkpoint))
    merged = merged.merge_and_unload()

    merged.save_pretrained(str(target), safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(str(target))

    # Keep the adapter config next to the merged model: the inference entry point reads
    # the base model name from it.
    adapter_config = checkpoint / "adapter_config.json"
    if adapter_config.is_file():
        shutil.copy2(adapter_config, target / "adapter_config.json")

    print(f"merged model written to {target}", flush=True)
    return target
