"""Merging the LoRA adapter into the base model.

vLLM serves a merged model, so the trained adapter is merged into the base weights and
written to a single directory that can be baked into the Docker image.

This runs on the Colab GPU (it loads the 4.66B-parameter base model), never locally.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


def _neutralise_incompatible_torchao() -> None:
    """Hide an outdated torchao so PEFT does not trip over its version check.

    Colab ships torchao 0.10.0, while the installed PEFT requires >= 0.16.0 and
    raises ``ImportError`` (instead of reporting "unavailable") when it sees the
    old one. The LoRA merge does not need torchao, so an outdated copy is hidden
    behind an import blocker and PEFT falls back to its regular LoRA path.
    """
    import importlib.metadata as _md
    import sys as _sys

    try:
        _ver = _md.version("torchao")
    except Exception:
        return
    try:
        _parts = tuple(int(p) for p in _ver.split("+")[0].split(".")[:3])
    except Exception:
        _parts = (0,)
    while len(_parts) < 3:
        _parts = _parts + (0,)
    if _parts >= (0, 16, 0):
        return
    print(f"hiding incompatible torchao {_ver} (>=0.16.0 required by peft)",
          flush=True)

    class _BlockTorchao:
        def find_module(self, name, path=None):  # legacy protocol
            if name == "torchao" or name.startswith("torchao."):
                return self
            return None

        def find_spec(self, name, path=None, target=None):  # modern protocol
            if name == "torchao" or name.startswith("torchao."):
                raise ImportError(f"torchao {_ver} blocked: incompatible with peft")
            return None

    _sys.meta_path.insert(0, _BlockTorchao())
    for _mod in [m for m in list(_sys.modules) if m == "torchao" or m.startswith("torchao.")]:
        del _sys.modules[_mod]


def _patch_peft_torchao_check() -> None:
    """Force PEFT's torchao probe to report "not available".

    Belt and braces next to :func:`_neutralise_incompatible_torchao`: even if
    torchao remains importable, the merge path never needs it, so answer False
    instead of raising on the version mismatch.
    """
    try:
        import peft.import_utils as _iu
    except Exception:
        return
    try:
        _iu.is_torchao_available = lambda *a, **k: False  # type: ignore[assignment]
    except Exception:
        pass


def merge_adapter(checkpoint: str | Path, output_dir: str | Path,
                  model_path: str | None = None) -> Path:
    """Merge the adapter in ``checkpoint`` into its base model and write it to ``output_dir``."""
    import torch

    _neutralise_incompatible_torchao()
    from peft import PeftModel
    _patch_peft_torchao_check()
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
