"""Loading a split's own ``zaic_tools`` runtime.

Every split ships its own copy of the runtime, and the copies are **not** identical:
``train/zaic_tools/utils.py`` and ``public_test/zaic_tools/utils.py`` define different
``FREE_TEXT_KEYS`` and ``LIVE_APPS`` sets, which changes the state hash (docs/VERIFIED_FACTS.md
§12.6). The rule that follows is absolute:

    never reimplement the hash — always compute it with the split's own runtime.

Two consequences this module handles:

1. A split's runtime is imported under a unique module name (``zaic_tools__<split>``) so two
   splits can be loaded in the same process — the scorer needs exactly that (replay the gold
   trajectory for the hash, replay the submission against the same split).
2. ``ToolEnv`` is called with an explicit ``root`` so the data root never depends on the
   runtime's own default (which is derived from the package's file location).
"""
from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

from .data import Split

_loaded: dict[str, ModuleType] = {}


def load_runtime(split: Split) -> ModuleType:
    """Import ``<split>/zaic_tools`` under a unique name and return the module.

    The module is cached per split, so repeated calls are cheap and the same objects
    (and therefore the same hash implementation) are reused.
    """
    if split.name in _loaded:
        return _loaded[split.name]

    package_dir = split.tools_dir
    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        raise FileNotFoundError(
            f"split {split.name!r} has no zaic_tools runtime at {package_dir}. "
            f"The competition kit ships one runtime per split; it is required to run "
            f"and score tools."
        )

    module_name = f"zaic_tools__{split.name}"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load the runtime at {init_file}")
    module = importlib.util.module_from_spec(spec)
    # Register before exec: the package's own relative imports resolve against sys.modules.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    _loaded[split.name] = module
    return module


def tool_env(split: Split, sample_id: str):
    """Create one session on a sample's initial state.

    Exactly one session per sample — the competition forbids reading the environment
    files directly, and this is the only sanctioned way in.
    """
    runtime = load_runtime(split)
    return runtime.ToolEnv(split.name, sample_id, root=split.root)


def catalog_tools(split: Split) -> dict[str, dict]:
    """{tool_name: tool_spec} for the split's whole catalog."""
    runtime = load_runtime(split)
    return runtime.catalog_tools(split.catalog())


def schema_error(split: Split, name: str, arguments: dict, allowed=None) -> str | None:
    """The split's own schema check. None when valid, else the error message."""
    runtime = load_runtime(split)
    return runtime.schema_error(name, arguments or {}, split.catalog(), allowed)
