"""Split discovery and sample loading.

A "data root" holds one or more split directories. The released kit ships:

    <root>/train+public_test/train/{train.jsonl, train_tools_catalog.json,
                                     train_environment/, zaic_tools/, audio/}
    <root>/train+public_test/public_test/...

but the private test arrives later as a new folder, so the split name is never
hardcoded anywhere: it is discovered from the directory listing (docs/PRD.md §6).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SPLIT_MARKERS = ("_tools_catalog.json", "_environment")


@dataclass(frozen=True)
class Split:
    """One split directory and the data root that contains it."""

    name: str
    path: Path
    root: Path

    @property
    def samples_file(self) -> Path:
        return self.path / f"{self.name}.jsonl"

    @property
    def catalog_file(self) -> Path:
        return self.path / f"{self.name}_tools_catalog.json"

    @property
    def environment_dir(self) -> Path:
        return self.path / f"{self.name}_environment"

    @property
    def tools_dir(self) -> Path:
        return self.path / "zaic_tools"

    @property
    def audio_dir(self) -> Path:
        return self.path / "audio"

    def sample(self, sample_id: str) -> dict:
        """Load one sample record by id (linear scan; the files are a few MB)."""
        with self.samples_file.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("id") == sample_id:
                    return record
        raise KeyError(f"{sample_id!r} not found in {self.samples_file}")

    def iter_samples(self):
        """Yield every sample record in file order."""
        if not self.samples_file.is_file():
            return
        with self.samples_file.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    yield json.loads(line)

    def ids(self) -> list[str]:
        return [s["id"] for s in self.iter_samples()]

    def catalog(self) -> dict:
        """The split's tool catalog: {"schema_version", "note", "apps": [...] }."""
        with self.catalog_file.open(encoding="utf-8") as fh:
            return json.load(fh)


def _looks_like_split(split_dir: Path) -> bool:
    """True when ``split_dir`` holds a split: <split>.jsonl + <split>_tools_catalog.json."""
    if not split_dir.is_dir():
        return False
    catalog = split_dir / f"{split_dir.name}_tools_catalog.json"
    if not catalog.is_file():
        return False
    return (split_dir / f"{split_dir.name}.jsonl").is_file()


def discover_splits(data_dir: str | Path, max_depth: int = 3) -> dict[str, Split]:
    """Find every split under ``data_dir``.

    Handles both layouts the kit can arrive in:
      * ``<data_dir>/<split>/`` (split folders at the top level)
      * ``<data_dir>/<bundle>/<split>/`` (a bundle folder such as train+public_test)

    Returns ``{split_name: Split}`` sorted by name. Raises FileNotFoundError with
    an actionable message when nothing is found.
    """
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data dir not found: {root}")

    found: dict[str, Split] = {}

    def walk(directory: Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        for entry in entries:
            if not entry.is_dir():
                continue
            if _looks_like_split(entry):
                found[entry.name] = Split(
                    name=entry.name, path=entry, root=directory
                )
                continue
            walk(entry, depth + 1)

    walk(root, 0)

    if not found:
        raise FileNotFoundError(
            f"No split found under {root}. Expected a folder named <split> that "
            f"contains <split>.jsonl and <split>_tools_catalog.json "
            f"(the released kit nests them under a bundle folder such as "
            f"'train+public_test')."
        )
    return found


def resolve_split(data_dir: str | Path, split: str | None) -> Split:
    """Resolve one split by name, or the only split present when ``split`` is None."""
    splits = discover_splits(data_dir)
    if split:
        if split not in splits:
            raise KeyError(
                f"split {split!r} not found under {data_dir}; available: "
                f"{', '.join(sorted(splits))}"
            )
        return splits[split]
    if len(splits) == 1:
        return next(iter(splits.values()))
    raise KeyError(
        f"{data_dir} contains {len(splits)} splits ({', '.join(sorted(splits))}); "
        f"pass --split to choose one"
    )


def installed_tools(sample: dict) -> list[str]:
    """Every tool installed on this sample, plus the always-available meta tool."""
    apps = sample.get("context", {}).get("installed_apps", {}) or {}
    tools: list[str] = []
    for app in apps.values():
        tools.extend(app.get("tools", []) or [])
    return tools
