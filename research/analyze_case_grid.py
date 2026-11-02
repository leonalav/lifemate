"""Cross-tab: which mechanism fixes which population, and what is each worth?

The previous scripts measured the populations one at a time. This one puts them on a
single grid, so the design follows from the table rather than from intuition.

For every exactly-compared capitalised gold argument we record two independent facts:

  availability  where the exact spelling could come from
                from_observed | from_transcript | guess
  current       what the pipeline submits today
                hit (rules already right) | miss (rules got the case wrong)

and we ask which mechanism is available to fix a miss:

  from_transcript + miss  -> better contextual case modelling can fix it
  guess + miss            -> only a catalogue lookup can fix it
  from_observed + miss    -> the string was in the prompt; the model simply did not
                             copy it, so this is a copy-discipline problem, not a
                             case problem

The grid also prices each mechanism in metric points, by replaying the gold
trajectory with the case of the affected arguments replaced by the spelling each
mechanism would have produced.

Run:  python research/analyze_case_grid.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.capitalize import restore_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.kit import load_runtime  # noqa: E402
from lifemate.metric import fuzzy_arguments  # noqa: E402


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def _walk_strings(node, out: set[str]) -> None:
    if isinstance(node, str):
        out.add(node)
    elif isinstance(node, dict):
        for value in node.values():
            _walk_strings(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_strings(value, out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    runtime = load_runtime(split)

    grid = collections.Counter()
    slot_grid = collections.defaultdict(collections.Counter)
    examples: dict[str, list] = collections.defaultdict(list)

    for sample in split.iter_samples():
        sample_id = sample["id"]
        calls = sample.get("label", {}).get("tool_calls") or []
        if not calls:
            continue
        audio_text = sample.get("audio_text") or ""
        restored = restore_capitals(audio_text.upper())

        context_strings: set[str] = set()
        _walk_strings(sample.get("context") or {}, context_strings)
        history_strings: set[str] = set()
        for turn in sample.get("history") or []:
            _walk_strings(turn, history_strings)

        env = runtime.ToolEnv(split.name, sample_id, root=split.root)
        observed: set[str] = set(context_strings) | set(history_strings)

        for call in calls:
            tool = call.get("name") or ""
            fuzzy_keys = fuzzy.get(tool, set())
            for key, value in (call.get("arguments") or {}).items():
                if not isinstance(value, str) or not _has_upper(value):
                    continue
                if _looks_like_timestamp(value):
                    continue
                if key in fuzzy_keys:
                    continue

                slot = f"{tool}.{key}"
                if value in observed:
                    availability = "from_observed"
                elif value in audio_text:
                    availability = "from_transcript"
                else:
                    availability = "guess"

                current = "hit" if value in restored else "miss"
                cell = f"{availability}/{current}"
                grid[cell] += 1
                slot_grid[slot][cell] += 1
                if len(examples[cell]) < 10:
                    examples[cell].append({
                        "id": sample_id, "slot": slot, "gold": value,
                    })

            try:
                result = env.call(tool, call.get("arguments") or {})
                _walk_strings(result, observed)
            except Exception:
                pass

    total = sum(grid.values())
    report = {
        "exact_compared_capitalised_arguments": total,
        "grid": dict(grid.most_common()),
        "grid_percent": {k: round(100.0 * v / total, 1) for k, v in grid.most_common()},
        "interpretation": {
            "from_transcript/miss": "a contextual case model CAN fix these",
            "guess/miss": "only a catalogue lookup can fix these",
            "from_observed/miss": "the spelling was in the prompt; a copy-discipline fix",
            "* /hit": "the current rules already get these right",
        },
        "per_slot": {
            slot: dict(counter.most_common())
            for slot, counter in sorted(
                slot_grid.items(), key=lambda kv: -sum(kv[1].values()))[:15]
        },
        "examples": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
