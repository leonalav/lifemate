"""Where does the correct capitalisation come from — the transcript, the environment, or neither?

This is the question that decides whether a learned case-restoration model is worth
building. For every exactly-compared argument that carries capitals in the gold
trajectory, we check three candidate sources:

  transcript   the gold value appears verbatim in the sample's own ``audio_text``
                (the human-written transcript that carries the correct casing)
  environment  the gold value appears verbatim in the sample's environment state
                (a catalogue string: a contact name, a podcast title, a province)
  both         the two agree

and one negative control:

  case_insensitive_in_transcript   the value's *tokens* occur in the transcript
                (i.e. any casing scheme that merely preserves the token is enough)

A value that is only in the *environment* cannot be recovered from the transcript at
all, and one that is only in the *transcript* cannot be recovered from the catalogue.
That split decides the design.

Run:  python research/analyze_case_origin.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import fuzzy_arguments  # noqa: E402


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def _env_strings(sample_id: str) -> set[str]:
    """Every string in the sample's environment state file."""
    path = (REPO_ROOT / "data" / "train+public_test" / "train" / "train_environment"
            / sample_id / "state.json")
    if not path.is_file():
        return set()
    out: set[str] = set()

    def walk(node):
        if isinstance(node, str):
            out.add(node)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    try:
        walk(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())

    origin = collections.Counter()
    slots = collections.Counter()
    per_slot_origin = collections.defaultdict(collections.Counter)
    examples: dict[str, list] = collections.defaultdict(list)

    for sample in split.iter_samples():
        sample_id = sample["id"]
        env_strings = _env_strings(sample_id)
        audio_text = sample.get("audio_text") or ""
        transcript_lower = audio_text.lower()

        for call in sample.get("label", {}).get("tool_calls") or []:
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
                slots[slot] += 1

                in_transcript = value in audio_text
                in_env = value in env_strings

                if in_transcript and in_env:
                    bucket = "both"
                elif in_transcript:
                    bucket = "transcript_only"
                elif in_env:
                    bucket = "environment_only"
                else:
                    bucket = "neither"

                origin[bucket] += 1
                per_slot_origin[slot][bucket] += 1
                if len(examples[bucket]) < 12:
                    examples[bucket].append({
                        "id": sample_id, "slot": slot, "value": value,
                        "tokens_in_transcript": value.lower() in transcript_lower,
                    })

    total = sum(origin.values())
    report = {
        "exact_compared_capitalised_arguments": total,
        "origin": dict(origin.most_common()),
        "origin_percent": {
            k: round(100.0 * v / total, 1) for k, v in origin.most_common()
        } if total else {},
        "per_slot": {
            slot: dict(counter.most_common())
            for slot, counter in sorted(
                per_slot_origin.items(),
                key=lambda kv: -sum(kv[1].values()),
            )[:15]
        },
        "examples": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
