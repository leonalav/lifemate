"""Is the correct capitalisation free to obtain, or must it be guessed?

This decides the design. For every exactly-compared argument that carries capitals, the
agent must produce the exact string. Three ways it can:

  from_transcript  the string occurs in the utterance itself, so restoring the case on
                   the transcript is enough (this is what a truecasing model would do)
  from_observed    the string occurred in an *earlier* tool result the agent already
                   has in its prompt, so it can copy the spelling verbatim
  from_context     the string is in the sample's context (a city, an address, a name)
  guess            none of the above — the spelling must be invented

The distinction that matters: ``from_observed`` is free and deterministic (the agent
already made the call, the string is in its prompt), while ``guess`` is the only place a
learned context model could help.

A previous pass counted a string as available when it appeared in the result of the very
call it was an argument to. That is circular — the agent cannot use a result it has not
received yet — so availability here is measured strictly *before* the call.

Run:  python research/analyze_case_availability.py --data-dir data/train+public_test
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
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    runtime = load_runtime(split)

    verdict = collections.Counter()
    per_slot = collections.defaultdict(collections.Counter)
    examples: dict[str, list] = collections.defaultdict(list)

    for sample in split.iter_samples():
        if args.limit and verdict.total() > 4000:
            break
        sample_id = sample["id"]
        calls = sample.get("label", {}).get("tool_calls") or []
        if not calls:
            continue

        context_strings: set[str] = set()
        _walk_strings(sample.get("context") or {}, context_strings)
        history_strings: set[str] = set()
        for turn in sample.get("history") or []:
            _walk_strings(turn, history_strings)

        audio_text = sample.get("audio_text") or ""

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
                    bucket = "from_observed"
                elif value in context_strings:
                    bucket = "from_context"
                elif value in audio_text:
                    bucket = "from_transcript"
                else:
                    bucket = "guess"

                verdict[bucket] += 1
                per_slot[slot][bucket] += 1
                if len(examples[bucket]) < 15:
                    examples[bucket].append({
                        "id": sample_id, "slot": slot, "value": value,
                    })

            # the result becomes visible only after this call
            try:
                result = env.call(tool, call.get("arguments") or {})
                _walk_strings(result, observed)
            except Exception:
                pass

    total = sum(verdict.values())
    report = {
        "exact_compared_capitalised_arguments": total,
        "availability": dict(verdict.most_common()),
        "availability_percent": {
            k: round(100.0 * v / total, 1) for k, v in verdict.most_common()
        } if total else {},
        "per_slot": {
            slot: dict(counter.most_common())
            for slot, counter in sorted(
                per_slot.items(), key=lambda kv: -sum(kv[1].values()))[:18]
        },
        "examples": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
