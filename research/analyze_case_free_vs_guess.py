"""Is the correct capitalisation free, or must it be guessed?

This decides the whole design. Three ways an agent can produce a correctly-capitalised
exactly-compared argument:

  free_prompt   the exact string is already in the prompt (context, history, or an
                 earlier *gold* tool result the agent will make anyway)
  needs_read    the exact string is only in the environment, and reaching it costs an
                 extra tool call — which costs precision
  needs_guess   neither: the agent must infer the case from the utterance

The gold trajectory tells us which reads happen anyway. Where a read precedes the
capitalised write in gold, the catalogue is free: the agent sees the exact spelling in
the tool result and copies it. Where it does not, the agent is guessing, and that is
where the case of a context-aware model would matter.

The script also prices the alternative: what one extra unmatched read call costs in
sample F1, measured with the official metric.

Run:  python research/analyze_case_free_vs_guess.py --data-dir data/train+public_test
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
from lifemate.kit import load_runtime  # noqa: E402


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
    samples_seen = 0

    for sample in split.iter_samples():
        if args.limit and samples_seen >= args.limit:
            break
        sample_id = sample["id"]
        calls = sample.get("label", {}).get("tool_calls") or []
        if not calls:
            continue

        prompt_strings: set[str] = set()
        _walk_strings(sample.get("context") or {}, prompt_strings)
        for turn in sample.get("history") or []:
            _walk_strings(turn, prompt_strings)

        # Replay the gold trajectory, accumulating what the agent would see.
        env = runtime.ToolEnv(split.name, sample_id, root=split.root)
        seen_before: list[set[str]] = []
        results: list[set[str]] = []
        for call in calls:
            seen_before.append(set(prompt_strings))
            strings: set[str] = set()
            try:
                result = env.call(call.get("name", ""), call.get("arguments") or {})
                _walk_strings(result, strings)
            except Exception:
                pass
            results.append(strings)
            prompt_strings |= strings

        has_targets = False
        for index, call in enumerate(calls):
            tool = call.get("name") or ""
            fuzzy_keys = fuzzy.get(tool, set())
            for key, value in (call.get("arguments") or {}).items():
                if not isinstance(value, str) or not _has_upper(value):
                    continue
                if _looks_like_timestamp(value):
                    continue
                if key in fuzzy_keys:
                    continue

                has_targets = True
                slot = f"{tool}.{key}"
                available = seen_before[index] | results[index]
                if value in available:
                    bucket = "free"            # already visible when the write happens
                elif value.lower() in {s.lower() for s in available}:
                    bucket = "free_but_wrong_case"   # the words are visible, case differs
                else:
                    bucket = "needs_guess"      # the string itself is nowhere visible
                verdict[bucket] += 1
                per_slot[slot][bucket] += 1
                if len(examples[bucket]) < 15:
                    examples[bucket].append({
                        "id": sample_id, "slot": slot, "value": value,
                    })

        if has_targets:
            samples_seen += 1

    total = sum(verdict.values())
    report = {
        "samples_with_capitalised_exact_args": samples_seen,
        "verdict": dict(verdict.most_common()),
        "verdict_percent": {
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
