"""Which copy-source is already visible to the model, at zero extra cost?

For every exactly-compared argument that carries capitals in the gold trajectory, we ask
where the agent could have *copied* the exact string from, given only what the policy
model already sees in its prompt:

  transcript   the sample's ``audio_text`` (the reference transcript with correct
               casing) — NOT available at inference: the ASR emits uppercase
  context      ``sample['context']`` — the user profile, the address, the city.
               Always in the prompt. Zero extra cost.
  environment  the environment state — only reachable by making a tool call.
               An extra call costs precision.

The first row is what a learned case-restoration model would have to guess. The second is
free. The third is purchasable at a measurable price.

Run:  python research/analyze_case_source_visible.py --data-dir data/train+public_test
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


def _walk_strings(node, out: set[str]) -> None:
    if isinstance(node, str):
        out.add(node)
    elif isinstance(node, dict):
        for value in node.values():
            _walk_strings(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_strings(value, out)


def _env_strings(sample_id: str) -> set[str]:
    path = (REPO_ROOT / "data" / "train+public_test" / "train" / "train_environment"
            / sample_id / "state.json")
    if not path.is_file():
        return set()
    out: set[str] = set()
    try:
        _walk_strings(json.loads(path.read_text(encoding="utf-8")), out)
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

    visible = collections.Counter()
    per_slot = collections.defaultdict(collections.Counter)
    gaps: list[dict] = []

    for sample in split.iter_samples():
        sample_id = sample["id"]
        env_strings = _env_strings(sample_id)
        audio_text = sample.get("audio_text") or ""
        context_strings: set[str] = set()
        _walk_strings(sample.get("context") or {}, context_strings)
        # the history is also in the prompt
        history_strings: set[str] = set()
        for turn in sample.get("history") or []:
            _walk_strings(turn, history_strings)

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
                in_transcript = value in audio_text
                in_context = value in context_strings
                in_history = value in history_strings
                in_env = value in env_strings

                if in_context or in_history:
                    bucket = "prompt_free"          # already in the prompt
                elif in_transcript:
                    bucket = "needs_transcript"     # needs the reference transcript
                elif in_env:
                    bucket = "needs_env_call"       # needs one extra read call
                else:
                    bucket = "unreachable"          # must be guessed

                visible[bucket] += 1
                per_slot[slot][bucket] += 1
                if bucket in ("unreachable", "needs_env_call") and len(gaps) < 25:
                    gaps.append({
                        "id": sample_id, "slot": slot, "value": value,
                        "bucket": bucket,
                        "in_transcript": in_transcript,
                    })

    total = sum(visible.values())
    report = {
        "exact_compared_capitalised_arguments": total,
        "visibility": dict(visible.most_common()),
        "visibility_percent": {
            k: round(100.0 * v / total, 1) for k, v in visible.most_common()
        } if total else {},
        "per_slot": {
            slot: dict(counter.most_common())
            for slot, counter in sorted(
                per_slot.items(), key=lambda kv: -sum(kv[1].values()))[:18]
        },
        "gaps": gaps,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
