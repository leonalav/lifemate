"""Can the exact spelling of an argument be recovered from what the agent has already seen?

The review's first proposal: contact names, device names and playlist names already
appear in the tool results of the conversation so far. If the model proposes
``"mẹ"`` and the agent has already seen ``"Mẹ"``, the argument can be snapped to the
observed spelling — which fixes capitalisation and ASR tone errors in one move.

This script measures the proposal on the train split, without touching the environment
files: the observable surface is what the agent legitimately sees at runtime —

  * the tool results in the sample's history (the conversation so far)
  * the sample's context (the user profile, the location)

The rule is deliberately strict, because a wrong snap is worse than no snap: the
argument is compared with exact equality, so a snap that changes the *words* breaks a
call that was already right. So a candidate is only accepted when it matches the
proposal after folding away the differences the review wants fixed — case and tone
marks — and not the differences that carry meaning — the letters themselves.

    fold(s) = lowercase, diacritics stripped, whitespace collapsed

A proposal matches an observed string when ``fold(proposal) == fold(observed)``. The
observed spelling is then emitted.

The population is the exactly-compared (non-fuzzy) string arguments of the gold
trajectories — the ones the metric compares with equality, where a case or tone error
costs the whole argument.

Run:  python research/analyze_observed_snap.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import fuzzy_arguments  # noqa: E402

WORD = re.compile(r"[^\W_]+", re.UNICODE)


def fold(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace.

    Folding away the diacritics is what makes the snap tolerant of a tone error, but
    it also makes two *different* Vietnamese words collide ("hoa"/"hòa" are the same
    word, but "hòa"/"hoà" are two spellings of it — the point). Vietnamese is
    tonal, so a folded match is a candidate, never a proof: the script measures how
    often the observed spelling is the one the gold uses.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.split())


def strings_in(value) -> list[str]:
    """Every string that appears anywhere in a tool result or the context."""
    found: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            found.extend(strings_in(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(strings_in(item))
    elif isinstance(value, str):
        found.append(value)
    return found


def observed_surface(sample: dict) -> list[str]:
    """The strings the agent has seen before it answers this turn.

    This is exactly what the runtime has: the tool results of the conversation so far
    and the sample's context. Nothing under ``<split>_environment/`` is read.
    """
    out: list[str] = []
    for turn in sample.get("history") or []:
        for result in turn.get("tool_results") or []:
            out.extend(strings_in(result))
    out.extend(strings_in(sample.get("context") or {}))
    return out


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    samples = list(split.iter_samples())

    stats = collections.Counter()
    by_tool = collections.Counter()
    examples: list[tuple[str, str, str, str]] = []

    for sample in samples:
        surface = observed_surface(sample)
        # The distinct observed spellings, by folded key. The first spelling seen wins;
        # the count of distinct spellings per folded key is a measure of the ambiguity.
        by_fold: dict[str, list[str]] = {}
        for text in surface:
            if not text.strip():
                continue
            key = fold(text)
            if not key:
                continue
            bucket = by_fold.setdefault(key, [])
            if text not in bucket:
                bucket.append(text)

        for call in sample.get("label", {}).get("tool_calls") or []:
            tool = call.get("name") or ""
            fuzzy_keys = fuzzy.get(tool, set())
            for key, value in (call.get("arguments") or {}).items():
                if not isinstance(value, str) or not value.strip():
                    continue
                if key in fuzzy_keys:
                    continue
                if _looks_like_timestamp(value):
                    continue
                if not _has_upper(value):
                    continue        # only the capitalised arguments are at risk
                stats["population"] += 1

                folded = fold(value)
                bucket = by_fold.get(folded)
                if not bucket:
                    stats["not_observed"] += 1
                    continue
                stats["observed"] += 1
                by_tool[tool] += 1
                if len(bucket) > 1:
                    stats["ambiguous_observed"] += 1
                if value in bucket:
                    stats["exact_present"] += 1
                else:
                    stats["different_spelling"] += 1
                    if len(examples) < 15:
                        examples.append((tool, key, value, " | ".join(bucket[:3])))

    total = stats["population"] or 1
    observed = stats["observed"] or 1
    report = {
        "split": args.split,
        "population_exactly_compared_capitalised_arguments": stats["population"],
        "observed_at_runtime": stats["observed"],
        "observed_percent": round(100.0 * stats["observed"] / total, 1),
        "of_observed_exact_spelling_present": stats["exact_present"],
        "of_observed_a_different_spelling_was_observed": stats["different_spelling"],
        "ambiguous_observed_folded_key": stats["ambiguous_observed"],
        "not_observed_at_runtime": stats["not_observed"],
        "top_tools_with_an_observed_match": by_tool.most_common(15),
        "examples_different_spelling": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
