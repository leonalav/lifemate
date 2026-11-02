"""The final capitalisation analysis: what is at stake, and what fixes it.

Everything above is consolidated here, with the baseline bugs fixed.

**Marginal price per population.** The gold trajectory is replayed with exactly one
population's case broken (lowercased) and the rest left at gold. The score drop is that
population's price. Populations:

  transcript   the value's words occur in the utterance, so the case is recoverable
               from the transcript — this is what a case-restoration model addresses
  catalogue    the value's words do NOT occur in the utterance ("USD", "Đồng Nai"):
               only the environment's catalogue has the spelling
  observed     the exact spelling was already in an earlier tool result

**Why the current rules miss.** For the transcript population the rules get the case
wrong. The cause decides whether a context-aware model can help:

  ambiguous      the word appears in both cases in the training transcripts, so the case
                 is a genuine context decision ("Minh" the name vs "minh" the pronoun)
  missing_vocab  the word is capitalised in the training transcripts but the derived
                 vocabulary does not carry it — a resource gap, not a modelling gap
  segment        the words are right, the phrase boundary is wrong

Run:  python research/analyze_case_final.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.capitalize import restore_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.kit import load_runtime  # noqa: E402
from lifemate.metric import (  # noqa: E402
    fuzzy_arguments,
    replay_hash,
    sample_score,
)

WORD = re.compile(r"[^\W_]+", re.UNICODE)


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


def _population(value: str, audio_text: str, observed: set[str]) -> str:
    if value in observed:
        return "observed"
    if value.lower() in audio_text.lower():
        return "transcript"
    return "catalogue"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    runtime = load_runtime(split)

    upper_seen: collections.Counter = collections.Counter()
    lower_seen: collections.Counter = collections.Counter()
    for sample in split.iter_samples():
        for word in WORD.findall(sample.get("audio_text") or ""):
            (upper_seen if word[0].isupper() else lower_seen)[word.lower()] += 1

    from lifemate.capitalize import _vocabulary
    proper, acronyms, ordinary = _vocabulary()

    totals = collections.Counter()
    samples = 0
    population_counts = collections.Counter()
    causes = collections.Counter()
    cause_examples: dict[str, list] = collections.defaultdict(list)
    catalogue_slots = collections.Counter()
    catalogue_values: collections.Counter = collections.Counter()

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

        classes: dict[int, list[tuple[str, str, str]]] = collections.defaultdict(list)

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

                population = _population(value, audio_text, observed)
                population_counts[population] += 1
                classes[index].append((key, population, value))

                if population == "catalogue":
                    catalogue_slots[f"{tool}.{key}"] += 1
                    catalogue_values[value] += 1

                if population == "transcript":
                    index_in_restored = restored.lower().find(value.lower())
                    spoken = (restored[index_in_restored:index_in_restored + len(value)]
                              if index_in_restored >= 0 else None)
                    if spoken == value:
                        continue        # the rules already get it right
                    words = [w.lower() for w in WORD.findall(value)]
                    if any(w in lower_seen and w in upper_seen for w in words):
                        cause = "ambiguous"
                    elif all(w in upper_seen for w in words) and all(
                            w not in proper and w not in ordinary for w in words):
                        cause = "missing_vocab"
                    else:
                        cause = "segment"
                    causes[cause] += 1
                    if len(cause_examples[cause]) < 15:
                        cause_examples[cause].append({
                            "id": sample_id, "slot": f"{tool}.{key}",
                            "gold": value, "pipeline": spoken,
                        })

            try:
                result = env.call(tool, call.get("arguments") or {})
                _walk_strings(result, observed)
            except Exception:
                pass

        samples += 1
        gold_hash = replay_hash(split, sample_id, calls)
        totals["gold"] += sample_score(calls, calls, gold_hash, gold_hash, fuzzy)

        for population in ("transcript", "catalogue", "observed"):
            variant = []
            for index, call in enumerate(calls):
                arguments = dict(call.get("arguments") or {})
                for key, pop, value in classes.get(index, []):
                    if pop == population:
                        arguments[key] = value.lower()
                variant.append({"name": call.get("name"), "arguments": arguments})
            pred_hash = replay_hash(split, sample_id, variant)
            totals[f"break_{population}"] += sample_score(
                variant, calls, pred_hash, gold_hash, fuzzy)

    n = samples or 1
    report = {
        "samples": samples,
        "gold": round(100.0 * totals["gold"] / n, 3),
        "population_counts": dict(population_counts.most_common()),
        "population_percent": {
            k: round(100.0 * v / sum(population_counts.values()), 1)
            for k, v in population_counts.most_common()
        },
        "marginal_price_in_points": {
            population: round(
                100.0 * (totals["gold"] - totals[f"break_{population}"]) / n, 3)
            for population in ("transcript", "catalogue", "observed")
        },
        "case_model_miss_causes": dict(causes.most_common()),
        "catalogue_slots": catalogue_slots.most_common(12),
        "catalogue_values": catalogue_values.most_common(25),
        "cause_examples": cause_examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
