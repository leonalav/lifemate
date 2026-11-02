"""Price each capitalisation mechanism separately, and diagnose why the rules miss.

The grid established *who* can fix each affected argument. This script prices them:
the gold trajectory is replayed three more times, each time repairing only one
mechanism's population, so the metric cost of leaving each one broken is measured
rather than estimated.

  fix_case_model     repairs every argument whose words occur in the utterance
                     (the population a contextual case model could reach)
  fix_catalogue      repairs every argument whose words do not occur in the utterance
                     (a catalogue lookup: currency codes, provinces, cities)
  fix_copy           repairs every argument whose spelling was already in the prompt
                     (a copy-discipline fix: the model did not copy what it was shown)

It also diagnoses the case-model misses, because "a context model would fix it" is only
true if the miss has a cause a context model can act on:

  missing_vocab      the word never appears capitalised in the training transcripts
  ambiguous          the word appears in both cases in training, so the case is a
                     genuine context decision
  segment            the words are right but the phrase boundary is wrong

Run:  python research/analyze_case_mechanism_price.py --data-dir data/train+public_test
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


def _as_spoken(gold_value: str, restored: str, audio_text: str) -> str | None:
    """The spelling the current pipeline submits: the gold value as the restored
    transcript spells it. None when the transcript does not contain the words."""
    if gold_value.lower() not in audio_text.lower():
        return None
    index = restored.lower().find(gold_value.lower())
    if index < 0:
        return None
    return restored[index:index + len(gold_value)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    runtime = load_runtime(split)

    # the vocabulary of the training transcripts, by case
    upper_seen: collections.Counter = collections.Counter()
    lower_seen: collections.Counter = collections.Counter()
    for sample in split.iter_samples():
        for word in WORD.findall(sample.get("audio_text") or ""):
            if word[0].isupper():
                upper_seen[word.lower()] += 1
            else:
                lower_seen[word.lower()] += 1

    totals = collections.Counter()
    samples = 0
    causes = collections.Counter()
    cause_examples: dict[str, list] = collections.defaultdict(list)

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

        # classify every affected argument, and record which mechanism could fix it
        classes: dict[int, list[tuple[str, str]]] = collections.defaultdict(list)

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

                if value in restored:
                    continue        # already right, nothing to fix
                if value in observed:
                    mechanism = "fix_copy"
                elif value.lower() in audio_text.lower():
                    mechanism = "fix_case_model"
                else:
                    mechanism = "fix_catalogue"
                classes[index].append((key, mechanism))

                if mechanism == "fix_case_model":
                    gold_words = [w.lower() for w in WORD.findall(value)]
                    if all(w in upper_seen for w in gold_words):
                        cause = "missing_vocab"
                    elif any(w in lower_seen and w in upper_seen for w in gold_words):
                        cause = "ambiguous"
                    else:
                        cause = "segment"
                    causes[cause] += 1
                    if len(cause_examples[cause]) < 12:
                        cause_examples[cause].append({
                            "id": sample_id, "slot": f"{tool}.{key}", "gold": value,
                            "restored": restored[:140],
                        })

            try:
                result = env.call(tool, call.get("arguments") or {})
                _walk_strings(result, observed)
            except Exception:
                pass

        samples += 1
        gold_hash = replay_hash(split, sample_id, calls)
        totals["gold"] += sample_score(calls, calls, gold_hash, gold_hash, fuzzy)

        # The broken baseline: the pipeline's spelling for every affected argument.
        # A mechanism is priced by repairing *its* population and leaving the others
        # broken, so the price is the points that population alone is worth.
        baseline = []
        for index, call in enumerate(calls):
            arguments = dict(call.get("arguments") or {})
            for key, _mech in classes.get(index, []):
                spoken = _as_spoken(call["arguments"][key], restored, audio_text)
                arguments[key] = spoken if spoken is not None else call["arguments"][key]
            baseline.append({"name": call.get("name"), "arguments": arguments})
        baseline_hash = replay_hash(split, sample_id, baseline)
        totals["baseline"] += sample_score(
            baseline, calls, baseline_hash, gold_hash, fuzzy)

        for mechanism in ("fix_case_model", "fix_catalogue", "fix_copy"):
            variant = json.loads(json.dumps(baseline))
            for index, call in enumerate(calls):
                for key, mech in classes.get(index, []):
                    if mech == mechanism:
                        variant[index]["arguments"][key] = call["arguments"][key]
            pred_hash = replay_hash(split, sample_id, variant)
            totals[mechanism] += sample_score(
                variant, calls, pred_hash, gold_hash, fuzzy)

    n = samples or 1
    report = {
        "samples": samples,
        "gold": round(100.0 * totals["gold"] / n, 3),
        "baseline_all_broken": round(100.0 * totals["baseline"] / n, 3),
        "score_when_mechanism_is_broken": {
            mechanism: round(100.0 * totals[mechanism] / n, 3)
            for mechanism in ("fix_case_model", "fix_catalogue", "fix_copy")
        },
        "price_of_each_mechanism_in_points": {
            mechanism: round(100.0 * (totals["gold"] - totals[mechanism]) / n, 3)
            for mechanism in ("fix_case_model", "fix_catalogue", "fix_copy")
        },
        "case_model_miss_causes": dict(causes.most_common()),
        "cause_examples": cause_examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
