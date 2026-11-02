"""How small can the artifact get before the score moves?

The exported model is 458 KiB of weights. Pruning the small ones shrinks it, but the
weights that matter are the ones the model relies on for the ambiguous tokens. This
sweep measures the metric score at each pruning threshold so the size is chosen on
evidence rather than taste.

It also measures the same sweep with the token-identity feature removed, because that
feature is the memorisation-prone one: it is what lets the model remember "minh" from
training. The public test uses names the training split never saw, so the honest
question is how much of the gain survives without it.

Run:  python research/prune_case_model.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.capitalize import restore_capitals  # noqa: E402
from lifemate.case_model import CaseModel, featurize, refine_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import (  # noqa: E402
    fuzzy_arguments,
    replay_hash,
    sample_score,
)


def _is_cap(token: str) -> bool:
    for char in token:
        if char.isalpha():
            return char.isupper()
    return False


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def _spelling(text: str, value: str) -> str | None:
    at = text.lower().find(value.lower())
    return text[at:at + len(value)] if at >= 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    samples = [s for s in split.iter_samples() if s.get("audio_text")]

    feature_rows, labels, sample_of = [], [], []
    for sample_index, sample in enumerate(samples):
        gold_text = sample["audio_text"]
        upper_tokens = gold_text.upper().split()
        gold_tokens = gold_text.split()
        if len(upper_tokens) != len(gold_tokens):
            continue
        rule_tokens = restore_capitals(gold_text.upper()).split()
        for index in range(len(upper_tokens)):
            feature_rows.append(featurize(upper_tokens, rule_tokens, index))
            labels.append(int(_is_cap(gold_tokens[index])))
            sample_of.append(sample_index)

    usable = sorted(set(sample_of))
    fold_of = {s: position % args.folds for position, s in enumerate(usable)}

    results = []
    for variant, drop in (("full", ()), ("no_word", ("word",))):
        rows = [{k: v for k, v in row.items() if k not in drop}
                for row in feature_rows]
        vectorizer = DictVectorizer(sparse=True)
        matrix = vectorizer.fit_transform(rows)

        for threshold in (0.0, 0.02, 0.05, 0.1, 0.2, 0.3):
            totals = collections.Counter()
            scored = 0
            for fold in range(args.folds):
                is_test = [fold_of.get(s, 0) == fold for s in sample_of]
                train_idx = [i for i, t in enumerate(is_test) if not t]
                test_idx = [i for i, t in enumerate(is_test) if t]
                if not test_idx:
                    continue
                model = LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs")
                model.fit(matrix[train_idx], [labels[i] for i in train_idx])

                names = vectorizer.get_feature_names_out()
                weights = {name: float(weight)
                           for name, weight in zip(names, model.coef_[0])
                           if abs(weight) > threshold}
                artifact = CaseModel({"intercept": float(model.intercept_[0]),
                                      "weights": weights})

                fold_samples = {s for s in usable if fold_of[s] == fold}
                for sample_index, sample in enumerate(samples):
                    if sample_index not in fold_samples:
                        continue
                    calls = sample.get("label", {}).get("tool_calls") or []
                    if not calls:
                        continue
                    scored += 1
                    upper = sample["audio_text"].upper()
                    rule_text = restore_capitals(upper)
                    model_text = refine_capitals(upper, artifact)

                    gold_hash = replay_hash(split, sample["id"], calls)
                    totals["gold"] += sample_score(
                        calls, calls, gold_hash, gold_hash, fuzzy)
                    for label, text in (("rule", rule_text), ("model", model_text)):
                        variant_calls = []
                        for call in calls:
                            arguments = dict(call.get("arguments") or {})
                            fuzzy_keys = fuzzy.get(call.get("name") or "", set())
                            for key, value in arguments.items():
                                if not isinstance(value, str) or key in fuzzy_keys:
                                    continue
                                if _looks_like_timestamp(value) or not _has_upper(value):
                                    continue
                                spoken = _spelling(text, value)
                                if spoken is not None:
                                    arguments[key] = spoken
                            variant_calls.append(
                                {"name": call.get("name"), "arguments": arguments})
                        pred_hash = replay_hash(split, sample["id"], variant_calls)
                        totals[label] += sample_score(
                            variant_calls, calls, pred_hash, gold_hash, fuzzy)

            n = scored or 1
            payload = json.dumps({"intercept": 0.0, "weights": weights})
            results.append({
                "variant": variant,
                "threshold": threshold,
                "weights_kept": len(weights),
                "size_kib": round(len(payload.encode("utf-8")) / 1024, 1),
                "rule_module": round(100.0 * totals["rule"] / n, 3),
                "context_model": round(100.0 * totals["model"] / n, 3),
                "delta_points": round(100.0 * (totals["model"] - totals["rule"]) / n, 3),
            })

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
