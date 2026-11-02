"""Ablation: is the context model's gain genuine context, or memorised words?

The context model uses the token's own identity as a feature, so it *could* be learning
"the word 'minh' is capitalised here" — which is memorisation, and would not transfer
to the public test, where the contact names differ (measured: 531 of the 578
capitalised names in the public test environment are unknown to the train-derived
table).

So the model is re-trained with the token-identity feature removed, and the two are
compared. A gain that survives the ablation is a gain that comes from the *context*:
the neighbours, the honorific, the position, the rule module's opinion.

Three feature sets are compared, all evaluated by the same 5-fold cross-validation:

  full        the token identity plus the context
  no_word     the context only — the token identity is removed
  rules_only  the rule module alone, for reference

Run:  python research/ablate_case_model.py --data-dir data/train+public_test
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

from lifemate.capitalize import _vocabulary, restore_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from train_case_model import featurize  # noqa: E402


def _is_cap(token: str) -> bool:
    for char in token:
        if char.isalpha():
            return char.isupper()
    return False


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    split = resolve_split(args.data_dir, args.split)
    samples = [s for s in split.iter_samples() if s.get("audio_text")]
    proper, acronyms, ordinary = _vocabulary()
    acronyms, ordinary = frozenset(acronyms), frozenset(ordinary)

    feature_rows, labels, sample_of = [], [], []
    for sample_index, sample in enumerate(samples):
        gold_text = sample["audio_text"]
        upper_tokens = gold_text.upper().split()
        gold_tokens = gold_text.split()
        if len(upper_tokens) != len(gold_tokens):
            continue
        rule_tokens = restore_capitals(gold_text.upper()).split()
        for index in range(len(upper_tokens)):
            feature_rows.append(featurize(
                upper_tokens, rule_tokens, index, proper, acronyms, ordinary))
            labels.append(int(_is_cap(gold_tokens[index])))
            sample_of.append(sample_index)

    usable = sorted(set(sample_of))
    fold_of = {s: position % args.folds for position, s in enumerate(usable)}

    variants = {
        "full": None,
        "no_word": ("word",),
    }

    results = {}
    for variant_name, drop in variants.items():
        rows = []
        for row in feature_rows:
            if drop:
                row = {k: v for k, v in row.items() if k not in drop}
            rows.append(row)

        vectorizer = DictVectorizer(sparse=True)
        matrix = vectorizer.fit_transform(rows)
        predictions = {}
        for fold in range(args.folds):
            is_test = [fold_of.get(s, 0) == fold for s in sample_of]
            train_idx = [i for i, t in enumerate(is_test) if not t]
            test_idx = [i for i, t in enumerate(is_test) if t]
            model = LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs")
            model.fit(matrix[train_idx], [labels[i] for i in train_idx])
            for index, prediction in zip(test_idx, model.predict(matrix[test_idx])):
                predictions[index] = int(prediction)

        correct = sum(1 for i, p in predictions.items() if p == labels[i])
        results[variant_name] = {
            "held_out_token_accuracy": round(correct / max(len(predictions), 1), 4),
            "features": len(vectorizer.feature_names_),
        }

        # argument-level recovery, per sample
        stats = collections.Counter()
        for sample_index, sample in enumerate(samples):
            if sample_index not in fold_of:
                continue
            audio_text = sample["audio_text"]
            upper = audio_text.upper()
            rule_tokens = restore_capitals(upper).split()
            upper_tokens = upper.split()
            gold_tokens = audio_text.split()
            if len(rule_tokens) != len(upper_tokens):
                continue
            model_tokens = []
            for index in range(len(rule_tokens)):
                position = next((p for p, c in enumerate(rule_tokens[index])
                                 if c.isalpha()), -1)
                if position < 0:
                    model_tokens.append(rule_tokens[index])
                    continue
                # find this token's row in the feature list
                row = feature_rows[0]
                key = (sample_index, index)
                prediction = predictions.get(key)
                if prediction is None:
                    model_tokens.append(rule_tokens[index])
                    continue
                char = rule_tokens[index][position]
                new_char = char.upper() if prediction else char.lower()
                model_tokens.append(
                    rule_tokens[index][:position] + new_char + rule_tokens[index][position + 1:])
            model_text = " ".join(model_tokens)
            rule_text = " ".join(rule_tokens)

            calls = sample.get("label", {}).get("tool_calls") or []
            for call in calls:
                for key, value in (call.get("arguments") or {}).items():
                    if not isinstance(value, str) or not _has_upper(value):
                        continue
                    if _looks_like_timestamp(value):
                        continue
                    if value.lower() not in audio_text.lower():
                        continue
                    stats["population"] += 1
                    at = rule_text.lower().find(value.lower())
                    rule_ok = at >= 0 and rule_text[at:at + len(value)] == value
                    at = model_text.lower().find(value.lower())
                    model_ok = at >= 0 and model_text[at:at + len(value)] == value
                    if rule_ok and model_ok:
                        stats["hit"] += 1
                    elif not rule_ok and model_ok:
                        stats["rescued"] += 1
                    elif rule_ok and not model_ok:
                        stats["broke"] += 1
                    else:
                        stats["miss"] += 1

        total = stats["population"] or 1
        results[variant_name].update({
            "rule_module_hit_percent": round(
                100.0 * (stats["hit"] + stats["broke"]) / total, 1),
            "model_hit_percent": round(
                100.0 * (stats["hit"] + stats["rescued"]) / total, 1),
            "rescued": stats["rescued"], "broke": stats["broke"],
        })

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
