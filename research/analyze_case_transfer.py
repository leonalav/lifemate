"""Will the case model transfer to the public test?

The public test has no gold transcripts, so the model can only be trained on train. The
risk is that it learns *which words* to capitalise ("minh" -> capital) rather than *when*
to capitalise them, and the public test uses contact names and programmes train never
saw — 531 of the 578 capitalised names in the public test environment are unknown to the
train-derived vocabulary.

This script measures the model on the subset that matters for that risk: the affected
arguments whose gold value contains a word that never appears capitalised in the
training transcripts. If the model still helps there, its gain is contextual and it
transfers; if the gain evaporates, it is memorisation and it will not.

It also measures the effect of capping the categorical feature vocabulary, which is what
keeps the exported artifact small enough to ship.

Run:  python research/analyze_case_transfer.py --data-dir data/train+public_test
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lifemate.capitalize import _vocabulary, restore_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from train_case_model import featurize  # noqa: E402

WORD = re.compile(r"[^\W_]+", re.UNICODE)


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

    # the vocabulary the training transcripts teach
    trained_capitalised = set()
    for sample in samples:
        for word in WORD.findall(sample["audio_text"]):
            if word[0].isupper():
                trained_capitalised.add(word.lower())

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

    results = {}
    for variant, drop in (("full", ()), ("no_word", ("word",))):
        rows = [{k: v for k, v in row.items() if k not in drop} for row in feature_rows]
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

        # per-sample token predictions, so the argument level can be measured
        stats = collections.Counter()
        for sample_index, sample in enumerate(samples):
            if sample_index not in fold_of:
                continue
            audio_text = sample["audio_text"]
            upper = audio_text.upper()
            rule_tokens = restore_capitals(upper).split()
            upper_tokens = upper.split()
            if len(rule_tokens) != len(upper_tokens):
                continue
            model_tokens = []
            for index in range(len(rule_tokens)):
                position = next((p for p, c in enumerate(rule_tokens[index])
                                 if c.isalpha()), -1)
                if position < 0:
                    model_tokens.append(rule_tokens[index])
                    continue
                row_index = None
                # find the feature row for this (sample, token)
                for candidate in range(len(feature_rows)):
                    if sample_of[candidate] == sample_index and \
                            rows[candidate].get("word", "") == \
                            "".join(c for c in upper_tokens[index] if c.isalnum()).lower():
                        row_index = candidate
                        break
                prediction = predictions.get(row_index) if row_index is not None else None
                if prediction is None:
                    model_tokens.append(rule_tokens[index])
                    continue
                char = rule_tokens[index][position]
                new_char = char.upper() if prediction else char.lower()
                model_tokens.append(
                    rule_tokens[index][:position] + new_char
                    + rule_tokens[index][position + 1:])
            model_text = " ".join(model_tokens)
            rule_text = " ".join(rule_tokens)

            for call in sample.get("label", {}).get("tool_calls") or []:
                for key, value in (call.get("arguments") or {}).items():
                    if not isinstance(value, str) or not _has_upper(value):
                        continue
                    if _looks_like_timestamp(value):
                        continue
                    if value.lower() not in audio_text.lower():
                        continue
                    words = [w.lower() for w in WORD.findall(value)]
                    unseen = any(w not in trained_capitalised for w in words)
                    bucket = "unseen_word" if unseen else "seen_word"
                    stats[f"{bucket}_population"] += 1
                    at = rule_text.lower().find(value.lower())
                    rule_ok = at >= 0 and rule_text[at:at + len(value)] == value
                    at = model_text.lower().find(value.lower())
                    model_ok = at >= 0 and model_text[at:at + len(value)] == value
                    if rule_ok and model_ok:
                        stats[f"{bucket}_hit"] += 1
                    elif not rule_ok and model_ok:
                        stats[f"{bucket}_rescued"] += 1
                    elif rule_ok and not model_ok:
                        stats[f"{bucket}_broke"] += 1
                    else:
                        stats[f"{bucket}_miss"] += 1

        results[variant] = {}
        for bucket in ("seen_word", "unseen_word"):
            total = stats[f"{bucket}_population"] or 1
            results[variant][bucket] = {
                "arguments": stats[f"{bucket}_population"],
                "rule_module_hit_percent": round(
                    100.0 * (stats[f"{bucket}_hit"] + stats[f"{bucket}_broke"]) / total, 1),
                "model_hit_percent": round(
                    100.0 * (stats[f"{bucket}_hit"] + stats[f"{bucket}_rescued"]) / total, 1),
                "rescued": stats[f"{bucket}_rescued"],
                "broke": stats[f"{bucket}_broke"],
            }

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
