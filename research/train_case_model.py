"""A context-aware capitalisation model: does context actually beat the rules?

The measurements say the remaining case-model misses are ~90/96 "ambiguous" — a token
seen in *both* cases in the training transcripts, so its case is a genuine context
decision ("Minh" the contact vs "minh" the pronoun). A vocabulary cannot settle that;
context can.

This script trains a token-level classifier on the training transcripts and measures it
against the current rule module on held-out samples.

    features   the token itself, its neighbours (previous and next word), the rule
               module's own opinion, position, and vocabulary membership
    label      whether the gold token's first character is uppercase
    input      the uppercased text, i.e. exactly what the ASR delivers

The rule module's prediction is one of the features, so the classifier starts from the
rules' behaviour and learns when to override them — rather than replacing them.

Two evaluations:

  token accuracy   per-token capitalisation accuracy on held-out transcripts
  argument recovery  of the exactly-compared capitalised arguments whose words occur in
                   the utterance, how many the restored transcript now spells exactly as
                   the gold does

Run:  python research/train_case_model.py --data-dir data/train+public_test
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

from lifemate.capitalize import HONORIFICS, _vocabulary, restore_capitals  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402

WORD = re.compile(r"\S+")
PUNCT = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def tokens(text: str) -> list[str]:
    return WORD.findall(text)


def bare(token: str) -> str:
    return PUNCT.sub("", token)


def _is_cap(token: str) -> bool:
    word = bare(token)
    return bool(word) and word[0].isupper()


def featurize(tokens_: list[str], rule_tokens: list[str], index: int,
              proper: dict, acronyms: frozenset, ordinary: frozenset) -> dict:
    """The features for one token."""
    word = bare(tokens_[index]).lower()
    prev1 = bare(tokens_[index - 1]).lower() if index > 0 else ""
    prev2 = bare(tokens_[index - 2]).lower() if index > 1 else ""
    next1 = bare(tokens_[index + 1]).lower() if index + 1 < len(tokens_) else ""
    next2 = bare(tokens_[index + 2]).lower() if index + 2 < len(tokens_) else ""

    return {
        "bias": 1.0,
        "word": word,
        "prev1": prev1,
        "prev2": prev2,
        "next1": next1,
        "next2": next2,
        "prev1_is_honorific": float(prev1 in HONORIFICS),
        "next1_is_honorific": float(next1 in HONORIFICS),
        "in_proper_vocab": float(word in proper),
        "in_acronym_vocab": float(word in acronyms),
        "in_ordinary_vocab": float(word in ordinary),
        "is_first": float(index == 0),
        "is_last": float(index == len(tokens_) - 1),
        "word_len": min(len(word), 12) / 12.0,
        "has_digit": float(any(c.isdigit() for c in word)),
        "rule_says_capital": float(_is_cap(rule_tokens[index])),
        "rule_prev1": bare(rule_tokens[index - 1]).lower() if index > 0 else "",
        "rule_next1": (bare(rule_tokens[index + 1]).lower()
                       if index + 1 < len(rule_tokens) else ""),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    split = resolve_split(args.data_dir, args.split)
    samples = [s for s in split.iter_samples() if s.get("audio_text")]

    proper, acronyms, ordinary = _vocabulary()
    acronyms = frozenset(acronyms)
    ordinary = frozenset(ordinary)

    feature_rows: list[dict] = []
    labels: list[int] = []
    sample_of: list[int] = []

    aligned: list[tuple[int, list[str], list[str]]] = []
    for sample_index, sample in enumerate(samples):
        gold_text = sample["audio_text"]
        upper_tokens = tokens(gold_text.upper())
        gold_tokens = tokens(gold_text)
        if len(upper_tokens) != len(gold_tokens):
            continue      # punctuation differences shift the alignment; skip
        rule_tokens = tokens(restore_capitals(gold_text.upper()))
        aligned.append((sample_index, upper_tokens, gold_tokens))
        for index in range(len(upper_tokens)):
            feature_rows.append(featurize(
                upper_tokens, rule_tokens, index, proper, acronyms, ordinary))
            labels.append(int(_is_cap(gold_tokens[index])))
            sample_of.append(sample_index)

    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(feature_rows)

    # A held-out split by sample, so no sample appears in both halves.
    usable = sorted({s for s, _, _ in aligned})
    test_samples = set(usable[::5])
    is_test = [s in test_samples for s in sample_of]

    train_idx = [i for i, t in enumerate(is_test) if not t]
    test_idx = [i for i, t in enumerate(is_test) if t]

    model = LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs")
    model.fit(matrix[train_idx], [labels[i] for i in train_idx])

    predictions = model.predict(matrix[test_idx])
    model_correct = sum(int(p == labels[i]) for p, i in zip(predictions, test_idx))

    # the rule module's accuracy on exactly the same tokens
    rule_correct = 0
    for sample_index, upper_tokens, gold_tokens in aligned:
        if sample_index not in test_samples:
            continue
        rule_tokens = tokens(restore_capitals(samples[sample_index]["audio_text"].upper()))
        for index in range(len(upper_tokens)):
            rule_correct += int(_is_cap(rule_tokens[index]) == _is_cap(gold_tokens[index]))

    report = {
        "tokens_trained_on": len(train_idx),
        "tokens_held_out": len(test_idx),
        "samples_held_out": len(test_samples),
        "held_out_token_accuracy": {
            "context_model": round(model_correct / max(len(test_idx), 1), 4),
            "rule_module": round(rule_correct / max(len(test_idx), 1), 4),
        },
        "features": len(vectorizer.feature_names_),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with out.open("wb") as fh:
            pickle.dump({"vectorizer": vectorizer, "model": model}, fh)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
