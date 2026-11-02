"""Does the context model's better token accuracy translate into recovered arguments?

Token accuracy is not the objective. The objective is the 250 exactly-compared
capitalised arguments whose words occur in the utterance: for each, does the restored
transcript now spell the value exactly as the gold does?

This script measures, on held-out samples:

  rule_module      the argument is spelled correctly in the restored transcript
  context_model    the same, after the context model rewrites the transcript
  delta            the arguments the context model rescues

and then prices the delta in metric points, by replaying the gold trajectory with the
transcript's spelling substituted for each affected argument and re-scoring.

The model is trained on the training split only and evaluated on samples it never saw.

Run:  python research/eval_case_model.py --data-dir data/train+public_test
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

from lifemate.capitalize import (  # noqa: E402
    HONORIFICS,
    _vocabulary,
    restore_capitals,
)
from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import (  # noqa: E402
    fuzzy_arguments,
    replay_hash,
    sample_score,
)

WORD = re.compile(r"\S+")
PUNCT = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)


def _has_upper(text: str) -> bool:
    return any(ch.isupper() for ch in text)


def _looks_like_timestamp(value: str) -> bool:
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def _bare(token: str) -> str:
    return PUNCT.sub("", token)


def _is_cap(token: str) -> bool:
    """True when the token's first alphabetic character is uppercase."""
    for char in token:
        if char.isalpha():
            return char.isupper()
    return False


def _first_alpha(token: str) -> int:
    for position, char in enumerate(token):
        if char.isalpha():
            return position
    return -1


def _apply_model(model_bundle, upper_text: str) -> str:
    """Rewrite the casing of a restored transcript with the context model.

    The model decides one thing per token: whether the token's first alphabetic
    character is uppercase. Everything else about the token — internal capitals, digits,
    punctuation — is taken from the rule module, which already knows that "VOV3" and
    "S+ Sport" keep their shape. The model only refines the first letter.
    """
    vectorizer = model_bundle["vectorizer"]
    model = model_bundle["model"]
    proper, acronyms, ordinary = _vocabulary()

    upper_tokens = upper_text.split()
    rule_tokens = restore_capitals(upper_text).split()
    if len(rule_tokens) != len(upper_tokens):
        return restore_capitals(upper_text)

    features = []
    for index in range(len(upper_tokens)):
        word = _bare(upper_tokens[index]).lower()
        prev1 = _bare(upper_tokens[index - 1]).lower() if index > 0 else ""
        prev2 = _bare(upper_tokens[index - 2]).lower() if index > 1 else ""
        next1 = _bare(upper_tokens[index + 1]).lower() if index + 1 < len(upper_tokens) else ""
        next2 = _bare(upper_tokens[index + 2]).lower() if index + 2 < len(upper_tokens) else ""
        features.append({
            "bias": 1.0, "word": word, "prev1": prev1, "prev2": prev2,
            "next1": next1, "next2": next2,
            "prev1_is_honorific": float(prev1 in HONORIFICS),
            "next1_is_honorific": float(next1 in HONORIFICS),
            "in_proper_vocab": float(word in proper),
            "in_acronym_vocab": float(word in acronyms),
            "in_ordinary_vocab": float(word in ordinary),
            "is_first": float(index == 0),
            "is_last": float(index == len(upper_tokens) - 1),
            "word_len": min(len(word), 12) / 12.0,
            "has_digit": float(any(c.isdigit() for c in word)),
            "rule_says_capital": float(_is_cap(rule_tokens[index])),
            "rule_prev1": _bare(rule_tokens[index - 1]).lower() if index > 0 else "",
            "rule_next1": (_bare(rule_tokens[index + 1]).lower()
                           if index + 1 < len(rule_tokens) else ""),
        })

    predictions = model.predict(vectorizer.transform(features))

    out = []
    for rule_token, prediction in zip(rule_tokens, predictions):
        position = _first_alpha(rule_token)
        if position < 0:
            out.append(rule_token)
            continue
        char = rule_token[position]
        if prediction:
            new_char = char.upper()
        else:
            new_char = char.lower()
        out.append(rule_token[:position] + new_char + rule_token[position + 1:])
    return " ".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="research/artifacts/case_model.pkl")
    args = parser.parse_args()

    import pickle
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    samples = [s for s in split.iter_samples() if s.get("audio_text")]

    # --- train on 4/5 of the samples -------------------------------------------------
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_case_model import featurize  # noqa
    proper, acronyms, ordinary = _vocabulary()
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
                upper_tokens, rule_tokens, index, proper,
                frozenset(acronyms), frozenset(ordinary)))
            labels.append(int(_is_cap(gold_tokens[index])))
            sample_of.append(sample_index)

    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(feature_rows)
    usable = sorted(set(sample_of))
    test_samples = set(usable[::5])
    is_test = [s in test_samples for s in sample_of]
    train_idx = [i for i, t in enumerate(is_test) if not t]

    model = LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs")
    model.fit(matrix[train_idx], [labels[i] for i in train_idx])
    bundle = {"vectorizer": vectorizer, "model": model}

    out_path = Path(args.model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as fh:
        pickle.dump(bundle, fh)

    # --- evaluate on the held-out samples --------------------------------------------
    stats = collections.Counter()
    totals = collections.Counter()
    rescued: list[dict] = []

    for sample in samples:
        sample_id = sample["id"]
        if sample_id not in {samples[i]["id"] for i in test_samples}:
            continue
        audio_text = sample["audio_text"]
        upper = audio_text.upper()
        rule_text = restore_capitals(upper)
        model_text = _apply_model(bundle, upper)

        calls = sample.get("label", {}).get("tool_calls") or []
        if not calls:
            continue

        def spelling(text: str, value: str) -> str | None:
            at = text.lower().find(value.lower())
            return text[at:at + len(value)] if at >= 0 else None

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
                if value.lower() not in audio_text.lower():
                    continue      # the catalogue population: not a case problem
                stats["transcript_population"] += 1
                rule_spelling = spelling(rule_text, value)
                model_spelling = spelling(model_text, value)
                if rule_spelling == value:
                    stats["rule_hit"] += 1
                else:
                    stats["rule_miss"] += 1
                if model_spelling == value:
                    stats["model_hit"] += 1
                elif rule_spelling != value:
                    stats["model_rescued"] += 1
                    if len(rescued) < 25:
                        rescued.append({
                            "id": sample_id, "slot": f"{tool}.{key}", "gold": value,
                            "rule": rule_spelling, "model": model_spelling,
                        })
                else:
                    stats["model_broke"] += 1

        # price the difference in metric points
        gold_hash = replay_hash(split, sample_id, calls)
        totals["gold"] += sample_score(calls, calls, gold_hash, gold_hash, fuzzy)
        for label, text in (("rule", rule_text), ("model", model_text)):
            variant = []
            for call in calls:
                arguments = dict(call.get("arguments") or {})
                fuzzy_keys = fuzzy.get(call.get("name") or "", set())
                for key, value in arguments.items():
                    if not isinstance(value, str) or key in fuzzy_keys:
                        continue
                    if _looks_like_timestamp(value) or not _has_upper(value):
                        continue
                    spoken = spelling(text, value)
                    if spoken is not None:
                        arguments[key] = spoken
                variant.append({"name": call.get("name"), "arguments": arguments})
            pred_hash = replay_hash(split, sample_id, variant)
            totals[label] += sample_score(
                variant, calls, pred_hash, gold_hash, fuzzy)

    n = sum(1 for s in samples if s["id"] in {samples[i]["id"] for i in test_samples}
            and s.get("label", {}).get("tool_calls"))
    n = n or 1
    total = stats["transcript_population"] or 1
    report = {
        "held_out_samples_scored": n,
        "transcript_population_arguments": stats["transcript_population"],
        "argument_recovery": {
            "rule_module_hit": stats["rule_hit"],
            "rule_module_miss": stats["rule_miss"],
            "rule_module_hit_percent": round(100.0 * stats["rule_hit"] / total, 1),
            "context_model_hit": stats["model_hit"],
            "context_model_hit_percent": round(100.0 * stats["model_hit"] / total, 1),
            "rescued_by_model": stats["model_rescued"],
            "broken_by_model": stats["model_broke"],
        },
        "metric_score_on_held_out": {
            "gold": round(100.0 * totals["gold"] / n, 3),
            "rule_module": round(100.0 * totals["rule"] / n, 3),
            "context_model": round(100.0 * totals["model"] / n, 3),
            "delta_points": round(100.0 * (totals["model"] - totals["rule"]) / n, 3),
        },
        "rescued_examples": rescued,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
