"""Which tokens should the model be allowed to touch?

The model refines the rule module's decision about a token's first character. For most
tokens that is the right division of labour: the rules know the shape of a token, the
model knows the context. But two kinds of token are decided *structurally* by the rules,
not by context, and the model has no way to reproduce that reasoning:

  honorifics    the rules capitalise "Mẹ" only when a name follows it ("Mẹ Lan"), and
                leave it alone when it stands alone ("gọi mẹ nhé"). The model sees the
                word "mẹ" and the neighbouring words, and cannot tell those apart
                reliably — so it lowercases the honorifics the rules got right.
                Measured: 17 of the 17 arguments the model breaks are honorifics
                ("Mẹ", "Bố", "Sếp Dũng", "Cô Hạnh", "Chú Năm") or codes ("VTV6").

  codes         "VTV6" is a channel code. Its shape is not a context question.

This measures the deferral policies:

  refine_all            the model decides every token (the current behaviour)
  skip_honorific        the rules own the honorifics
  skip_honorific_code   the rules own the honorifics and the codes
  skip_code             the rules own the codes only

Run:  python research/analyze_case_deferral.py --data-dir data/train+public_test
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

from lifemate.capitalize import (  # noqa: E402
    HONORIFICS,
    is_code,
    restore_capitals,
)
from lifemate.case_model import (  # noqa: E402
    CaseModel,
    _first_alpha,
    featurize,
)
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


def _refine(upper_text: str, artifact: CaseModel, policy: str) -> str:
    """The model's transcript, with the deferral policy applied."""
    upper_tokens = upper_text.split()
    rule_tokens = restore_capitals(upper_text).split()
    if len(rule_tokens) != len(upper_tokens):
        return restore_capitals(upper_text)

    out = []
    for index, rule_token in enumerate(rule_tokens):
        position = _first_alpha(rule_token)
        if position < 0:
            out.append(rule_token)
            continue
        bare = "".join(c for c in rule_token if c.isalnum())
        lowered = bare.lower()
        defer = False
        if policy in ("skip_honorific", "skip_honorific_code") \
                and lowered in HONORIFICS:
            defer = True
        if policy in ("skip_code", "skip_honorific_code") and is_code(bare):
            defer = True
        if defer:
            out.append(rule_token)
            continue
        features = featurize(upper_tokens, rule_tokens, index)
        char = rule_token[position]
        new_char = char.upper() if artifact.predicts_capital(features) else char.lower()
        out.append(rule_token[:position] + new_char + rule_token[position + 1:])
    return " ".join(out)


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

    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(feature_rows)
    usable = sorted(set(sample_of))
    fold_of = {s: position % args.folds for position, s in enumerate(usable)}

    policies = ("refine_all", "skip_honorific", "skip_code", "skip_honorific_code")
    results = {}

    for policy in policies:
        totals = collections.Counter()
        stats = collections.Counter()
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
            artifact = CaseModel({
                "intercept": float(model.intercept_[0]),
                "weights": {n: float(w) for n, w in zip(names, model.coef_[0])},
            })

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
                model_text = _refine(upper, artifact, policy)

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
                        if value.lower() not in sample["audio_text"].lower():
                            continue
                        stats["population"] += 1
                        rule_ok = _spelling(rule_text, value) == value
                        model_ok = _spelling(model_text, value) == value
                        if rule_ok and model_ok:
                            stats["hit"] += 1
                        elif model_ok:
                            stats["rescued"] += 1
                        elif rule_ok:
                            stats["broke"] += 1
                        else:
                            stats["miss"] += 1

                gold_hash = replay_hash(split, sample["id"], calls)
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
                            spoken = _spelling(text, value)
                            if spoken is not None:
                                arguments[key] = spoken
                        variant.append({"name": call.get("name"), "arguments": arguments})
                    pred_hash = replay_hash(split, sample["id"], variant)
                    totals[label] += sample_score(
                        variant, calls, pred_hash, gold_hash, fuzzy)

        n = scored or 1
        total = stats["population"] or 1
        results[policy] = {
            "rule_module": round(100.0 * totals["rule"] / n, 3),
            "context_model": round(100.0 * totals["model"] / n, 3),
            "delta_points": round(100.0 * (totals["model"] - totals["rule"]) / n, 3),
            "rule_hit_percent": round(
                100.0 * (stats["hit"] + stats["broke"]) / total, 1),
            "model_hit_percent": round(
                100.0 * (stats["hit"] + stats["rescued"]) / total, 1),
            "rescued": stats["rescued"], "broke": stats["broke"],
        }

    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
