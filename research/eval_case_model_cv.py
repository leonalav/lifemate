"""Cross-validated evaluation of the context model against the rule module.

The single held-out split gave a noisy estimate on 278 samples. This runs 5-fold
cross-validation over the whole train split, so every sample is scored by a model that
never saw it, and the resulting score is directly comparable to the whole-split
ceiling numbers (perfect capitalisation = +9.258 points over lowercasing).

The accounting is exact:

  hit       the restored transcript spells the gold value exactly
  miss      it does not, and the model neither fixed nor broke anything
  rescued   the rules got it wrong and the model got it right
  broke     the rules got it right and the model got it wrong

The metric variant substitutes the transcript's spelling into the affected argument and
re-scores with the official metric, so the delta is in metric points.

Run:  python research/eval_case_model_cv.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def _first_alpha(token: str) -> int:
    for position, char in enumerate(token):
        if char.isalpha():
            return position
    return -1


def _apply_model(bundle, upper_text: str) -> str:
    """The model decides each token's first letter; the rule module keeps the rest."""
    upper_tokens = upper_text.split()
    rule_tokens = restore_capitals(upper_text).split()
    if len(rule_tokens) != len(upper_tokens):
        return restore_capitals(upper_text)

    features = []
    for index in range(len(upper_tokens)):
        word = upper_tokens[index]
        bare = "".join(c for c in word if c.isalnum())
        lower = bare.lower()
        prev1 = _lower_bare(upper_tokens[index - 1]) if index > 0 else ""
        prev2 = _lower_bare(upper_tokens[index - 2]) if index > 1 else ""
        next1 = _lower_bare(upper_tokens[index + 1]) if index + 1 < len(upper_tokens) else ""
        next2 = _lower_bare(upper_tokens[index + 2]) if index + 2 < len(upper_tokens) else ""
        features.append({
            "bias": 1.0, "word": lower, "prev1": prev1, "prev2": prev2,
            "next1": next1, "next2": next2,
            "prev1_is_honorific": float(prev1 in HONORIFICS),
            "next1_is_honorific": float(next1 in HONORIFICS),
            "in_proper_vocab": float(lower in bundle["proper"]),
            "in_acronym_vocab": float(lower in bundle["acronyms"]),
            "in_ordinary_vocab": float(lower in bundle["ordinary"]),
            "is_first": float(index == 0),
            "is_last": float(index == len(upper_tokens) - 1),
            "word_len": min(len(lower), 12) / 12.0,
            "has_digit": float(any(c.isdigit() for c in lower)),
            "rule_says_capital": float(_is_cap(rule_tokens[index])),
            "rule_prev1": _lower_bare(rule_tokens[index - 1]) if index > 0 else "",
            "rule_next1": (_lower_bare(rule_tokens[index + 1])
                           if index + 1 < len(rule_tokens) else ""),
        })

    for row in features:
        for dropped in bundle.get("drop_features") or ():
            row.pop(dropped, None)

    predictions = bundle["model"].predict(bundle["vectorizer"].transform(features))
    out = []
    for rule_token, prediction in zip(rule_tokens, predictions):
        position = _first_alpha(rule_token)
        if position < 0:
            out.append(rule_token)
            continue
        char = rule_token[position]
        new_char = char.upper() if prediction else char.lower()
        out.append(rule_token[:position] + new_char + rule_token[position + 1:])
    return " ".join(out)


def _lower_bare(token: str) -> str:
    return "".join(c for c in token if c.isalnum()).lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model-out", default="research/artifacts/case_model.pkl")
    parser.add_argument("--drop-features", nargs="*", default=[],
                        help="feature names to remove, e.g. --drop-features word")
    parser.add_argument("--label", default="model")
    args = parser.parse_args()

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    samples = [s for s in split.iter_samples() if s.get("audio_text")]
    proper, acronyms, ordinary = _vocabulary()

    # --- feature matrix over the whole split ---------------------------------------
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

    if args.drop_features:
        dropped = set(args.drop_features)
        feature_rows = [{k: v for k, v in row.items() if k not in dropped}
                        for row in feature_rows]

    vectorizer = DictVectorizer(sparse=True)
    matrix = vectorizer.fit_transform(feature_rows)
    usable = sorted(set(sample_of))
    fold_of = {sample_index: position % args.folds
               for position, sample_index in enumerate(usable)}

    stats = collections.Counter()
    totals = collections.Counter()
    scored_samples = 0
    rescued: list[dict] = []
    broken: list[dict] = []
    final_bundle = None

    for fold in range(args.folds):
        is_test = [fold_of.get(s, 0) == fold for s in sample_of]
        train_idx = [i for i, t in enumerate(is_test) if not t]
        test_idx = [i for i, t in enumerate(is_test) if t]
        if not test_idx:
            continue
        model = LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs")
        model.fit(matrix[train_idx], [labels[i] for i in train_idx])
        bundle = {"vectorizer": vectorizer, "model": model,
                  "proper": proper, "acronyms": frozenset(acronyms),
                  "ordinary": frozenset(ordinary),
                  "drop_features": list(args.drop_features)}
        final_bundle = bundle

        fold_samples = {usable[i] for i in range(len(usable))
                        if fold_of[usable[i]] == fold}
        for sample_index, sample in enumerate(samples):
            if sample_index not in fold_samples:
                continue
            audio_text = sample["audio_text"]
            upper = audio_text.upper()
            rule_text = restore_capitals(upper)
            model_text = _apply_model(bundle, upper)

            calls = sample.get("label", {}).get("tool_calls") or []
            if not calls:
                continue
            scored_samples += 1

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
                    stats["population"] += 1
                    rule_ok = spelling(rule_text, value) == value
                    model_ok = spelling(model_text, value) == value
                    if rule_ok and model_ok:
                        stats["hit"] += 1
                    elif not rule_ok and model_ok:
                        stats["rescued"] += 1
                        if len(rescued) < 25:
                            rescued.append({
                                "id": sample["id"], "slot": f"{tool}.{key}",
                                "gold": value, "rule": spelling(rule_text, value),
                            })
                    elif rule_ok and not model_ok:
                        stats["broke"] += 1
                        if len(broken) < 25:
                            broken.append({
                                "id": sample["id"], "slot": f"{tool}.{key}",
                                "gold": value, "model": spelling(model_text, value),
                            })
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
                        spoken = spelling(text, value)
                        if spoken is not None:
                            arguments[key] = spoken
                    variant.append({"name": call.get("name"), "arguments": arguments})
                pred_hash = replay_hash(split, sample["id"], variant)
                totals[label] += sample_score(
                    variant, calls, pred_hash, gold_hash, fuzzy)

    n = scored_samples or 1
    total = stats["population"] or 1
    report = {
        "label": args.label,
        "dropped_features": args.drop_features,
        "samples_scored": scored_samples,
        "folds": args.folds,
        "transcript_population_arguments": stats["population"],
        "argument_recovery": {
            "hit_both": stats["hit"],
            "rescued_by_model": stats["rescued"],
            "broken_by_model": stats["broke"],
            "miss_both": stats["miss"],
            "rule_module_hit_percent": round(
                100.0 * (stats["hit"] + stats["broke"]) / total, 1),
            "context_model_hit_percent": round(
                100.0 * (stats["hit"] + stats["rescued"]) / total, 1),
        },
        "metric_score": {
            "gold": round(100.0 * totals["gold"] / n, 3),
            "rule_module": round(100.0 * totals["rule"] / n, 3),
            "context_model": round(100.0 * totals["model"] / n, 3),
            "delta_points": round(100.0 * (totals["model"] - totals["rule"]) / n, 3),
        },
        "rescued_examples": rescued,
        "broken_examples": broken,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.model_out and final_bundle and not args.drop_features:
        out = Path(args.model_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            pickle.dump(final_bundle, fh)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
