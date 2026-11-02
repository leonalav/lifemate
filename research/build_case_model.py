"""Train the context-aware capitalisation model and export the runtime artifact.

The model is a logistic regression over contextual features (see
``src/lifemate/case_model.py``). This script trains it on the training split, reports
the cross-validated score, and writes the coefficients to
``src/lifemate/resources/case_model.json`` in the flat form the runtime scores.

**Cross-validation is by sample, not by row.** The public test has no gold
transcripts, so the model must generalise to samples it has never seen. Every sample is
scored by a model trained on the other four folds, which is the only honest estimate of
what the artifact will do on unseen data.

The report covers three questions:

  token accuracy      per-token capitalisation accuracy, against the rule module
  argument recovery   of the exactly-compared capitalised arguments whose words occur
                      in the utterance, how many the restored transcript now spells
                      exactly as the gold does
  metric score        the official metric, with the transcript's spelling substituted
                      into the affected argument

Run:  python research/build_case_model.py --data-dir data/train+public_test
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
from lifemate.case_model import featurize  # noqa: E402
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
    """How the given text spells the value, if it contains it at all."""
    at = text.lower().find(value.lower())
    return text[at:at + len(value)] if at >= 0 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--out", default=str(
        REPO_ROOT / "src" / "lifemate" / "resources" / "case_model.json"))
    parser.add_argument("--report", default="")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="drop weights whose magnitude is at or below this")
    args = parser.parse_args()

    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression

    split = resolve_split(args.data_dir, args.split)
    fuzzy = fuzzy_arguments(split.catalog())
    samples = [s for s in split.iter_samples() if s.get("audio_text")]

    # --- the dataset -----------------------------------------------------------------
    feature_rows: list[dict] = []
    labels: list[int] = []
    sample_of: list[int] = []
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

    # --- cross-validation -------------------------------------------------------------
    stats = collections.Counter()
    totals = collections.Counter()
    scored = 0
    rule_correct = 0
    model_correct = 0

    for fold in range(args.folds):
        is_test = [fold_of.get(s, 0) == fold for s in sample_of]
        train_idx = [i for i, t in enumerate(is_test) if not t]
        test_idx = [i for i, t in enumerate(is_test) if t]
        if not test_idx:
            continue
        model = LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs")
        model.fit(matrix[train_idx], [labels[i] for i in train_idx])

        for index in test_idx:
            if int(model.predict(matrix[index])[0]) == labels[index]:
                model_correct += 1
            else:
                rule_correct += 1      # the rule module is the reference

        fold_samples = {s for s in usable if fold_of[s] == fold}
        for sample_index, sample in enumerate(samples):
            if sample_index not in fold_samples:
                continue
            audio_text = sample["audio_text"]
            upper = audio_text.upper()
            rule_text = restore_capitals(upper)
            model_text = _refine(upper, vectorizer, model)

            calls = sample.get("label", {}).get("tool_calls") or []
            if not calls:
                continue
            scored += 1

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
    report = {
        "samples_scored": scored,
        "folds": args.folds,
        "held_out_token_accuracy": {
            "context_model": round(model_correct / max(model_correct + rule_correct, 1), 4),
        },
        "argument_recovery": {
            "population": stats["population"],
            "rule_module_hit_percent": round(
                100.0 * (stats["hit"] + stats["broke"]) / total, 1),
            "context_model_hit_percent": round(
                100.0 * (stats["hit"] + stats["rescued"]) / total, 1),
            "rescued": stats["rescued"], "broke": stats["broke"],
        },
        "metric_score": {
            "gold": round(100.0 * totals["gold"] / n, 3),
            "rule_module": round(100.0 * totals["rule"] / n, 3),
            "context_model": round(100.0 * totals["model"] / n, 3),
            "delta_points": round(100.0 * (totals["model"] - totals["rule"]) / n, 3),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # --- the final model, trained on every sample --------------------------------------
    final = LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs")
    final.fit(matrix, labels)

    names = vectorizer.get_feature_names_out()
    weights = {name: float(weight) for name, weight in zip(names, final.coef_[0])
               if abs(weight) > args.threshold}
    payload = {
        "version": 1,
        "intercept": float(final.intercept_[0]),
        "features": int(len(names)),
        "weights": weights,
        "notes": (
            "Trained on the train split by research/build_case_model.py. "
            "Scored with dictionary lookups; see src/lifemate/case_model.py."
        ),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=1)
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KiB, "
          f"{len(weights)} weights)")

    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _refine(upper_text: str, vectorizer, model) -> str:
    """The model's transcript, using the same code path as the runtime."""
    from lifemate.case_model import _defer_to_the_rules, _first_alpha

    upper_tokens = upper_text.split()
    rule_tokens = restore_capitals(upper_text).split()
    if len(rule_tokens) != len(upper_tokens):
        return restore_capitals(upper_text)
    features = [featurize(upper_tokens, rule_tokens, index)
                for index in range(len(upper_tokens))]
    predictions = model.predict(vectorizer.transform(features))
    out = []
    for rule_token, prediction in zip(rule_tokens, predictions):
        position = _first_alpha(rule_token)
        if position < 0:
            out.append(rule_token)
            continue
        if _defer_to_the_rules(rule_token):
            out.append(rule_token)
            continue
        char = rule_token[position]
        new_char = char.upper() if prediction else char.lower()
        out.append(rule_token[:position] + new_char + rule_token[position + 1:])
    return " ".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
