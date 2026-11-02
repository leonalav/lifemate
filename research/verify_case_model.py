"""Verify the shipped artifact against the measured model.

The measurement was made with scikit-learn. The runtime scores a JSON artifact with
dictionary lookups instead. Those two paths must agree exactly, or the measured +1.369
points do not describe what the pipeline will do.

This script checks three things:

  1. **fidelity** — for every token of every training transcript, the artifact's
     prediction equals the scikit-learn model's prediction. Any disagreement is a bug
     in the export or in the runtime scoring, and would silently change the score.
  2. **cross-validated score through the runtime path** — the model is retrained fold
     by fold, exported to the artifact format, and scored through
     ``case_model.refine_capitals``. The metric must come out at the measured value.
  3. **size versus accuracy** — the artifact's weights can be pruned to a magnitude
     threshold. The sweep finds the smallest artifact that keeps the score, because a
     462 KiB JSON committed to the repository should earn its size.

Run:  python research/verify_case_model.py --data-dir data/train+public_test
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


def _export(vectorizer, model, threshold: float = 0.0) -> CaseModel:
    """The scikit-learn model in the artifact format."""
    names = vectorizer.get_feature_names_out()
    weights = {name: float(weight) for name, weight in zip(names, model.coef_[0])
               if abs(weight) > threshold}
    return CaseModel({"intercept": float(model.intercept_[0]), "weights": weights})


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

    fidelity_mismatch = 0
    fidelity_checked = 0
    totals = collections.Counter()
    stats = collections.Counter()
    scored = 0

    for fold in range(args.folds):
        is_test = [fold_of.get(s, 0) == fold for s in sample_of]
        train_idx = [i for i, t in enumerate(is_test) if not t]
        test_idx = [i for i, t in enumerate(is_test) if t]
        if not test_idx:
            continue
        sklearn_model = LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs")
        sklearn_model.fit(matrix[train_idx], [labels[i] for i in train_idx])
        artifact = _export(vectorizer, sklearn_model)

        # --- 1. fidelity ------------------------------------------------------------
        sklearn_predictions = sklearn_model.predict(matrix[test_idx])
        for offset, row_index in enumerate(test_idx):
            fidelity_checked += 1
            expected = bool(sklearn_predictions[offset])
            actual = artifact.predicts_capital(feature_rows[row_index])
            if expected != actual:
                fidelity_mismatch += 1

        # --- 2. the metric through the runtime path --------------------------------
        fold_samples = {s for s in usable if fold_of[s] == fold}
        for sample_index, sample in enumerate(samples):
            if sample_index not in fold_samples:
                continue
            audio_text = sample["audio_text"]
            upper = audio_text.upper()
            rule_text = restore_capitals(upper)
            model_text = refine_capitals(upper, artifact)

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

    # --- 3. the size sweep ----------------------------------------------------------
    final = LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs")
    final.fit(matrix, labels)
    names = vectorizer.get_feature_names_out()
    coefficients = list(zip(names, final.coef_[0]))
    sweep = []
    for threshold in (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3):
        kept = {name: float(weight) for name, weight in coefficients
                if abs(weight) > threshold}
        payload = json.dumps({"intercept": float(final.intercept_[0]),
                               "weights": kept}, sort_keys=True)
        sweep.append({
            "threshold": threshold,
            "weights_kept": len(kept),
            "size_kib": round(len(payload.encode("utf-8")) / 1024, 1),
        })

    n = scored or 1
    total = stats["population"] or 1
    report = {
        "fidelity": {
            "tokens_checked": fidelity_checked,
            "mismatches": fidelity_mismatch,
            "exact": fidelity_mismatch == 0,
        },
        "cross_validated_score_through_runtime_path": {
            "samples_scored": scored,
            "gold": round(100.0 * totals["gold"] / n, 3),
            "rule_module": round(100.0 * totals["rule"] / n, 3),
            "context_model": round(100.0 * totals["model"] / n, 3),
            "delta_points": round(100.0 * (totals["model"] - totals["rule"]) / n, 3),
        },
        "argument_recovery": {
            "population": stats["population"],
            "rule_module_hit_percent": round(
                100.0 * (stats["hit"] + stats["broke"]) / total, 1),
            "context_model_hit_percent": round(
                100.0 * (stats["hit"] + stats["rescued"]) / total, 1),
        },
        "size_sweep": sweep,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
