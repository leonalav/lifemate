"""The official metric, reimplemented and proven exact.

Ported from ``research/scorer.py`` (the reference implementation every number in the PRD is
quoted from) and generalised over splits: the fuzzy set is selected per **(tool, argument)**,
never pooled across tools — pooling inflates the score and hides real errors
(docs/VERIFIED_FACTS.md §12.2).

The official formula (docs/VERIFIED_FACTS.md §1):

* gold empty  -> 1.0 iff the submission is empty, else 0.0
* else        -> state gate: the submitted state hash must equal the gold hash, else 0.0
* matching    -> same tool name; identical argument KEY SET; every non-fuzzy argument equal
                  with a type-sensitive comparison; every fuzzy argument matched by
                  token-F1 > 0.6 after lowercase + NFC
* one submitted call matches at most one gold call (maximum bipartite matching)
* P = M/(M+U), R = M/|G|, F1 = 2PR/(P+R)
* the sample score is F1; the split score is 100/N * sum of sample scores

The state hash always comes from the split's own runtime — never reimplemented here.
"""
from __future__ import annotations

import collections
import re
import unicodedata
from pathlib import Path

from .data import Split
from .kit import tool_env

FUZZY_MARKER = "x-match"
FUZZY_VALUE = "fuzzy"


def _norm(value) -> str:
    """Lowercase + NFC. Diacritics are preserved: the matcher is diacritic-sensitive."""
    return unicodedata.normalize("NFC", str(value).lower())


def _tokens(value) -> collections.Counter:
    return collections.Counter(re.findall(r"[^\W_]+", _norm(value)))


def token_f1(a, b) -> float:
    """Token-level F1 over NFC-lowercased word tokens."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 1.0 if _norm(a) == _norm(b) else 0.0
    common = sum((ta & tb).values())
    if common == 0:
        return 0.0
    precision = common / sum(ta.values())
    recall = common / sum(tb.values())
    return 2 * precision * recall / (precision + recall)


def fuzzy_arguments(catalog: dict) -> dict[str, set[str]]:
    """{tool_name: {argument names matched fuzzily}} for one catalog."""
    out: dict[str, set[str]] = {}
    for app in catalog.get("apps", []) or []:
        for tool in app.get("tools", []) or []:
            properties = (tool.get("parameters") or {}).get("properties") or {}
            fuzzy = {
                key
                for key, spec in properties.items()
                if (spec or {}).get(FUZZY_MARKER) == FUZZY_VALUE
            }
            if fuzzy:
                out[tool["name"]] = fuzzy
    return out


def calls_match(pred: dict, gold: dict, fuzzy: dict[str, set[str]]) -> bool:
    """True when one submitted call matches one gold call.

    * same tool name
    * identical argument key set (adding or dropping an argument breaks the match)
    * fuzzy arguments: token-F1 > 0.6
    * every other argument: type-sensitive equality
    """
    if pred.get("name") != gold.get("name"):
        return False
    pred_args = pred.get("arguments") or {}
    gold_args = gold.get("arguments") or {}
    if set(pred_args) != set(gold_args):
        return False
    fuzzy_keys = fuzzy.get(gold["name"], set())
    for key, gold_value in gold_args.items():
        pred_value = pred_args[key]
        if key in fuzzy_keys:
            if not (
                isinstance(pred_value, str)
                and isinstance(gold_value, str)
                and token_f1(pred_value, gold_value) > 0.6
            ):
                return False
        else:
            if type(pred_value) is not type(gold_value) or pred_value != gold_value:
                return False
    return True


def max_bipartite_match(preds: list[dict], golds: list[dict],
                        fuzzy: dict[str, set[str]]) -> int:
    """Maximum number of submitted calls that can be matched to distinct gold calls."""
    adjacency = [
        [j for j in range(len(golds)) if calls_match(pred, golds[j], fuzzy)]
        for pred in preds
    ]
    matched_gold = [-1] * len(golds)

    def augment(pred_index: int, seen: list[bool]) -> bool:
        for gold_index in adjacency[pred_index]:
            if seen[gold_index]:
                continue
            seen[gold_index] = True
            if matched_gold[gold_index] == -1 or augment(matched_gold[gold_index], seen):
                matched_gold[gold_index] = pred_index
                return True
        return False

    matched = 0
    for pred_index in range(len(preds)):
        if augment(pred_index, [False] * len(golds)):
            matched += 1
    return matched


def sample_score(pred_calls: list[dict], gold_calls: list[dict],
                 pred_hash: str, gold_hash: str,
                 fuzzy: dict[str, set[str]]) -> float:
    """The official per-sample score."""
    if not gold_calls:
        return 1.0 if not pred_calls else 0.0
    if pred_hash != gold_hash:
        return 0.0
    matched = max_bipartite_match(pred_calls, gold_calls, fuzzy)
    unmatched = len(pred_calls) - matched
    precision = matched / (matched + unmatched) if (matched + unmatched) else 0.0
    recall = matched / len(gold_calls)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def replay_hash(split: Split, sample_id: str, calls: list[dict]) -> str:
    """The state hash after replaying ``calls`` on a fresh session.

    One session per call sequence, always starting from the sample's initial state.
    """
    env = tool_env(split, sample_id)
    for call in calls or []:
        env.call(call.get("name", ""), call.get("arguments") or {})
    return env.get_state_hash()


def gold_hash(split: Split, sample: dict) -> str:
    """The gold state hash: the hash after replaying the gold trajectory."""
    return replay_hash(split, sample["id"], sample.get("label", {}).get("tool_calls") or [])


def score_split(split: Split, samples: list[dict], submissions: dict[str, list[dict]],
                progress=None) -> dict:
    """Score a whole split.

    ``submissions`` maps sample id -> list of submitted calls. The gold hash is computed
    with the split's own runtime, so the gate is exact.
    """
    fuzzy = fuzzy_arguments(split.catalog())
    total = 0.0
    perfect = 0
    gate_killed = 0
    rows = []
    for index, sample in enumerate(samples):
        sample_id = sample["id"]
        gold_calls = sample.get("label", {}).get("tool_calls") or []
        pred_calls = submissions.get(sample_id, [])
        if not gold_calls:
            score = 1.0 if not pred_calls else 0.0
            gold = ""
            pred = ""
            killed = False
        else:
            gold = gold_hash(split, sample)
            pred = replay_hash(split, sample_id, pred_calls)
            killed = pred != gold
            gate_killed += int(killed)
            score = sample_score(pred_calls, gold_calls, pred, gold, fuzzy)
        total += score
        perfect += int(score == 1.0)
        rows.append({
            "id": sample_id,
            "score": score,
            "gold_calls": len(gold_calls),
            "pred_calls": len(pred_calls),
            "gate_killed": killed,
        })
        if progress is not None and (index + 1) % progress == 0:
            print(f"  scored {index + 1}/{len(samples)}", flush=True)
    n = len(samples) or 1
    return {
        "split": split.name,
        "samples": len(samples),
        "mean": total / n,
        "score": 100.0 * total / n,
        "perfect": perfect,
        "gate_killed": gate_killed,
        "rows": rows,
    }


def load_submission_file(path: str | Path) -> dict[str, list[dict]]:
    """Read a submission JSONL: one row per sample, {"id", "tool_calls", "state_hash"}."""
    rows: dict[str, list[dict]] = {}
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            import json

            record = json.loads(line)
            sample_id = record.get("id")
            if not sample_id:
                raise ValueError(f"{path}:{number}: row has no 'id'")
            rows[sample_id] = record.get("tool_calls") or []
    return rows
