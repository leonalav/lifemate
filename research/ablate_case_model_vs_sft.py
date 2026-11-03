"""Ablation: does the policy model cast the arguments better than the recasing code?

The pipeline no longer restores the capitalisation the recogniser threw away. The
transcript reaches the model in lowercase, and the model — trained on lowercase
transcripts paired with the correctly-cased gold arguments — casts the arguments
itself (docs/DECISIONS.md, Decision 24).

That is a hypothesis, not a result. This script measures it, one arm per invocation:

    A   case_mode=model, no hint   the shipped pipeline: the context model recases the
                                   transcript and the policy model copies the case
                                   from it. This is the baseline to beat.
    B   case_mode=none,  no hint   the transcript is lowercase; the policy model has to
                                   cast the arguments itself.
    C   case_mode=none,  hint on   the same, plus a pretrained truecaser's recasing in
                                   the prompt — the safety net for names the model has
                                   never seen.

Every arm is a full cycle on the same split: build the dataset, train, merge, infer,
score. The arms differ in exactly one switch, so the difference is the switch.

**The metric that decides it.** The scorer compares an argument exactly unless the
tool marks it fuzzy, and a fuzzy argument is matched on token-F1, which lowercases
both sides — so capitalisation only matters where the comparison is exact. The report
therefore counts the *exactly-compared capitalised arguments*: the arguments whose gold
value carries a capital, split into

    exact          the submission spells it exactly like the gold
    case_only      the same words, the wrong case ("mai anh" for "Mai Anh")
    other          different words — the model did not find the value at all
    no_call        the submission did not make that call, so there is nothing to compare

Run (on Colab — both training and inference need a GPU):

    python research/ablate_case_model_vs_sft.py --arm B \
        --data-dir data/train+public_test \
        --transcripts /content/drive/transcripts_train.jsonl \
        --output-dir /content/drive/ablation

    python research/ablate_case_model_vs_sft.py --summarize \
        --output-dir /content/drive/ablation
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.casing import set_mode  # noqa: E402
from lifemate.data import resolve_split  # noqa: E402
from lifemate.metric import fuzzy_arguments, score_split  # noqa: E402

ARMS = {
    "A": {"case_mode": "model", "case_hint": False,
          "label": "context model recases the transcript (the shipped pipeline)"},
    "B": {"case_mode": "none", "case_hint": False,
          "label": "lowercase transcript, the policy model casts the arguments"},
    "C": {"case_mode": "none", "case_hint": True,
          "label": "lowercase transcript plus a truecaser hint in the prompt"},
}


def _has_upper(text: str) -> bool:
    return any(char.isupper() for char in text)


def _looks_like_timestamp(value: str) -> bool:
    """True for an ISO date-time, whose capitals are part of the format.

    "2026-11-29T09:00:00+07:00" carries an uppercase "T", but that is the format, not
    a name the model failed to capitalise.
    """
    return len(value) >= 16 and value[4] == "-" and value[10] in ("T", " ")


def argument_case_report(split, samples: list[dict], submissions: dict) -> dict:
    """How the submission spells the exactly-compared capitalised arguments.

    The population is every string argument the scorer compares exactly (not fuzzy)
    whose gold value carries a capital. Fuzzy arguments are excluded: token-F1
    lowercases both sides, so a capital cannot change their score.
    """
    fuzzy = fuzzy_arguments(split.catalog())
    stats: collections.Counter = collections.Counter()
    misses: list[dict] = []

    for sample in samples:
        gold_calls = sample.get("label", {}).get("tool_calls") or []
        pred_calls = submissions.get(sample["id"], []) or []

        for gold in gold_calls:
            tool = gold.get("name") or ""
            fuzzy_keys = fuzzy.get(tool, set())
            gold_args = gold.get("arguments") or {}

            # Every submitted call with the same name and the same argument key set.
            # A trajectory may make the same call twice, so the best match wins.
            candidates = [
                pred for pred in pred_calls
                if pred.get("name") == tool
                and set(pred.get("arguments") or {}) == set(gold_args)
            ]

            for key, gold_value in gold_args.items():
                if not isinstance(gold_value, str) or key in fuzzy_keys:
                    continue
                if _looks_like_timestamp(gold_value) or not _has_upper(gold_value):
                    continue
                stats["population"] += 1
                if not candidates:
                    stats["no_call"] += 1
                    continue

                submitted = [
                    (c.get("arguments") or {}).get(key) for c in candidates
                ]
                if gold_value in submitted:
                    stats["exact"] += 1
                    continue
                folded = [
                    value for value in submitted
                    if isinstance(value, str)
                    and value.casefold() == gold_value.casefold()
                ]
                if folded:
                    stats["case_only"] += 1
                    misses.append({
                        "id": sample["id"], "tool": tool, "argument": key,
                        "gold": gold_value, "submitted": folded[0],
                    })
                    continue
                other = [value for value in submitted if isinstance(value, str)]
                stats["other"] += 1
                misses.append({
                    "id": sample["id"], "tool": tool, "argument": key,
                    "gold": gold_value,
                    "submitted": other[0] if other else None,
                })

    population = stats["population"] or 1
    return {
        "population": stats["population"],
        "exact": stats["exact"],
        "case_only_miss": stats["case_only"],
        "other_miss": stats["other"],
        "no_call": stats["no_call"],
        "exact_percent": round(100.0 * stats["exact"] / population, 2),
        "case_only_miss_percent": round(100.0 * stats["case_only"] / population, 2),
        "other_miss_percent": round(100.0 * stats["other"] / population, 2),
        "examples": misses[:40],
    }


def run_arm(args) -> dict:
    """One arm: train, merge, infer, score."""
    from lifemate.agent.policy import VllmPolicy
    from lifemate.infer import InferConfig, run_inference
    from lifemate.merge import merge_adapter
    from lifemate.train import run_training

    arm = ARMS[args.arm]
    arm_dir = Path(args.output_dir) / f"arm_{args.arm}"
    arm_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== arm {args.arm}: {arm['label']} ===", flush=True)
    print(f"    case mode: {arm['case_mode']}   case hint: {arm['case_hint']}", flush=True)

    # The policy is process-wide: the dataset and the runtime must agree on it.
    set_mode(arm["case_mode"])

    transcripts = args.transcripts
    if transcripts and not Path(transcripts).is_file():
        raise SystemExit(
            f"transcript cache not found: {transcripts}\n"
            "Build it first:\n"
            "  python -m lifemate transcribe --split train --data-dir DATA "
            "--out TRANSCRIPTS --asr-model-dir MODEL_DIR"
        )

    # 1. train -------------------------------------------------------------------
    if not args.skip_train:
        print("--- training ---", flush=True)
        run_training(
            data_dir=args.data_dir,
            output_dir=str(arm_dir),
            config_path=args.config,
            epochs=args.epochs,
            split=args.split,
            limit=args.limit,
            case_mode=arm["case_mode"],
            transcript_cache=transcripts,
        )

    # 2. merge -------------------------------------------------------------------
    checkpoint = arm_dir / "checkpoints" / "best"
    if not (checkpoint / "adapter_config.json").is_file():
        checkpoint = arm_dir / "checkpoints" / "last"
    merged = arm_dir / "merged_model"
    if not (merged / "config.json").is_file():
        print("--- merging the adapter ---", flush=True)
        merge_adapter(checkpoint, str(merged))

    # 3. infer -------------------------------------------------------------------
    submission_file = arm_dir / f"submission_{args.split}.jsonl"
    print("--- inference ---", flush=True)
    from lifemate.agent.policy import release_cuda_memory
    release_cuda_memory()
    policy = VllmPolicy(str(merged),
                        max_model_len=args.max_model_len,
                        gpu_memory_utilization=args.gpu_memory_utilization)
    config = InferConfig(
        max_steps=args.max_steps,
        time_budget_s=args.time_budget_s,
        model_path=str(merged),
        transcript_cache=transcripts,
        limit=args.limit,
        case_mode=arm["case_mode"],
        case_hint=arm["case_hint"],
        truecaser_model=args.truecaser_model,
        use_asr_transcripts=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
    )
    run_inference(args.data_dir, args.split, policy, str(submission_file), config)

    # 4. score -------------------------------------------------------------------
    from lifemate.metric import load_submission_file

    split = resolve_split(args.data_dir, args.split)
    samples = list(split.iter_samples())
    if args.limit:
        samples = samples[: args.limit]
    submissions = load_submission_file(submission_file)
    result = score_split(split, samples, submissions)

    report = {
        "arm": args.arm,
        "label": arm["label"],
        "case_mode": arm["case_mode"],
        "case_hint": arm["case_hint"],
        "split": args.split,
        "samples": result["samples"],
        "metric_score": round(result["score"], 4),
        "perfect": result["perfect"],
        "gate_killed": result["gate_killed"],
        "arguments": argument_case_report(split, samples, submissions),
    }
    report_path = arm_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"report -> {report_path}", flush=True)
    return report


def summarize(output_dir: str) -> int:
    """Print the comparison table of every arm that has a report."""
    root = Path(output_dir)
    reports = []
    for path in sorted(root.glob("arm_*/report.json")):
        with path.open(encoding="utf-8") as fh:
            reports.append(json.load(fh))
    if not reports:
        print(f"no reports under {root}")
        return 1

    print(f"{'arm':<4} {'casing':<8} {'hint':<6} {'metric':>8} {'exact args':>11} "
          f"{'case-only':>10} {'other':>7}")
    print("-" * 60)
    for report in reports:
        arguments = report["arguments"]
        print(f"{report['arm']:<4} {report['case_mode']:<8} "
              f"{str(report['case_hint']):<6} {report['metric_score']:>8.3f} "
              f"{arguments['exact_percent']:>10.2f}% "
              f"{arguments['case_only_miss_percent']:>9.2f}% "
              f"{arguments['other_miss_percent']:>6.2f}%")
    print()
    for report in reports:
        print(f"arm {report['arm']}: {report['label']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default=None,
                        help="the arm to run")
    parser.add_argument("--summarize", action="store_true",
                        help="print the comparison of the arms that have run")
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", default="runs/ablation")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs/sft_qlora.yaml"))
    parser.add_argument("--transcripts", default="",
                        help="the raw ASR transcript cache (from 'lifemate transcribe')")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N samples (a quick signal)")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--time-budget-s", type=float, default=25.0)
    parser.add_argument("--truecaser-model", default="")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6,
                        help="fraction of GPU memory vLLM may use (lower on small GPUs "
                             "or when inference shares the process with training)")
    parser.add_argument("--max-model-len", type=int, default=16384,
                        help="vLLM context length (lower to save KV-cache memory)")
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse the checkpoints already in the arm's directory")
    args = parser.parse_args()

    if args.summarize:
        return summarize(args.output_dir)
    if not args.arm:
        parser.error("--arm is required (A, B or C), or use --summarize")
    run_arm(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
