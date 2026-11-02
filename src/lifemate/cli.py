"""The command line. The contract is frozen in docs/PRD.md §6.

    python -m lifemate train      --data-dir DIR --output-dir DIR --config FILE
                                   [--resume [PATH]] [--epochs N] [--batch-size N] [--smoke]
    python -m lifemate infer      --data-dir DIR --split NAME --checkpoint DIR
                                   --output-file FILE [--limit N] [--time-budget-s 25]
                                   [--max-steps 8]
    python -m lifemate score      --split train --submission FILE [--ids-file FILE]
                                   [--report FILE]
    python -m lifemate transcribe --split NAME --data-dir DIR --out FILE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lifemate",
        description="LifeMate — offline voice-to-action agent (Zalo AI Challenge 2026).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train -----------------------------------------------------------------------
    train = sub.add_parser("train", help="QLoRA supervised fine-tuning of the policy.")
    train.add_argument("--data-dir", required=True,
                       help="data root holding the split folders")
    train.add_argument("--output-dir", required=True,
                       help="output root for checkpoints, logs and the dataset dump")
    train.add_argument("--config", default=None,
                       help="training config (configs/sft_qlora.yaml)")
    train.add_argument("--resume", nargs="?", const="auto", default=None,
                       help="resume from a checkpoint; with no value, from the 'last' "
                            "checkpoint in --output-dir")
    train.add_argument("--epochs", type=int, default=None)
    train.add_argument("--batch-size", type=int, default=None)
    train.add_argument("--split", default="train", help="split to train on")
    train.add_argument("--limit", type=int, default=None,
                       help="use only the first N samples")
    train.add_argument("--smoke", action="store_true",
                       help="verify the pipeline on a tiny subset instead of training")

    # --- infer -----------------------------------------------------------------------
    infer = sub.add_parser("infer", help="run the agent over a split.")
    infer.add_argument("--data-dir", required=True)
    infer.add_argument("--split", default=None,
                       help="split name; discovered from --data-dir when omitted")
    infer.add_argument("--checkpoint", required=True,
                       help="checkpoint directory (the 'best' or 'last' one)")
    infer.add_argument("--output-file", required=True,
                       help="where to write the submission JSONL")
    infer.add_argument("--limit", type=int, default=None)
    infer.add_argument("--ids-file", default=None,
                       help="restrict inference to the ids in this file")
    infer.add_argument("--time-budget-s", type=float, default=25.0)
    infer.add_argument("--max-steps", type=int, default=8)
    infer.add_argument("--model-path", default=None,
                       help="base model weights (defaults to the checkpoint's config)")
    infer.add_argument("--transcript-cache", default=None,
                       help="JSONL cache of transcripts, reused across runs")
    infer.add_argument("--asr-model-dir", default=None,
                       help="directory holding the gipformer ONNX bundle")
    infer.add_argument("--no-asr", action="store_true",
                       help="skip transcription and use the split's own text field")

    # --- score -----------------------------------------------------------------------
    score = sub.add_parser("score", help="score a submission against the gold labels.")
    score.add_argument("--split", required=True, help="a labelled split, e.g. 'train'")
    score.add_argument("--data-dir", default=None,
                       help="data root; defaults to data/ next to the repository")
    score.add_argument("--submission", required=True)
    score.add_argument("--ids-file", default=None)
    score.add_argument("--report", default=None, help="write the per-sample report here")
    score.add_argument("--limit", type=int, default=None)

    # --- transcribe ------------------------------------------------------------------
    transcribe = sub.add_parser("transcribe", help="transcribe a split's audio.")
    transcribe.add_argument("--split", required=True)
    transcribe.add_argument("--data-dir", required=True)
    transcribe.add_argument("--out", required=True,
                            help="output JSONL: {path, text} per utterance")
    transcribe.add_argument("--asr-model-dir", default=None)
    transcribe.add_argument("--limit", type=int, default=None)

    # --- merge ------------------------------------------------------------------------
    merge = sub.add_parser("merge", help="merge the LoRA adapter into the base model.")
    merge.add_argument("--checkpoint", required=True,
                       help="the adapter directory (e.g. <output>/checkpoints/best)")
    merge.add_argument("--output-dir", required=True,
                       help="where to write the merged model")
    merge.add_argument("--model-path", default=None,
                       help="the base model; read from adapter_config.json when omitted")

    # --- validate ---------------------------------------------------------------------
    validate = sub.add_parser("validate", help="pre-flight check a submission file.")
    validate.add_argument("--split", required=True)
    validate.add_argument("--data-dir", default=None)
    validate.add_argument("--submission", required=True)

    return parser


def _default_data_dir() -> str:
    """The repository's data folder, so --data-dir can be omitted for score/validate."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "data"
        if candidate.is_dir():
            return str(candidate)
    return "data"


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        from .train import run_training

        state = run_training(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            resume=args.resume is not None,
            resume_path=None if args.resume in (None, "auto") else args.resume,
            split=args.split,
            smoke=args.smoke,
            limit=args.limit,
        )
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0

    if args.command == "infer":
        from .agent.policy import VllmPolicy
        from .infer import InferConfig, run_inference

        checkpoint = Path(args.checkpoint)
        model_path = args.model_path or _model_from_checkpoint(checkpoint)
        policy = VllmPolicy(model_path)
        config = InferConfig(
            max_steps=args.max_steps,
            time_budget_s=args.time_budget_s,
            model_path=model_path,
            transcript_cache=args.transcript_cache or "",
            asr_model_dir=args.asr_model_dir or "",
            limit=args.limit,
            ids_file=args.ids_file,
        )
        report = run_inference(args.data_dir, args.split, policy,
                               args.output_file, config)
        print(json.dumps(
            {k: v for k, v in report.__dict__.items() if k != "stop_reasons"},
            ensure_ascii=False, indent=2))
        print("stop reasons:", report.stop_reasons)
        return 0

    if args.command == "score":
        from .data import resolve_split
        from .infer import load_ids
        from .metric import load_submission_file, score_split

        data_dir = args.data_dir or _default_data_dir()
        split = resolve_split(data_dir, args.split)
        samples = list(split.iter_samples())
        if args.ids_file:
            wanted = set(load_ids(args.ids_file))
            samples = [s for s in samples if s["id"] in wanted]
        if args.limit:
            samples = samples[: args.limit]

        submissions = load_submission_file(args.submission)
        result = score_split(split, samples, submissions)
        report = {
            "split": result["split"],
            "samples": result["samples"],
            "score": round(result["score"], 6),
            "mean": round(result["mean"], 9),
            "perfect": result["perfect"],
            "gate_killed": result["gate_killed"],
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.report:
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as fh:
                for row in result["rows"]:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"per-sample report -> {path}")
        return 0

    if args.command == "transcribe":
        from .asr import Recognizer, TranscriptStore, audio_path
        from .data import resolve_split

        split = resolve_split(args.data_dir, args.split)
        recognizer = Recognizer(args.asr_model_dir)
        store = TranscriptStore(args.out, recognizer)
        samples = list(split.iter_samples())
        if args.limit:
            samples = samples[: args.limit]
        count = 0
        for sample in samples:
            wav = audio_path(sample, split.path)
            if wav is None:
                continue
            store.get(wav)
            for turn in sample.get("history") or []:
                relative = turn.get("audio")
                if relative:
                    path = split.path / relative
                    if path.is_file():
                        store.get(path)
            count += 1
            if count % 50 == 0:
                print(f"  transcribed {count} samples", flush=True)
        store.flush()
        print(f"transcripts -> {args.out}  ({len(store.cache)} utterances)")
        return 0

    if args.command == "merge":
        from .merge import merge_adapter

        merge_adapter(args.checkpoint, args.output_dir, args.model_path)
        return 0

    if args.command == "validate":
        from .data import resolve_split
        from .infer import validate_submission

        data_dir = args.data_dir or _default_data_dir()
        split = resolve_split(data_dir, args.split)
        result = validate_submission(args.submission, split)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["ok"] else 1

    parser.error(f"unknown command {args.command!r}")
    return 2


def _model_from_checkpoint(checkpoint: Path) -> str:
    """The base model a checkpoint was trained from, read from its adapter config."""
    config = checkpoint / "adapter_config.json"
    if config.is_file():
        with config.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        base = payload.get("base_model_name_or_path")
        if base:
            return base
    return "Qwen/Qwen3.5-4B"


if __name__ == "__main__":
    sys.exit(main())
