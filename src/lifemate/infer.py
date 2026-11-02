"""Inference: run the agent over a split and write a submission.

The output is UTF-8 JSONL, one row per sample, in the split's own manifest order:

    {"id": "...", "tool_calls": [...], "state_hash": "..."}

The state hash always comes from the split's own runtime, so it is computed with the same
canonicalisation the organiser's scorer uses (docs/VERIFIED_FACTS.md §12.6).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .agent.loop import Agent
from .agent.policy import Policy
from .asr import TranscriptStore, audio_path
from .casing import set_mode as set_case_mode
from .data import Split, resolve_split
from .truecaser import load_truecaser


@dataclass
class InferConfig:
    """Runtime knobs for inference (docs/PRD.md §4.2)."""

    max_steps: int = 8
    time_budget_s: float = 25.0
    model_path: str = ""
    transcript_cache: str = ""
    asr_model_dir: str = ""
    limit: int | None = None
    ids_file: str | None = None
    verbose: bool = True
    # The casing policy the transcript reaches the model in. Empty means the active
    # one (lowercase by default). "rules" and "model" are the ablation baselines.
    case_mode: str = ""
    # Whether a pretrained truecaser contributes a casing hint to the prompt.
    case_hint: bool = False
    # Which truecaser weights to use; empty means the default (PhoBERT).
    truecaser_model: str = ""
    # Force the ASR transcript even on a labelled split, where the split's own
    # transcript would otherwise be used. The ablation needs this: the model is
    # trained on the recogniser's output and must be scored on it.
    use_asr_transcripts: bool = False


@dataclass
class InferReport:
    """What the run did — used by the notebook to print a useful summary."""

    split: str = ""
    samples: int = 0
    rows_written: int = 0
    tool_calls: int = 0
    stop_reasons: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    output_file: str = ""


def load_ids(path: str | Path) -> list[str]:
    """Read a newline-separated id list (or a JSON list)."""
    text = Path(path).read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return [str(x) for x in json.loads(text)]
    return [line.strip() for line in text.splitlines() if line.strip()]


def make_transcript_fn(split: Split, store: TranscriptStore, prefer_asr: bool = False):
    """A function that returns the transcript of any turn, for the prompt builder.

    A labelled split carries its own ``audio_text``; an unlabelled split does not, so
    the audio is transcribed. The labelled text is preferred when it exists because it
    is free and exact — unless ``prefer_asr`` is set, which is what the ablation needs:
    the policy is scored on the recogniser's output, the same text it was trained on.
    """

    def transcript_fn(turn: dict) -> str:
        if not prefer_asr:
            text = turn.get("audio_text")
            if text:
                return text
        relative = turn.get("audio")
        if not relative:
            return ""
        path = split.path / relative
        if not path.is_file():
            return ""
        return store.get(path)

    return transcript_fn


def run_inference(data_dir: str, split_name: str | None, policy: Policy,
                  output_file: str, config: InferConfig) -> InferReport:
    """Run the agent over a split and write the submission file."""
    split = resolve_split(data_dir, split_name)
    samples = list(split.iter_samples())

    if config.ids_file:
        wanted = set(load_ids(config.ids_file))
        samples = [s for s in samples if s["id"] in wanted]
    if config.limit:
        samples = samples[: config.limit]

    store = TranscriptStore(config.transcript_cache or None, root=split.path)
    transcript_fn = make_transcript_fn(split, store, config.use_asr_transcripts)

    # The casing policy is process-wide, so the prompt builder and the runtime agree.
    if config.case_mode:
        set_case_mode(config.case_mode)

    # The truecaser is a hint, not a dependency: when the weights are not in the image
    # the pipeline runs without it.
    case_fn = None
    if config.case_hint:
        case_fn = load_truecaser(config.truecaser_model or None)
        if case_fn is None:
            print("  the truecaser is unavailable — running without the casing hint",
                  flush=True)

    agent = Agent(
        split,
        policy,
        max_steps=config.max_steps,
        time_budget_s=config.time_budget_s,
        transcript_fn=transcript_fn,
        case_fn=case_fn,
        case_mode=config.case_mode or None,
    )

    report = InferReport(split=split.name, output_file=output_file)
    started = time.monotonic()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fh:
        for index, sample in enumerate(samples):
            # A labelled split carries its own transcript; an unlabelled one does not,
            # so the audio is transcribed (and cached by path). The ablation forces
            # the ASR path even on a labelled split.
            transcript = ""
            if not config.use_asr_transcripts:
                transcript = sample.get("audio_text") or ""
            if not transcript:
                wav = audio_path(sample, split.path)
                transcript = store.get(wav) if wav is not None else ""

            trajectory = agent.run(sample, transcript)
            fh.write(json.dumps(trajectory.as_submission_row(), ensure_ascii=False) + "\n")
            fh.flush()

            report.rows_written += 1
            report.tool_calls += len(trajectory.tool_calls)
            reason = trajectory.stop_reason or "unknown"
            report.stop_reasons[reason] = report.stop_reasons.get(reason, 0) + 1
            if config.verbose and (index + 1) % 25 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"  {index + 1}/{len(samples)} samples  "
                    f"({elapsed / (index + 1):.1f}s/sample)",
                    flush=True,
                )

    store.flush()
    report.samples = len(samples)
    report.elapsed_s = time.monotonic() - started
    return report


def validate_submission(submission_file: str | Path, split: Split) -> dict:
    """Pre-flight check of a submission file.

    Verifies the things the organiser's scorer will trip on: every sample present exactly
    once, in manifest order, and every proposed call schema-valid against the split's
    catalog and the sample's installed apps.
    """
    from .kit import schema_error

    expected = split.ids()
    seen: list[str] = []
    invalid: list[dict] = []
    unknown: list[str] = []

    with Path(submission_file).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = record.get("id")
            seen.append(sample_id)
            if sample_id not in expected:
                unknown.append(sample_id)
                continue
            installed = set()
            for app in (split.sample(sample_id).get("context", {})
                        .get("installed_apps") or {}).values():
                installed.update(app.get("tools") or [])
            installed.add("sys.ask_missing_info")
            for call in record.get("tool_calls") or []:
                error = schema_error(split, call.get("name", ""),
                                     call.get("arguments") or {}, installed)
                if error:
                    invalid.append({"line": number, "id": sample_id,
                                    "name": call.get("name"), "error": error})

    missing = [i for i in expected if i not in set(seen)]
    duplicates = [i for i in set(seen) if seen.count(i) > 1]
    return {
        "rows": len(seen),
        "expected": len(expected),
        "missing": missing,
        "duplicates": sorted(duplicates),
        "unknown": unknown,
        "invalid_calls": invalid,
        "ok": not missing and not duplicates and not unknown and not invalid,
    }
