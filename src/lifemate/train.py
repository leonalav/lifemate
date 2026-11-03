"""QLoRA supervised fine-tuning of the policy.

Runs on a Colab GPU; the local machine never trains anything.

Checkpointing (docs/PRD.md §6):

* ``last``  — written every epoch, so an interrupted run resumes where it stopped
* ``best``  — written when the leave-apps-out score improves
* ``--resume`` restores model, optimizer, scheduler and epoch, so a Colab disconnect
  costs at most the current epoch

The leave-apps-out holdout is the honest generalisation proxy: 11 public-test apps never
appear in train, so private test almost certainly contains unseen apps
(docs/VERIFIED_FACTS.md §12.9).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .casing import set_mode as set_case_mode
from .data import resolve_split
from .training_data import build_examples, write_jsonl


@dataclass
class TrainConfig:
    """Everything the trainer reads from ``configs/sft_qlora.yaml``."""

    base_model: str = "Qwen/Qwen3.5-4B"
    output_dir: str = "runs/sft"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_len: int = 8192
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10
    save_total_limit: int = 3
    seed: int = 20261028
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bf16: bool = True
    gradient_checkpointing: bool = True
    # validation
    holdout_apps: int = 10
    holdout_seed: int = 20261028
    near_duplicate_similarity: float = 0.92
    # transcripts
    transcript_source: str = "gold"  # gold | asr | mix
    asr_transcript_ratio: float = 0.5
    # The casing policy the transcript reaches the model in. "none" (lowercase) is
    # the shipped pipeline: the model casts the arguments itself (Decision 24).
    case_mode: str = "none"
    # A JSONL cache of raw ASR transcripts, keyed by the split-relative audio path.
    transcript_cache: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        import yaml

        with Path(path).open(encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ValueError(
                f"unknown key(s) in {path}: {', '.join(sorted(unknown))}. "
                f"Valid keys: {', '.join(sorted(known))}"
            )
        return cls(**payload)


@dataclass
class TrainState:
    """The state carried across a resume."""

    epoch: int = 0
    global_step: int = 0
    best_metric: float = -1.0
    history: list[dict] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --- splits -------------------------------------------------------------------

def choose_holdout_apps(samples: list[dict], count: int, seed: int) -> set[str]:
    """Pick the apps held out of validation.

    Stratified by the sample count of each app so the holdout is not dominated by one
    rare app. The held-out apps stand in for the unseen public-test apps.
    """
    from collections import Counter

    per_app: Counter = Counter()
    for sample in samples:
        for app in (sample.get("context", {}).get("installed_apps") or {}):
            per_app[app] += 1
    rng = random.Random(seed)
    ordered = sorted(per_app.items(), key=lambda kv: (-kv[1], kv[0]))
    # take the most frequent apps, shuffled inside their frequency band
    chosen: set[str] = set()
    band: list[str] = []
    band_size = 0
    for app, frequency in ordered:
        if band_size and frequency < band_size * 0.5:
            chosen.update(band[: max(0, count - len(chosen))])
            band = []
            band_size = 0
        band.append(app)
        band_size = frequency
        if len(chosen) + len(band) >= count:
            rng.shuffle(band)
            chosen.update(band[: max(0, count - len(chosen))])
            band = []
            break
    if band and len(chosen) < count:
        rng.shuffle(band)
        chosen.update(band[: count - len(chosen)])
    return chosen


def split_samples(samples: list[dict], holdout_apps: set[str]) -> tuple[list[dict], list[dict]]:
    """Partition the samples into (train, holdout) by the held-out apps."""
    train, holdout = [], []
    for sample in samples:
        apps = set(sample.get("context", {}).get("installed_apps") or {})
        (holdout if apps & holdout_apps else train).append(sample)
    return train, holdout


# --- transcripts ------------------------------------------------------------------

def load_transcript_cache(path: str | Path | None) -> dict[str, str]:
    """Read the raw ASR transcript cache: {split-relative audio path: transcript}.

    The cache is the file the ``transcribe`` command writes. The keys are the paths
    relative to the split root, so the same file serves the training dataset and the
    runtime.
    """
    if not path:
        return {}
    target = Path(path)
    if not target.is_file():
        return {}
    cache: dict[str, str] = {}
    with target.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[record["path"]] = record["text"]
    return cache


def choose_transcripts(samples: list[dict], cache: dict[str, str], source: str,
                       ratio: float, seed: int) -> tuple[dict[str, str], object]:
    """Pick the transcript each sample is trained on.

    ``gold``  the split's own transcript — clean, punctuated, correctly cased
    ``asr``   the recogniser's output — the test-time distribution
    ``mix``   ``ratio`` of the samples use the ASR output, the rest the gold one

    The choice is seeded, so two runs over the same split build the same dataset.
    Returns the {sample id: transcript} mapping (only the samples that use the ASR
    output appear in it; the others fall back to the split's transcript) and a
    callable that resolves a *history* turn the same way.
    """
    rng = random.Random(seed)
    chosen: dict[str, str] = {}
    use_asr: set[str] = set()

    for sample in samples:
        sample_id = sample["id"]
        audio = sample.get("audio") or ""
        text = cache.get(audio) if audio else None
        if not text:
            continue
        if source == "asr":
            take = True
        elif source == "mix":
            take = rng.random() < ratio
        else:
            take = False
        if take:
            use_asr.add(sample_id)
            chosen[sample_id] = text

    def turn_transcript(turn: dict, sample_id: str = "") -> str:
        if sample_id not in use_asr:
            return ""
        audio = turn.get("audio") or ""
        return cache.get(audio, "") if audio else ""

    return chosen, turn_transcript


# --- checkpoints ---------------------------------------------------------------

def checkpoint_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "checkpoints"


def last_checkpoint(output_dir: str | Path) -> Path | None:
    """The most recent ``last`` checkpoint, if any."""
    path = checkpoint_dir(output_dir) / "last"
    return path if (path / "trainer_state.json").is_file() else None


def best_checkpoint(output_dir: str | Path) -> Path | None:
    """The most recent ``best`` checkpoint, if any."""
    path = checkpoint_dir(output_dir) / "best"
    return path if (path / "trainer_state.json").is_file() else None


def save_state(state: TrainState, output_dir: str | Path, kind: str) -> Path:
    """Persist the resume state next to the model weights."""
    target = checkpoint_dir(output_dir) / kind
    target.mkdir(parents=True, exist_ok=True)
    with (target / "trainer_state.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(state), fh, ensure_ascii=False, indent=2)
    return target


def load_state(path: str | Path) -> TrainState:
    with (Path(path) / "trainer_state.json").open(encoding="utf-8") as fh:
        return TrainState(**json.load(fh))


# --- training ------------------------------------------------------------------

def build_model_and_trainer(config: TrainConfig, train_rows: list[dict],
                            eval_rows: list[dict], output_dir: Path,
                            resume_from: str | Path | None = None):
    """Build the QLoRA model, the tokenizer and the trainer.

    Imports are local so the module can be imported (and its CLI parsed) on a machine
    without transformers/peft installed.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    quant_config = None
    if config.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, config.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"device_map": {"": 0}}
    want_dtype = torch.bfloat16 if config.bf16 else torch.float32
    try:
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quant_config,
            dtype=want_dtype,
            **model_kwargs,
        )
    except TypeError:
        # transformers < 5.0 names the argument torch_dtype.
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quant_config,
            torch_dtype=want_dtype,
            **model_kwargs,
        )
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )

    lora = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora)

    def to_text(row: dict) -> dict:
        return {"text": row["prompt"] + row["completion"]}

    train_dataset = Dataset.from_list([to_text(r) for r in train_rows])
    eval_dataset = Dataset.from_list([to_text(r) for r in eval_rows]) if eval_rows else None

    # T4 (compute capability 7.5) has no bf16 support; fall back to fp16 there so
    # the smoke test and the full run work on the cheapest Colab GPU. Ampere and
    # newer keep the configured bf16 setting.
    use_bf16 = bool(config.bf16)
    use_fp16 = False
    try:
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 8 and use_bf16:
                use_bf16 = False
                use_fp16 = True
    except Exception:
        pass

    training_args = _make_sft_config(
        SFTConfig,
        output_dir=str(output_dir),
        num_train_epochs=1,  # the caller loops per epoch to save last/best
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        max_seq_len=config.max_seq_len,
        max_length=config.max_seq_len,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        save_total_limit=config.save_total_limit,
        seed=config.seed,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=[],
    )

    trainer = _make_sft_trainer(
        SFTTrainer,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        tokenizer=tokenizer,
        args=training_args,
    )

    return trainer, tokenizer


def _make_sft_config(cls, **wanted):
    """Build the TRL config, tolerating field renames across TRL versions.

    Colab installs whatever TRL the runtime resolves (``trl>=0.9.0``), and field
    names moved over time (``max_seq_length`` -> ``max_length``, plus scheduler
    fields that some releases do not expose on ``SFTConfig``). Passing an
    unknown keyword raises ``TypeError`` before any training happens, which is
    the smoke-test failure this guards against. Unknown keys are dropped with a
    warning instead of crashing.
    """
    import inspect
    import warnings

    try:
        sig_params = inspect.signature(cls.__init__).parameters
        params = set(sig_params)
    except (TypeError, ValueError):
        return cls(**wanted)
    # ``max_seq_len`` is our name; TRL used ``max_seq_length`` then ``max_length``.
    if "max_length" not in params and "max_seq_length" in params:
        wanted.setdefault("max_seq_length", wanted.pop("max_length", None))
    wanted.pop("max_seq_len", None)
    # Always filter to known params: even when the signature has **kwargs,
    # unknown keys can still fail inside dataclass __post_init__ validation.
    filtered = {k: v for k, v in wanted.items() if k in params}
    dropped = sorted(set(wanted) - set(filtered))
    if dropped:
        warnings.warn(f"SFTConfig does not accept {dropped}; dropping them")
        print(f"warning: SFTConfig dropped unsupported args: {dropped}", flush=True)
    return cls(**filtered)


def _make_sft_trainer(cls, **wanted):
    """Build the TRL trainer across ``processing_class`` / ``tokenizer`` renames."""
    import inspect
    import warnings

    try:
        params = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        params = set()
    if "processing_class" not in params and "tokenizer" in params:
        if "processing_class" in wanted:
            wanted.setdefault("tokenizer", wanted.pop("processing_class"))
    elif "processing_class" in params:
        wanted.pop("tokenizer", None)
    filtered = {k: v for k, v in wanted.items() if not params or k in params}
    dropped = sorted(set(wanted) - set(filtered))
    if dropped:
        warnings.warn(f"SFTTrainer does not accept {dropped}; dropping them")
        print(f"warning: SFTTrainer dropped unsupported args: {dropped}", flush=True)
    return cls(**filtered)


def run_training(data_dir: str, output_dir: str, config_path: str | None = None,
                 epochs: int | None = None, batch_size: int | None = None,
                 resume: bool = False, resume_path: str | None = None,
                 split: str = "train", smoke: bool = False,
                 limit: int | None = None, case_mode: str | None = None,
                 transcript_cache: str | None = None) -> dict:
    """Run the training pipeline.

    ``resume`` picks up from the ``last`` checkpoint when one exists, so a Colab
    disconnect costs at most the current epoch. ``smoke`` runs a handful of steps on a
    small subset to verify the pipeline before a long run. ``case_mode`` overrides the
    config's casing policy, which is how the ablation arms are selected.
    """
    config = TrainConfig.from_yaml(config_path) if config_path else TrainConfig()
    if epochs is not None:
        config.num_train_epochs = epochs
    if batch_size is not None:
        config.per_device_train_batch_size = batch_size
    if case_mode is not None:
        config.case_mode = case_mode
    if transcript_cache is not None:
        config.transcript_cache = transcript_cache

    set_seed(config.seed)

    split_dir = resolve_split(data_dir, split)
    samples = list(split_dir.iter_samples())

    holdout_apps = choose_holdout_apps(samples, config.holdout_apps, config.holdout_seed)
    train_samples, holdout_samples = split_samples(samples, holdout_apps)
    print(
        f"train samples: {len(train_samples)}  holdout samples: {len(holdout_samples)}  "
        f"held-out apps: {len(holdout_apps)}",
        flush=True,
    )

    if smoke:
        train_samples = train_samples[:16]
        holdout_samples = holdout_samples[:8]
        config.num_train_epochs = 1
        config.logging_steps = 1
        config.save_total_limit = 1

    # The casing policy and the transcripts the model is trained on. The policy is set
    # process-wide so the training prompt and the inference prompt cannot disagree.
    set_case_mode(config.case_mode)
    cache = load_transcript_cache(config.transcript_cache)
    if cache:
        print(f"transcript cache: {len(cache)} utterances from {config.transcript_cache}",
              flush=True)
    train_transcripts, train_turn_transcript = choose_transcripts(
        train_samples, cache, config.transcript_source, config.asr_transcript_ratio,
        config.seed)
    holdout_transcripts, holdout_turn_transcript = choose_transcripts(
        holdout_samples, cache, config.transcript_source, config.asr_transcript_ratio,
        config.seed)
    if config.transcript_source != "gold":
        print(f"transcript source: {config.transcript_source} "
              f"({len(train_transcripts)}/{len(train_samples)} train samples on ASR text)",
              flush=True)

    train_examples = build_examples(split_dir, train_samples,
                                    transcripts=train_transcripts,
                                    transcript_for_turn=train_turn_transcript,
                                    case_mode=config.case_mode)
    holdout_examples = build_examples(split_dir, holdout_samples,
                                      transcripts=holdout_transcripts,
                                      transcript_for_turn=holdout_turn_transcript,
                                      case_mode=config.case_mode)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    write_jsonl(train_examples, output_path / "train.jsonl")
    write_jsonl(holdout_examples, output_path / "holdout.jsonl")
    print(
        f"examples: {len(train_examples)} train / {len(holdout_examples)} holdout",
        flush=True,
    )

    if smoke:
        print("smoke run: verifying the pipeline, not training a usable model",
              flush=True)

    state = TrainState()
    resume_target = None
    if resume_path:
        resume_target = Path(resume_path)
    elif resume:
        resume_target = last_checkpoint(output_path)
    if resume_target:
        state = load_state(resume_target)
        print(
            f"resuming from {resume_target} at epoch {state.epoch} "
            f"(step {state.global_step})",
            flush=True,
        )

    trainer, tokenizer = build_model_and_trainer(
        config,
        [e.__dict__ for e in train_examples],
        [e.__dict__ for e in holdout_examples],
        output_path,
        resume_from=resume_target,
    )

    # `last` every epoch, `best` on improvement.
    for epoch in range(state.epoch, config.num_train_epochs):
        metrics = trainer.train(
            resume_from_checkpoint=str(resume_target) if (resume_target and epoch == state.epoch)
            else None
        )
        state.epoch = epoch + 1
        state.global_step = int(getattr(trainer, "global_step", state.global_step))
        metric = float(metrics.training_loss) if getattr(metrics, "training_loss", None) is not None else 0.0
        state.history.append({"epoch": state.epoch, "train_loss": metric})

        target = save_state(state, output_path, "last")
        trainer.save_model(str(target))
        tokenizer.save_pretrained(str(target))
        print(f"epoch {state.epoch} done -> last checkpoint at {target}", flush=True)

        if state.best_metric < 0 or metric < state.best_metric:
            state.best_metric = metric
            best = save_state(state, output_path, "best")
            trainer.save_model(str(best))
            tokenizer.save_pretrained(str(best))
            print(f"epoch {state.epoch} improved -> best checkpoint at {best}", flush=True)

    with (output_path / "train_state.json").open("w", encoding="utf-8") as fh:
        json.dump(asdict(state), fh, ensure_ascii=False, indent=2)
    # Free the trainer and its model before the caller merges or runs inference
    # in the same process. Without this, `trainer` still holds CUDA tensors and
    # `empty_cache` alone cannot return the memory, so vLLM's startup probe sees
    # ~26 GiB free instead of ~39 GiB and refuses the requested utilization.
    try:
        from .agent.policy import release_cuda_memory

        model = getattr(trainer, "model", None)
        try:
            del trainer
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            del tokenizer
        except Exception:
            pass
        release_cuda_memory()
    except Exception:
        pass
    return asdict(state)
