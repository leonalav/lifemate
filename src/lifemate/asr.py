"""Speech recognition.

The plan of record is **stock gipformer1.5-68M-rnnt** via sherpa-onnx on CPU: 68M params,
MIT, released 2026-08-21 (pre-cutoff), ONNX published upstream so there is no export step,
and it runs on CPU so it never contends with the LLM for VRAM
(docs/VERIFIED_FACTS.md §5, docs/DECISIONS.md, Decision 2).

Fine-tuning the ASR is a stretch goal gated behind a diagnostic, not the default.

Transcription is cached by audio path so re-runs are identical and a resumed run does not
re-transcribe what it already has. The cached text is the **raw** recogniser output: the
capitalisation policy lives in ``casing.py`` and is applied where the text becomes model
input (Decision 24).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MODEL_ID = "g-group-ai-lab/gipformer1.5-68M-rnnt"
MODEL_FILES = ("encoder.onnx", "decoder.onnx", "joiner.onnx", "tokens.txt")


def _read_wave(wav_path: str | Path) -> tuple["object", int]:
    """Load a mono 16-bit wav file as normalised float32 samples.

    ``sherpa_onnx.read_wave`` does not exist in the installed package; the
    official offline-decode example defines this helper locally with stdlib
    ``wave`` (see ``python-api-examples/offline-decode-files.py``), so we do
    the same. Returns ``(samples, sample_rate)``.
    """
    import wave

    import numpy as np

    with wave.open(str(wav_path), "rb") as fh:
        if fh.getnchannels() != 1:
            raise ValueError(f"{wav_path}: expected mono audio, got {fh.getnchannels()} channels")
        if fh.getsampwidth() != 2:
            raise ValueError(f"{wav_path}: expected 16-bit audio")
        frames = fh.readframes(fh.getnframes())
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, fh.getframerate()


@dataclass
class Transcript:
    """One transcription."""

    path: str
    text: str


class Recognizer:
    """gipformer via sherpa-onnx, running on CPU.

    The ONNX bundle is fetched from Hugging Face on first use and cached under
    ``cache_dir`` (on Drive in Colab, so a session reset does not re-download it).
    """

    def __init__(self, model_dir: str | Path | None = None, cache_dir: str | Path | None = None,
                 num_threads: int = 4, use_int8: bool = True):
        import sherpa_onnx

        self.model_dir = Path(model_dir) if model_dir else self._download(cache_dir)
        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            tokens=str(self.model_dir / "tokens.txt"),
            encoder=str(self.model_dir / "encoder.onnx"),
            decoder=str(self.model_dir / "decoder.onnx"),
            joiner=str(self.model_dir / "joiner.onnx"),
            num_threads=num_threads,
        )

    @staticmethod
    def _download(cache_dir: str | Path | None) -> Path:
        """Fetch the ONNX bundle from Hugging Face into the cache directory."""
        from huggingface_hub import hf_hub_download

        target = Path(cache_dir or Path.home() / ".cache" / "gipformer")
        target.mkdir(parents=True, exist_ok=True)
        for name in MODEL_FILES:
            hf_hub_download(
                repo_id=MODEL_ID,
                filename=name,
                local_dir=str(target),
            )
        return target

    def transcribe(self, wav_path: str | Path) -> str:
        """Transcribe one 16 kHz mono wav file.

        gipformer emits uppercase text with no punctuation. The pipeline keeps that
        raw form: the casing policy in ``casing.py`` is applied where the text
        becomes model input, not here, so one cached transcript serves every policy
        (lowercase for the shipped pipeline, restored for the ablation arms).

        The recasing that used to live here — the rule module and the context model
        over it — is the ablation baseline, not the runtime path (Decision 24).
        """
        samples, sample_rate = _read_wave(wav_path)
        stream = self._recognizer.create_stream()
        stream.accept_waveform(sample_rate, samples)
        self._recognizer.decode_stream(stream)
        return stream.result.text


class TranscriptStore:
    """Transcripts cached by audio path, persisted as JSONL.

    A resumed run reuses everything already transcribed, which keeps a re-run cheap and
    guarantees the model sees the same text it saw before.

    Keys are **split-relative** when a root is given, so the same cache file serves
    training and inference: the trainer looks a transcript up by the sample's
    ``audio`` field, the runtime by the path it resolves from the split, and both
    land on the same key.
    """

    def __init__(self, path: str | Path | None = None, recognizer: Recognizer | None = None,
                 root: str | Path | None = None):
        self.path = Path(path) if path else None
        self.recognizer = recognizer
        self.root = Path(root).resolve() if root else None
        self.cache: dict[str, str] = {}
        if self.path and self.path.is_file():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    self.cache[record["path"]] = record["text"]

    def key_for(self, wav_path: str | Path) -> str:
        """The cache key of an audio path: split-relative when that is possible."""
        path = Path(wav_path)
        if self.root is not None:
            try:
                return path.resolve().relative_to(self.root).as_posix()
            except ValueError:
                pass
        return path.as_posix()

    def get(self, wav_path: str | Path) -> str:
        key = self.key_for(wav_path)
        if key in self.cache:
            return self.cache[key]
        if self.recognizer is None:
            raise RuntimeError(
                "no recognizer configured: the audio transcript is required for inference"
            )
        text = self.recognizer.transcribe(wav_path)
        self.cache[key] = text
        self._flush()
        return text

    def _flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            for key, text in self.cache.items():
                fh.write(json.dumps({"path": key, "text": text}, ensure_ascii=False) + "\n")

    def flush(self) -> None:
        self._flush()


def audio_path(sample: dict, split_path) -> Path | None:
    """Resolve a sample's final-request audio file."""
    relative = sample.get("audio")
    if not relative:
        return None
    candidate = Path(split_path) / relative
    if candidate.is_file():
        return candidate
    # some samples store the audio directly as audio/<id>.wav
    flat = Path(split_path) / "audio" / f"{sample['id']}.wav"
    if flat.is_file():
        return flat
    return None
