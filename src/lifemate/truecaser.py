"""A small pretrained truecaser, used as a hint in the prompt.

The pipeline's transcript is lowercase (Decision 24): the policy model casts the
arguments itself, because it is the only component that knows what the user actually
said. But a model can still get a name wrong — it has never heard "Mai Anh" before,
and the train split's contact names are largely disjoint from the public test's.

This module is the cheap safety net for exactly that case. It is a **pretrained
Vietnamese masked language model** (``vinai/phobert-base`` by default) scoring the
candidate spellings of every word in the utterance, and it produces a *hint*: a
recased transcript that is shown to the policy model next to the lowercase one. The
hint is advisory — the policy still decides, and it can override it.

Why an MLM rather than a dedicated truecasing checkpoint: there is no maintained
pretrained Vietnamese truecasing model, and the PhoBERT class is what the Vietnamese
speech community has. The scoring is the standard pseudo-log-likelihood: for every
word, each candidate spelling is substituted into the sentence and the model's
log-probability of the substituted tokens is summed. The best-scoring spelling wins.

Cost: one forward pass per (word, candidate). A 30-word utterance with three
candidates per word is 90 short forwards, batched — well under a second on the
competition's A30, and the model is 135M parameters, so the system total stays at
4.87B against the 6B cap.

The module is optional everywhere it is used: if ``transformers`` is missing, or the
weights are not in the image, :func:`load_truecaser` returns ``None`` and the pipeline
runs without the hint.
"""
from __future__ import annotations

import re
from typing import Protocol

DEFAULT_MODEL_ID = "vinai/phobert-base"

_ALPHA = re.compile(r"[^\W\d_]", re.UNICODE)


class Truecaser(Protocol):
    """Anything that can propose a spelling for a piece of text."""

    def case(self, text: str) -> str:
        ...


def _forms(word: str) -> list[str]:
    """The candidate spellings of one word.

    Lowercase, Titlecase and UPPERCASE, plus the original spelling when it already
    carries capitals (an acronym like "VOV3" must not be flattened).
    """
    if not _ALPHA.search(word):
        return [word]

    candidates: list[str] = []

    def add(value: str) -> None:
        if value and value not in candidates:
            candidates.append(value)

    add(word.lower())

    # Titlecase: the first letter, the rest lower.
    for index, char in enumerate(word):
        if _ALPHA.match(char):
            add(word[:index] + char.upper() + word[index + 1:].lower())
            break

    add(word.upper())

    if any(char.isupper() for char in word):
        add(word)

    return candidates


class PhobertTruecaser:
    """A truecaser built on a pretrained Vietnamese masked language model."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str | None = None,
                 batch_size: int = 16, max_length: int = 512):
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForMaskedLM.from_pretrained(model_id)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.batch_size = batch_size
        self.max_length = max_length
        self._cache: dict[str, str] = {}

    # -- scoring ------------------------------------------------------------------------
    def case(self, text: str) -> str:
        """Return the recased version of ``text``.

        The output is whitespace-normalised (single spaces), which is what the
        recogniser emits anyway.
        """
        if not text or not text.strip():
            return text
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        words = text.split()
        # The candidate spellings per word.
        options = [_forms(word) for word in words]
        if all(len(candidates) == 1 for candidates in options):
            result = " ".join(words)
            self._cache[text] = result
            return result

        scores = self._score(words, options)

        out = []
        for index, word in enumerate(words):
            candidates = options[index]
            if len(candidates) == 1:
                out.append(candidates[0])
                continue
            best = max(candidates, key=lambda cand: scores.get((index, cand), -1e9))
            out.append(best)
        result = " ".join(out)
        self._cache[text] = result
        return result

    def _score(self, words: list[str], options: list[list[str]]) -> dict:
        """The log-probability of every (word, candidate) pair.

        Each candidate is substituted into the sentence and the model is asked how
        likely the substituted tokens are. Batched, because one forward pass per pair
        would be slow for a 30-word utterance.
        """
        import torch

        jobs: list[tuple[int, str, str, int, int]] = []
        for index, candidates in enumerate(options):
            for candidate in candidates:
                prefix = " ".join(words[:index])
                offset = len(prefix) + (1 if prefix else 0)
                sentence = " ".join(words[:index] + [candidate] + words[index + 1:])
                jobs.append((index, candidate, sentence, offset, offset + len(candidate)))

        scores: dict = {}
        for start in range(0, len(jobs), self.batch_size):
            batch = jobs[start:start + self.batch_size]
            encoded = self.tokenizer(
                [job[2] for job in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.no_grad():
                logits = self.model(**encoded).logits
            logprobs = torch.log_softmax(logits, dim=-1)

            for row, job in enumerate(batch):
                index, candidate, _sentence, begin, end = job
                total = 0.0
                token_ids = encoded["input_ids"][row]
                span = offsets[row].tolist()
                for position, (token_start, token_end) in enumerate(span):
                    if token_start == token_end:  # a special token
                        continue
                    if token_end <= begin or token_start >= end:
                        continue
                    total += float(logprobs[row, position, token_ids[position]])
                scores[(index, candidate)] = total
        return scores


class NullTruecaser:
    """A truecaser that changes nothing — used when the hint is switched off."""

    def case(self, text: str) -> str:
        return text


_LOADED: dict[str, object] = {}


def load_truecaser(model_id: str | None = None, device: str | None = None) -> Truecaser | None:
    """Load the truecaser, or None when it is not available.

    The weights are optional: the competition image has no network access, so they
    have to be baked in. When they are missing the pipeline runs without the hint
    rather than failing.
    """
    target = model_id or DEFAULT_MODEL_ID
    if target in _LOADED:
        return _LOADED[target]
    try:
        instance = PhobertTruecaser(target, device=device)
    except Exception:  # no transformers, no weights, no network — the hint is optional
        _LOADED[target] = None
        return None
    _LOADED[target] = instance
    return instance
