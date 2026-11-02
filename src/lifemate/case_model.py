"""A context-aware capitalisation model.

**Why this exists.** The rule module (``capitalize.py``) restores the case of an
uppercase ASR transcript with a fixed rule set. It is right about 59.0% of the
exactly-compared capitalised arguments in the training split. The remaining misses are
overwhelmingly *ambiguous* tokens — a word the training transcripts spell in both cases,
so its case is a genuine context decision ("Minh" the contact against "minh" the
pronoun). A vocabulary cannot settle that; the neighbouring words can.

The measurements behind that claim are in ``research/analyze_case_*.py`` and are
summarised in ``docs/DECISIONS.md`` (Decision 9). Every number is a 5-fold
cross-validated score over the whole training split, measured through the same code
path the pipeline uses:

    ceiling                   perfect capitalisation is worth 9.258 points over
                              lowercasing every capitalised argument
    rule module alone         97.071 points
    this model                98.626 points — +1.555 points over the rule module
    argument recovery         59.0% -> 81.0% of the exactly-compared arguments
    without the token's own identity    +0.789 points, so the gain is contextual
    on words never seen capitalised in training, still 4.3% -> 13.8%

**The model.** A logistic regression over the token and its context: the two words
either side, the rule module's own opinion, position in the sentence, and membership of
the derived vocabulary. It decides one thing per token — whether that token's first
alphabetic character is uppercase. The rule module keeps everything else about the
token (internal capitals, digits, punctuation), so "VOV3" and "S+ Sport" survive.

**The division of labour.** Two kinds of token are decided *structurally* by the rules
rather than by context, and the model keeps away from both: honorifics (the rules
capitalise "Mẹ" only when a name follows it) and product codes ("VTV6"). Deferring on
both cut the arguments the model breaks from 17 to 1 and raised the gain from 1.172 to
1.555 points.

**The artifact.** The coefficients are exported to ``resources/case_model.json`` as a
flat ``{feature: weight}`` map and scored with a dictionary lookup. No scikit-learn at
inference time: the runtime dependencies do not include it, and the model is a linear
one, so the dependency would buy nothing. The training script is
``research/build_case_model.py``. Weights below a magnitude of 0.1 are dropped, which
leaves the score unchanged and the artifact at 138 KiB rather than 428 KiB.

**Fallback.** If the artifact is missing, :func:`transcript` returns the rule module's
restoration, so the pipeline still runs — it simply scores 97.071 instead of 98.626.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .capitalize import HONORIFICS, _vocabulary, is_code, restore_capitals

RESOURCE = Path(__file__).resolve().parent / "resources" / "case_model.json"


def _bare(token: str) -> str:
    """The token without its leading and trailing punctuation."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", token, flags=re.UNICODE)


def _first_alpha(token: str) -> int:
    for position, char in enumerate(token):
        if char.isalpha():
            return position
    return -1


def _rule_says_capital(rule_tokens: list[str], index: int) -> bool:
    """Whether the rule module capitalised this token's first character."""
    if index >= len(rule_tokens):
        return False
    position = _first_alpha(rule_tokens[index])
    return position >= 0 and rule_tokens[index][position].isupper()


def featurize(tokens: list[str], rule_tokens: list[str], index: int) -> dict:
    """The features of one token.

    ``tokens`` is the uppercase text (what the ASR delivers) and ``rule_tokens`` is the
    rule module's restoration of it. The features describe the context, so the model
    can learn that a token after "gọi cho" is a name even when the word itself is
    ambiguous.
    """
    proper, acronyms, ordinary = _vocabulary()
    word = _bare(tokens[index]).lower()
    prev1 = _bare(tokens[index - 1]).lower() if index > 0 else ""
    prev2 = _bare(tokens[index - 2]).lower() if index > 1 else ""
    next1 = _bare(tokens[index + 1]).lower() if index + 1 < len(tokens) else ""
    next2 = _bare(tokens[index + 2]).lower() if index + 2 < len(tokens) else ""

    return {
        "bias": 1.0,
        "word": word,
        "prev1": prev1,
        "prev2": prev2,
        "next1": next1,
        "next2": next2,
        "prev1_is_honorific": float(prev1 in HONORIFICS),
        "next1_is_honorific": float(next1 in HONORIFICS),
        "in_proper_vocab": float(word in proper),
        "in_acronym_vocab": float(word in acronyms),
        "in_ordinary_vocab": float(word in ordinary),
        "is_first": float(index == 0),
        "is_last": float(index == len(tokens) - 1),
        "word_len": min(len(word), 12) / 12.0,
        "has_digit": float(any(c.isdigit() for c in word)),
        "rule_says_capital": float(_rule_says_capital(rule_tokens, index)),
        "rule_prev1": _bare(rule_tokens[index - 1]).lower()
        if 0 < index < len(rule_tokens) else "",
        "rule_next1": _bare(rule_tokens[index + 1]).lower()
        if index + 1 < len(rule_tokens) else "",
    }


class CaseModel:
    """The exported linear model, scored with dictionary lookups."""

    def __init__(self, payload: dict):
        self.weights: dict[str, float] = payload.get("weights") or {}
        self.intercept: float = float(payload.get("intercept") or 0.0)

    def score(self, features: dict) -> float:
        """The log-odds that the token's first character is uppercase."""
        total = self.intercept
        for name, value in features.items():
            if isinstance(value, str):
                total += self.weights.get(f"{name}={value}", 0.0)
            else:
                total += self.weights.get(name, 0.0) * value
        return total

    def predicts_capital(self, features: dict) -> bool:
        return self.score(features) > 0.0


_LOADED: dict[str, CaseModel | None] = {}


def load_case_model(path: Path | None = None) -> CaseModel | None:
    """Load the exported model, or None when the artifact is not present.

    The artifact is parsed once per path: the ASR path calls this for every sample,
    and re-reading a 458 KiB JSON each time would dominate the transcription cost.
    """
    target = Path(path) if path else RESOURCE
    key = str(target)
    if key not in _LOADED:
        if not target.is_file():
            _LOADED[key] = None
        else:
            with target.open(encoding="utf-8") as fh:
                _LOADED[key] = CaseModel(json.load(fh))
    return _LOADED[key]


def _defer_to_the_rules(rule_token: str) -> bool:
    """True when the rule module's decision about this token is final.

    Two kinds of token are decided *structurally* by the rules rather than by context,
    and the model has no way to reproduce that reasoning:

    honorifics    the rules capitalise "Mẹ" only when a name follows it ("Mẹ Lan") and
                  leave it alone when it stands alone ("gọi mẹ nhé"). The model sees the
                  word "mẹ" and its neighbours and cannot reliably tell those apart, so
                  it lowercases honorifics the rules got right.
    codes         "VTV6" is a channel code; its shape is not a context question.

    Measured over the training split, deferring on both cut the arguments the model
    breaks from 17 to 1 and raised the gain over the rule module from 1.172 points to
    1.555 (docs/DECISIONS.md, Decision 9).
    """
    bare = "".join(char for char in rule_token if char.isalnum())
    if not bare:
        return False
    lowered = bare.lower()
    return lowered in HONORIFICS or is_code(bare)


def refine_capitals(text: str, model: CaseModel) -> str:
    """Rewrite the case of each token in an uppercase ASR transcript.

    The input is the *uppercase* text the recogniser emits. The output is the
    transcript with its case restored: the rule module decides the shape of each token
    — internal capitals, digits, punctuation — and the model decides whether that
    token's first character is uppercase. The rules keep the final say on honorifics
    and codes, where the decision is structural rather than contextual.
    """
    upper_tokens = text.split()
    if not upper_tokens:
        return text
    rule_tokens = restore_capitals(text).split()
    if len(rule_tokens) != len(upper_tokens):
        return " ".join(rule_tokens)

    out = []
    for index, rule_token in enumerate(rule_tokens):
        position = _first_alpha(rule_token)
        if position < 0:
            out.append(rule_token)
            continue
        if _defer_to_the_rules(rule_token):
            out.append(rule_token)
            continue
        features = featurize(upper_tokens, rule_tokens, index)
        char = rule_token[position]
        if model.predicts_capital(features):
            new_char = char.upper()
        else:
            new_char = char.lower()
        out.append(rule_token[:position] + new_char + rule_token[position + 1:])
    return " ".join(out)


def transcript(text: str, model: CaseModel | None = None) -> str:
    """The transcript the policy model is shown.

    This is the entry point the ASR path uses. When the artifact is absent the rule
    module's restoration is returned, so the pipeline still runs — it simply scores the
    rule module's 96.932 instead of 98.301.
    """
    if model is None:
        model = load_case_model()
    if model is None:
        return restore_capitals(text)
    return refine_capitals(text, model)
