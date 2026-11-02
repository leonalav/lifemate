"""Should the render path use ``is_acronym``?

``is_acronym`` recognises "VOV3" and "E5" as product codes, but the render path only
consults the *derived* acronym set, which holds the codes the training transcripts
spelled consistently. The result is that "VOV3" is lowercased to "vov3" even though the
gold keeps it in caps.

This measures the change before it is made: the render path is asked to keep the
original uppercase form of a token that ``is_acronym`` accepts, and the transcripts are
compared against the gold token by token.

The risk is real — ``is_acronym`` also accepts any unknown short uppercase word, so
"MAI" or "BAN" could be mistaken for a code. The measurement reports both the gain and
the damage so the decision is made on numbers.

Run:  python research/analyze_acronym_render.py --data-dir data/train+public_test
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lifemate.capitalize import (  # noqa: E402
    _SENTENCE_END,
    _THOUSANDS,
    HONORIFICS,
    _vocabulary,
    is_acronym,
    restore_capitals,
)
from lifemate.data import resolve_split  # noqa: E402

TOKEN = re.compile(r"\S+")


def restore_with_acronyms(text: str) -> str:
    """``restore_capitals``, but a token ``is_acronym`` accepts keeps its case.

    The render path is re-implemented here rather than patched into the module, so the
    measurement can compare the two without touching the shipped code.
    """
    if not text:
        return text
    proper, acronyms, _ordinary = _vocabulary()

    out: list[str] = []
    capitalise_next = True
    previous = ""

    for part in _SENTENCE_END.split(text):
        if part in (".", "!", "?", "\n"):
            out.append(part)
            capitalise_next = True
            previous = ""
            continue
        for chunk in _THOUSANDS.split(part):
            if _THOUSANDS.fullmatch(chunk):
                out.append(chunk)
                continue
            rendered, last = _render(chunk, capitalise_next, previous, proper,
                                     acronyms)
            out.append(rendered)
            capitalise_next = False
            previous = last
    return "".join(out)


def _render(chunk: str, capitalise_next: bool, previous: str,
            proper: dict[str, str], acronyms: frozenset[str]) -> tuple[str, str]:
    segments = re.findall(r"\S+|\s+", chunk)
    word_indices = [i for i, s in enumerate(segments) if s.strip()]
    bare_words = {}
    for i in word_indices:
        bare = segments[i].strip(",.;:!?\"'()")
        if bare:
            bare_words[i] = bare

    name_positions = set()
    honorific_positions = set()
    for position, index in enumerate(word_indices):
        bare = bare_words.get(index)
        if bare is None:
            continue
        if bare.lower() in HONORIFICS and position + 1 < len(word_indices):
            name_positions.add(word_indices[position + 1])
            honorific_positions.add(index)

    rendered_segments = []
    last_word = previous
    for i, segment in enumerate(segments):
        if not segment.strip():
            rendered_segments.append(segment)
            continue
        bare = bare_words.get(i, segment)
        leading = segment[: segment.index(bare)]
        trailing = segment[len(leading) + len(bare):]
        lowered = bare.lower()

        if capitalise_next:
            text = bare[0].upper() + bare[1:].lower()
            capitalise_next = False
        elif i in honorific_positions:
            text = bare[0].upper() + bare[1:].lower()
        elif i in name_positions:
            text = bare[0].upper() + bare[1:].lower()
        elif lowered in proper:
            text = proper[lowered]
        elif lowered in acronyms or is_acronym(bare):
            text = bare
        else:
            text = bare.lower()

        rendered_segments.append(leading + text + trailing)
        last_word = bare.lower().strip(",.;:!?")

    return "".join(rendered_segments), last_word


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/train+public_test")
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    split = resolve_split(args.data_dir, args.split)

    stats = collections.Counter()
    fixed = collections.Counter()
    broken = collections.Counter()
    examples_fixed = []
    examples_broken = []

    for sample in split.iter_samples():
        text = sample.get("audio_text") or ""
        if not text:
            continue
        upper = text.upper()
        current = restore_capitals(upper)
        proposed = restore_with_acronyms(upper)

        gold_tokens = TOKEN.findall(text)
        current_tokens = TOKEN.findall(current)
        proposed_tokens = TOKEN.findall(proposed)
        if not (len(gold_tokens) == len(current_tokens) == len(proposed_tokens)):
            continue

        for gold, now, then in zip(gold_tokens, current_tokens, proposed_tokens):
            stats["tokens"] += 1
            if now == gold:
                stats["current_ok"] += 1
            if then == gold:
                stats["proposed_ok"] += 1
            if now != gold and then == gold:
                stats["fixed"] += 1
                fixed[(gold, now)] += 1
                if len(examples_fixed) < 25:
                    examples_fixed.append({"gold": gold, "current": now,
                                           "proposed": then})
            elif now == gold and then != gold:
                stats["broken"] += 1
                broken[(gold, then)] += 1
                if len(examples_broken) < 25:
                    examples_broken.append({"gold": gold, "current": now,
                                            "proposed": then})

    total = stats["tokens"] or 1
    report = {
        "tokens": stats["tokens"],
        "current_rule_module_accuracy": round(
            100.0 * stats["current_ok"] / total, 3),
        "proposed_accuracy": round(100.0 * stats["proposed_ok"] / total, 3),
        "fixed": stats["fixed"],
        "broken": stats["broken"],
        "net": stats["fixed"] - stats["broken"],
        "most_common_fixes": fixed.most_common(15),
        "most_common_breakages": broken.most_common(15),
        "examples_fixed": examples_fixed,
        "examples_broken": examples_broken,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
