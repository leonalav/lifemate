"""Restoring the capitalisation an uppercase ASR transcript lost.

gipformer emits uppercase text with no punctuation, so the transcript that reaches the
model loses every capital letter — including the ones that decide whether an argument
matches. Measured on the train split: **3,164 of the 3,904** string arguments in the gold
trajectories are compared with exact equality, and **362 samples** contain one that
carries capitals, for example ``contacts.search.query = 'Sếp Dũng'`` or
``news.search.query = 'VOV3'``.

The reconstruction is measured, not guessed. Round-tripping the 1,545 gold transcripts
(uppercase them, restore the capitals, compare with the original):

    lowercase only          0.9729 similarity    1/1545 exact
    sentence case           0.9866 similarity  663/1545 exact
    this module            0.9927 similarity  959/1545 exact

And on the metric that decides the exactly-compared arguments — can the model copy a gold
query value verbatim out of the transcript it is shown — the improvement is decisive:

    lowercase only          26/300 query arguments reproduced   ( 8.7%)
    sentence case           28/300                            ( 9.3%)
    this module            176/300                            (58.7%)

The context model in ``case_model.py`` takes that last figure further, to 238/300
(79.3%): the tokens this module leaves alone are ambiguous ones, and their case is a
question about the neighbouring words. See ``docs/DECISIONS.md``, Decision 23.

The vocabulary is **derived from the training transcripts**, not hand-written: the public
test has no gold transcripts, so the resource has to come from train. It was built by
``research/build_capitalization.py`` and holds 324 proper words and 15 acronyms.

The rules, in the order they earned their place:

* a word in the derived vocabulary keeps its recorded spelling — "Zalo", "Sếp", "Huế",
  "VOV3", "SJC"
* a personal name right after a Vietnamese honorific is capitalised — "cô Mai", "bác Ba"
  (this is what recovers the names that also occur in lowercase elsewhere, like "mai"
  the month and "ba" the father)
* a product code whose letters the derived acronym set recognises keeps its caps —
  "VOV3", "VTV1", "RON95", "RV183", "E5"
* the first word of a sentence is capitalised
* everything else is lowercased

The rules cannot settle an ambiguous token on their own, so the context model in
``case_model.py`` decides the remaining cases; it is layered on top of this module and
leaves the decisions below to it. See ``docs/DECISIONS.md``, Decision 9.

**This module is not in the runtime path any more.** The transcript reaches the policy
model in lowercase and the model casts the arguments itself (Decision 24); this code is
kept as the ablation baseline, selectable with ``--case-mode rules``. Every number
above was measured with it in the path.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

RESOURCE = Path(__file__).resolve().parent / "resources" / "capitalization.json"

# Vietnamese honorifics that precede a personal name.
#
# Only the ones that reliably introduce a name are listed. "anh", "chị", "em" and "tôi"
# were measured and removed: they are ordinary words far more often than name markers,
# and capitalising the word after them damaged more transcripts than it repaired
# (similarity 0.9683 -> 0.9755 without them).
HONORIFICS = {
    "sếp", "sep", "cô", "co", "chú", "chu", "bác", "bac", "thầy", "thay",
    "bố", "bo", "mẹ", "me", "má", "ma", "ông", "ong", "bà", "cụ", "cu",
    "cô giáo", "chú bác", "anh trai", "chị em", "bác ba", "cô mai", "chị hai",
}

_SENTENCE_END = re.compile(r"([.!?\n])")
_THOUSANDS = re.compile(r"(\d\.\d)")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)

# Ordinary Vietnamese words that a short uppercase token could be mistaken for. The
# derived vocabulary only knows the words that appear in the training transcripts; a
# word that never appears in train ("quán") cannot be derived, so the common ones are
# listed here. The list only ever *prevents* a capitalisation, so a false entry costs
# a lower-case letter and never changes the meaning.
ORDINARY_WORDS = frozenset({
    "anh", "bà", "bác", "bạn", "bày", "bé", "bên", "bếp", "bố", "buổi", "bữa",
    "cà", "cái", "cạnh", "cần", "cả", "các", "căn", "càng", "câu", "cho", "chỗ",
    "chú", "cô", "có", "con", "còn", "cùng", "cũng", "cuối", "cụ", "cứ",
    "đa", "đang", "đây", "để", "đến", "đi", "đó", "được", "đúng",
    "gà", "gì", "giờ", "giữ", "giữa", "gần", "hay", "hơn", "hoa", "học", "hôm",
    "hứa", "hẹn", "hơi",
    "khi", "không", "khách", "khó", "kiểu", "kỹ",
    "là", "lại", "làm", "lên", "lúc", "lưu", "lượt",
    "mà", "mình", "mọi", "mới", "mùa", "mở", "mỗi", "một",
    "này", "nên", "nói", "nhé", "như", "nhớ", "nhà", "ngoài", "ngay", "nghe",
    "nơi", "nữa",
    "pha", "phải", "phía",
    "qua", "quán", "quay", "quên",
    "rất", "ra", "rồi", "rơi", "ruồi",
    "sau", "sáng", "sinh", "so", "sớm", "sự",
    "ta", "tại", "tháng", "thay", "theo", "thì", "thôi", "thứ", "thu", "tối",
    "trên", "trước", "trong", "trắc", "tuần", "từ", "tự",
    "và", "vào", "vẫn", "vì", "việc", "với", "về", "vừa",
    "xa", "xoay", "xong",
    "ý", "yêu", "yếu",
})


@lru_cache(maxsize=1)
def _vocabulary() -> tuple[dict[str, str], frozenset[str], frozenset[str]]:
    """The derived proper-word, acronym and ordinary-word sets, loaded once."""
    if not RESOURCE.is_file():
        return {}, frozenset(), frozenset()
    with RESOURCE.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    proper = {k.lower(): v for k, v in (payload.get("proper") or {}).items()}
    acronyms = {k.lower(): v for k, v in (payload.get("acronyms") or {}).items()}
    ordinary = {k.lower() for k in (payload.get("ordinary") or [])}
    return proper, frozenset(acronyms), frozenset(ordinary)


def is_acronym(word: str) -> bool:
    """True for an acronym or product code such as "SJC", "VOV3", "E5".

    The derived vocabulary decides the known cases. For an unknown short uppercase word
    the answer depends on whether the word is an ordinary Vietnamese word, so the
    ordinary-word list is consulted: "CHO" and "QUÁN" are words, "SJC" is not.
    """
    proper, acronyms, ordinary = _vocabulary()
    lowered = word.lower()
    if lowered in acronyms:
        return True
    if lowered in proper:
        return False
    if lowered in ordinary:
        return False
    if lowered in ORDINARY_WORDS:
        return False
    if not (2 <= len(word) <= 4):
        return False
    if not word.isupper():
        return False
    # an unknown short uppercase word is more likely an acronym than a noun
    return True


def is_code(word: str) -> bool:
    """True for a product code whose letters the derived acronym set recognises.

    "VOV3", "VTV1", "RON95" and "RV183" are codes: the training transcripts spell them
    in caps, and the transcripts keep them in caps. The derived acronym set holds their
    alphabetic part ("vov", "vtv", "ron", "rv"), so the digits are matched against it.

    This is deliberately narrower than :func:`is_acronym`. That function accepts any
    unknown short uppercase word, which is right for deciding whether a token *looks*
    like a code but wrong for deciding its spelling: it would turn "giúp" into "GIÚP".
    Measured against the training transcripts, using it in the render path broke 6,974
    tokens and fixed 13; this test breaks none and fixes eleven.
    """
    proper, acronyms, _ordinary = _vocabulary()
    if not any(char.isdigit() for char in word):
        return False
    alpha = "".join(char for char in word if char.isalpha()).lower()
    if not alpha or alpha not in acronyms:
        return False
    return alpha not in proper


def restore_capitals(text: str, honorifics: set[str] | None = None,
                     capitalise_honorific: bool = True) -> str:
    """Restore the capitalisation of an uppercase transcript.

    ``text`` is expected to be uppercase; the result is the best reconstruction of the
    original casing that can be recovered without a language model.
    """
    if not text:
        return text
    honorifics = HONORIFICS if honorifics is None else honorifics
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
            rendered, last = _render_chunk(
                chunk, capitalise_next, honorifics, previous, proper, acronyms,
                capitalise_honorific,
            )
            out.append(rendered)
            capitalise_next = False
            previous = last

    return "".join(out)


def _render_chunk(chunk: str, capitalise_next: bool, honorifics: set[str],
                  previous: str, proper: dict[str, str],
                  acronyms: frozenset[str],
                  capitalise_honorific: bool = False) -> tuple[str, str]:
    """Render one chunk, returning the text and the last plain word seen."""
    if not chunk:
        return "", previous

    # Split into (token, whitespace-after) pairs, so the original spacing survives.
    segments = re.findall(r"\S+|\s+", chunk)

    word_indices = [i for i, segment in enumerate(segments) if segment.strip()]
    bare_words = {}
    for i in word_indices:
        token = segments[i]
        bare = token.strip(",.;:!?\"'()")
        if bare:
            bare_words[i] = bare

    # A name follows an honorific: "cô Mai", "bác Ba", "Sếp Dũng". The honorific is
    # looked at from the left, so the name after it is capitalised.
    name_positions = set()
    honorific_positions = set()
    for position, index in enumerate(word_indices):
        bare = bare_words.get(index)
        if bare is None:
            continue
        if bare.lower() in honorifics and position + 1 < len(word_indices):
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
        elif i in honorific_positions and capitalise_honorific:
            # the honorific of a name phrase is capitalised too: "Sếp Dũng", "Bác Ba"
            text = bare[0].upper() + bare[1:].lower()
        elif i in name_positions:
            text = bare[0].upper() + bare[1:].lower()
        elif lowered in proper:
            text = proper[lowered]
        elif lowered in acronyms:
            text = bare
        elif is_code(bare):
            # a product code keeps its caps: "VOV3", "RON95", "RV183"
            text = bare
        else:
            text = bare.lower()

        rendered_segments.append(leading + text + trailing)
        last_word = bare.lower().strip(",.;:!?")

    return "".join(rendered_segments), last_word
