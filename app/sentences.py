"""Split a transcript into sentences so the earbuds can start speaking sooner.

Why this exists
---------------
MEASURED in the existing app: waiting for the whole utterance to be translated
before any audio plays costs ~595 ms before the first audible word. Translating
sentence 1 and sending it immediately lets text-to-speech start while sentences
2..n are still being translated, so the *perceived* latency drops to roughly the
first sentence's share of the work.

What this does NOT do
---------------------
It does not make the server faster. Total server time for a multi-sentence
utterance is the same or very slightly worse (more, smaller MT calls have more
per-call overhead than one large call). The gain is entirely in *time to first
audible word*, which is the number a person actually experiences. Both are
reported separately -- `first_sentence_ms` and `total_server_ms` -- because
collapsing them into one figure would be marketing, not measurement.

For a single-sentence utterance there is nothing to overlap, so splitting is
skipped entirely and the fast path is unchanged.

Why not a sentence-splitting library
------------------------------------
`nltk.punkt`, `pysbd`, `spacy` etc. all add a dependency, a model download, or
both, and none of them handle the language range here (100 ASR languages, incl.
Chinese/Japanese/Thai full stops and Arabic punctuation) better than the
terminator set below for *transcribed speech*, which has simpler structure than
written prose: no footnotes, no citations, few abbreviations, and Whisper
normalises most punctuation already.

The known weakness is abbreviations ("Dr. Smith", "U.S.A."), so the abbreviation
guard below is explicit and tested rather than hoped for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

# Sentence terminators across the scripts this server serves. Includes the
# CJK ideographic full stop, Arabic question mark, Devanagari danda, Ethiopic
# full stop, and Armenian/Greek marks -- a Latin-only regex would silently
# refuse to split Chinese, Hindi or Arabic and hand those languages the slow
# path while pretending to support them.
_TERMINATORS: Final[str] = (
    "."      # ASCII full stop
    "!?"     # ASCII exclamation / question
    "\u3002"  # 。 ideographic full stop (zh, ja)
    "\uff01"  # ！ fullwidth exclamation
    "\uff1f"  # ？ fullwidth question
    "\u061f"  # ؟ Arabic question mark
    "\u06d4"  # ۔ Urdu full stop
    "\u0964"  # । Devanagari danda
    "\u0965"  # ॥ Devanagari double danda
    "\u1362"  # ። Ethiopic full stop
    "\u0589"  # ։ Armenian full stop
    "\u037e"  # ; Greek question mark
    "\u2026"  # … ellipsis
)

# Closing quotes/brackets that may follow a terminator and still belong to the
# same sentence: `He said "stop!" and left.` must not split inside the quote.
_TRAILING: Final[str] = "\"'\u201d\u2019\u00bb\u3001\uff09)]}\u300d\u300f"

# Terminators that are followed by whitespace in the scripts that use spaces.
_SPACED_SPLIT: Final[re.Pattern[str]] = re.compile(
    rf"(?<=[{re.escape(_TERMINATORS)}])[{re.escape(_TRAILING)}]*\s+"
)

# CJK terminators need their own rule, found by TEST not by design: the first
# version of this module required `\s+` after the terminator, and Chinese and
# Japanese do not put a space after 。 -- so "这是第一句话。这是第二句话。"
# came back as ONE piece. The claim to support those languages was hollow: the
# code accepted them and quietly handed them the slow path.
_CJK_TERMINATORS: Final[str] = "\u3002\uff01\uff1f"
_CJK_SPLIT: Final[re.Pattern[str]] = re.compile(
    rf"(?<=[{re.escape(_CJK_TERMINATORS)}])[{re.escape(_TRAILING)}]*"
)

# An opening quote/bracket that is still unclosed means a terminator inside it
# is quoted speech, not a sentence end: `He said "stop!" and left.` must stay
# one sentence. Also found by test: the naive rule split it after `stop!`.
_OPENERS: Final[str] = "\"'\u201c\u2018\u00ab([{\u300c\u300e"
_CLOSERS: Final[str] = "\"'\u201d\u2019\u00bb)]}\u300d\u300f"
# Straight quotes are both opener and closer, so they are counted for parity
# rather than matched as a pair.
_AMBIGUOUS: Final[str] = "\"'"

# Tokens where a trailing period is NOT a sentence end. Lower-cased, without
# the dot. Kept short on purpose: every entry here is a case where splitting
# would be wrong, and a long speculative list would start creating the
# opposite error (refusing to split a real sentence end).
_ABBREVIATIONS: Final[frozenset[str]] = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "rev", "hon",
        "vs", "etc", "eg", "ie", "approx", "dept", "est", "min", "max",
        "no", "vol", "fig", "al", "inc", "ltd", "co", "corp",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec", "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        "am", "pm", "a.m", "p.m",
    }
)

_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\w", re.UNICODE)
_LAST_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"([\w.]+)\.\s*$", re.UNICODE)
# A single capital letter before the dot: "J. R. Tolkien", "U.S.A."
_INITIAL_RE: Final[re.Pattern[str]] = re.compile(r"(?:^|\s)(?:\w\.){1,}\w?\.?\s*$", re.UNICODE)


@dataclass(frozen=True)
class Split:
    """The result of splitting, with the reason it turned out this way.

    `reason` exists so a caller (and /metrics) can tell "one sentence" from
    "splitting was disabled" from "the pieces were too small to be worth
    separate MT calls" -- three very different situations that would otherwise
    all look like a single-element list.
    """

    sentences: tuple[str, ...]
    reason: str

    @property
    def is_split(self) -> bool:
        return len(self.sentences) > 1


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x3040 <= o <= 0x30FF      # hiragana + katakana
        or 0x3400 <= o <= 0x4DBF   # CJK ext A
        or 0x4E00 <= o <= 0x9FFF   # CJK unified
        or 0xF900 <= o <= 0xFAFF   # CJK compatibility
        or 0xAC00 <= o <= 0xD7AF   # hangul syllables
    )


def _length_units(text: str) -> int:
    """Length in rough "word-equivalents", so min_chars means the same thing
    across scripts.

    A raw len() is script-biased: "这是第一句话。" is a complete sentence in 7
    characters and would be judged a fragment under min_chars=12, then merged
    into its neighbour -- silently disabling splitting for exactly the
    languages the CJK rule above was added to support. CJK characters are
    therefore weighted ~2.5x, which is roughly the character-per-word ratio
    difference between CJK and Latin script.
    """
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return int(other + cjk * 2.5)


def _join_sep(left: str, right: str) -> str:
    """Space between spaced scripts, nothing between CJK.

    Injecting a space between two Chinese sentences would put a character in
    the transcript that the speaker never said, and the round-trip test asserts
    that never happens.
    """
    if not left or not right:
        return ""
    if _is_cjk(left[-1]) or left[-1] in _CJK_TERMINATORS:
        return ""
    return " "


def _inside_quotes(chunk: str) -> bool:
    """True if `chunk` has an unclosed quote or bracket.

    Used to decide that a terminator was inside quoted speech. Straight quotes
    are counted for odd parity because `"` is both opener and closer; directional
    quotes and brackets are matched as pairs.
    """
    depth = 0
    straight = 0
    for ch in chunk:
        if ch in _AMBIGUOUS:
            straight += 1
        elif ch in _OPENERS:
            depth += 1
        elif ch in _CLOSERS:
            depth = max(0, depth - 1)
    return depth > 0 or straight % 2 == 1


def _ends_with_abbreviation(chunk: str) -> bool:
    """True if the chunk's final period is part of an abbreviation."""
    if _INITIAL_RE.search(chunk):
        return True
    match = _LAST_TOKEN_RE.search(chunk)
    if not match:
        return False
    return match.group(1).lower().strip(".") in _ABBREVIATIONS


def split_sentences(
    text: str,
    *,
    min_chars: int = 12,
    max_sentences: int = 8,
    enabled: bool = True,
) -> Split:
    """Split `text` into sentences suitable for independent translation.

    min_chars
        A fragment shorter than this is merged into its neighbour. Sending
        "Yes." as its own MT call spends a whole round trip of overhead on two
        syllables, and a translation model given a two-word fragment with no
        context is also more likely to mistranslate it.
    max_sentences
        Cap on pieces. Beyond this the tail is merged, because each piece is a
        separate MT call and past a certain count the per-call overhead
        outweighs the overlap gain.
    """
    raw = (text or "").strip()
    if not enabled:
        return Split((raw,) if raw else (), "splitting disabled")
    if not raw:
        return Split((), "empty text")
    if not _WORD_RE.search(raw):
        # Consistent with mt.is_translatable: no word characters means there is
        # nothing to translate, and an engine given this returns a
        # hallucination rather than a blank (measured: '  ' -> 'مـنـهـا').
        return Split((), "no word characters")

    # Two passes: space-delimited scripts first, then CJK inside each piece,
    # because a mixed-script utterance ("Hello there. 这是第二句话。") needs both.
    pieces: list[str] = []
    for spaced in _SPACED_SPLIT.split(raw):
        for cjk in _CJK_SPLIT.split(spaced):
            if cjk.strip():
                pieces.append(cjk.strip())
    if len(pieces) <= 1:
        return Split((raw,), "single sentence: nothing to overlap")

    # Re-join pieces whose split point was not a real sentence end: an
    # abbreviation, or a terminator inside unclosed quotes. The separator is a
    # space for spaced scripts and nothing for CJK, so that rejoining a CJK
    # split does not inject a space that was never in the transcript.
    merged: list[str] = []
    rejoined: list[str] = []  # why each rejoin happened, for the reason field
    for piece in pieces:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]}{_join_sep(merged[-1], piece)}{piece}"
            rejoined.append("abbreviation")
        elif merged and _inside_quotes(merged[-1]):
            merged[-1] = f"{merged[-1]}{_join_sep(merged[-1], piece)}{piece}"
            rejoined.append("quoted speech")
        else:
            merged.append(piece)

    # Merge fragments below min_chars into the previous piece (or the next one,
    # if the short fragment is first). Length is measured in *units*, not raw
    # characters: see _length_units -- a raw len() would judge a complete
    # 7-character Chinese sentence to be a fragment and merge it away.
    def _short(piece: str) -> bool:
        return _length_units(piece) < min_chars

    packed: list[str] = []
    for piece in merged:
        if packed and _short(piece):
            packed[-1] = f"{packed[-1]}{_join_sep(packed[-1], piece)}{piece}"
        else:
            packed.append(piece)
    # A short leading fragment is now packed[0]; fold it forward once.
    if len(packed) > 1 and _short(packed[0]):
        packed[1] = f"{packed[0]}{_join_sep(packed[0], packed[1])}{packed[1]}"
        packed.pop(0)

    if len(packed) > max_sentences:
        head = packed[: max_sentences - 1]
        tail = packed[max_sentences - 1]
        for extra in packed[max_sentences:]:
            tail = f"{tail}{_join_sep(tail, extra)}{extra}"
        packed = head + [tail]
        reason = f"split into {len(packed)} pieces (capped at {max_sentences})"
    elif len(packed) <= 1:
        # Say WHICH rule collapsed it. The first version of this branch always
        # blamed min_chars, so an utterance kept whole because of an
        # abbreviation or a quotation reported a reason that was not true.
        if rejoined:
            why = ", ".join(sorted(set(rejoined)))
            return Split(
                (raw,),
                f"kept whole: the apparent sentence breaks were {why}, not real "
                f"sentence ends",
            )
        return Split(
            (raw,),
            "kept whole: the pieces were shorter than the minimum, so splitting "
            "would cost more in per-call overhead than it saves",
        )
    else:
        reason = f"split into {len(packed)} sentences"
        if rejoined:
            reason += f" (rejoined at: {', '.join(sorted(set(rejoined)))})"

    return Split(tuple(packed), reason)
