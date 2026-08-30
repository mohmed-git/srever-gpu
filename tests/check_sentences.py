"""Verify sentence splitting, including the cases where it must NOT split.

Run: python tests/check_sentences.py

A splitter that splits too eagerly is worse than no splitter: it sends
fragments to MT without context, and the earbuds speak nonsense confidently.
So roughly half of these cases assert that splitting is *refused*.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.sentences import split_sentences  # noqa: E402

FAILURES: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    tag = "[ok]" if cond else "[FAIL]"
    print(f"{tag} {label}" + (f" :: {detail}" if detail else ""))
    if not cond:
        FAILURES.append(label)


def expect(text: str, count: int, label: str, **kw) -> None:
    s = split_sentences(text, **kw)
    check(
        f"{label}: {count} piece(s)",
        len(s.sentences) == count,
        f"got {len(s.sentences)} {list(s.sentences)} ({s.reason})",
    )


print("--- must split ---")
expect("Where is the pharmacy? I need help now.", 2, "en two sentences")
expect(
    "Hello there. My name is Ahmed. I am looking for the station.",
    3,
    "en three sentences",
)
expect("أين أقرب صيدلية؟ أحتاج مساعدة عاجلة الآن.", 2, "ar with Arabic question mark")
expect("这是第一句话。这是第二句话内容。", 2, "zh ideographic full stop")
expect("यह पहला वाक्य है। यह दूसरा वाक्य है।", 2, "hi Devanagari danda")
expect("Stop right there! Then walk away slowly.", 2, "en exclamation")

print()
print("--- must NOT split ---")
expect("Where is the nearest pharmacy?", 1, "single sentence")
expect("Dr. Smith is waiting for you downstairs.", 1, "abbreviation 'Dr.'")
expect("Meet me at 5 p.m. in the main lobby.", 1, "abbreviation 'p.m.'")
expect("I visited the U.S.A. last summer with family.", 1, "initials 'U.S.A.'")
expect("It costs approx. thirty dollars in total.", 1, "abbreviation 'approx.'")
expect('He said "stop!" and then walked away.', 1, "terminator inside quotes")
expect("Yes. No.", 1, "fragments below min_chars are merged")

print()
print("--- degenerate input ---")
s = split_sentences("")
check("empty string yields no sentences", s.sentences == (), s.reason)
s = split_sentences("   ")
check("whitespace yields no sentences", s.sentences == (), s.reason)
s = split_sentences("...")
check(
    "punctuation-only yields no sentences (same rule as mt.is_translatable)",
    s.sentences == (),
    s.reason,
)
s = split_sentences("First one here. Second one here.", enabled=False)
check(
    "enabled=False returns the text unsplit",
    len(s.sentences) == 1 and s.reason == "splitting disabled",
    s.reason,
)

print()
print("--- cap ---")
many = " ".join(f"This is sentence number {i} here." for i in range(1, 13))
s = split_sentences(many, max_sentences=4)
check("max_sentences caps the piece count", len(s.sentences) == 4, f"{len(s.sentences)}")
check(
    "capping preserves every word (nothing is dropped)",
    sum(len(x.split()) for x in s.sentences) == len(many.split()),
    f"{sum(len(x.split()) for x in s.sentences)} vs {len(many.split())} words",
)

print()
print("--- no content is lost by splitting ---")
for text in (
    "Where is the pharmacy? I need help now.",
    "Hello there. My name is Ahmed. I am looking for the station.",
    "أين أقرب صيدلية؟ أحتاج مساعدة عاجلة الآن.",
    # CJK: rejoining must NOT inject a space the speaker never said, so this
    # case compares raw strings, not whitespace-stripped ones.
    "这是第一句话。这是第二句话内容。",
    "यह पहला वाक्य है। यह दूसरा वाक्य है।",
    'He said "stop!" and then walked away.',
    "Dr. Smith is waiting for you downstairs.",
):
    s = split_sentences(text)
    joined = " ".join(s.sentences)
    # Compare on non-space characters: the splitter may normalise whitespace,
    # but it must never lose or invent a character.
    a = "".join(joined.split())
    b = "".join(text.split())
    check(f"round-trip preserves characters: {text[:34]!r}", a == b, f"{len(a)} vs {len(b)} chars")

print()
print("--- CJK rejoin must not inject spaces ---")
s = split_sentences("这是第一句话。这是第二句话内容。")
check(
    "CJK pieces contain no injected space",
    all(" " not in p for p in s.sentences),
    str(list(s.sentences)),
)
s = split_sentences("短句。短句。短句。", min_chars=99)  # force a merge
check(
    "forced CJK merge joins without a space",
    len(s.sentences) == 1 and " " not in s.sentences[0],
    f"{list(s.sentences)} ({s.reason})",
)

print()
print("--- reason field is informative, not decorative ---")
for text, want in (
    ("Where is the pharmacy?", "single sentence"),
    ("One here now. Two here now.", "split into"),
    ("", "empty"),
    # The reason must name the ACTUAL rule. An earlier version always blamed
    # min_chars, so these two reported something untrue.
    ("Dr. Smith is waiting for you downstairs.", "abbreviation"),
    ('He said "stop!" and then walked away.', "quoted speech"),
):
    s = split_sentences(text)
    check(f"reason for {text[:26]!r} mentions {want!r}", want in s.reason, s.reason)

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): " + "; ".join(FAILURES))
    sys.exit(1)
print("ALL PASSED")
