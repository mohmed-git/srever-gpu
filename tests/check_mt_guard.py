"""Verify the blank-input hallucination guard in the MT engine.

Regression this locks down (MEASURED, not theoretical): before the guard,
translating the string "  " through M2M100-418M CT2 int8 returned the Arabic
string 'مـنـها' -- a confident hallucination. An engine handed nothing does not
reliably return nothing, so untranslatable input must be rejected before
inference, and the result must be flagged `hollow`, never reported as a fast
success.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.mt import build_mt_engine, is_translatable  # noqa: E402


def main() -> int:
    failures: list[str] = []

    # Pure-function checks first: they need no model.
    for text, expected in [
        ("مرحبا", True),
        ("你好", True),
        ("hello", True),
        ("123", True),
        ("🙂", False),
        ("  ", False),
        ("", False),
        ("...", False),
        ("\n\t", False),
    ]:
        got = is_translatable(text)
        status = "ok" if got == expected else "FAIL"
        if got != expected:
            failures.append(f"is_translatable({text!r}) = {got}, expected {expected}")
        print(f"[{status}] is_translatable({text!r}) = {got}")

    engine = build_mt_engine(Settings())
    engine.load()
    print(f"backend={engine.name} load={engine.load_seconds:.2f}s")

    # Untranslatable input must be flagged hollow, whatever the model emits.
    for bad in ["  ", "...", "", "\n\t "]:
        result = engine.translate(bad, "en", "ar")
        ok = result.hollow
        if not ok:
            failures.append(f"blank input {bad!r} was NOT flagged hollow (text={result.text!r})")
        print(
            f"[{'ok' if ok else 'FAIL'}] {bad!r} -> hollow={result.hollow} "
            f"text={result.text!r} reason={(result.hollow_reason or '')[:50]!r}"
        )

    # Real input must still translate, and must NOT be flagged hollow.
    for text, src, dst in [
        ("Hello there.", "en", "ar"),
        ("أين أقرب صيدلية؟", "ar", "en"),
    ]:
        result = engine.translate(text, src, dst)
        ok = (not result.hollow) and bool(result.text.strip())
        if not ok:
            failures.append(f"real input {text!r} produced hollow/empty result")
        print(
            f"[{'ok' if ok else 'FAIL'}] {src}->{dst} {text!r} -> {result.text!r} "
            f"({result.mt_ms} ms, {result.output_tokens} tok)"
        )

    print()
    if failures:
        print(f"FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
