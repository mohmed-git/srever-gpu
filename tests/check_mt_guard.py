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
from app.mt import build_mt_engine, is_translatable, detect_person_mismatch  # noqa: E402


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

    print("\n--- Evaluating detect_person_mismatch against benchmark cases ---")
    detector_cases = [
        # Branch 2: ar -> en (omission errors)
        ("أنا بخير أنت؟", "How are you?", "ar", "en", True),
        ("أنا بخير أنت؟", "I am fine, and you?", "ar", "en", False),
        ("بخير", "Fine", "ar", "en", False),
        ("الحمد لله", "Praise be to God", "ar", "en", False),
        ("هو بخير", "He is fine", "ar", "en", False),
        ("هي بخير", "She is fine", "ar", "en", False),
        ("هم بخير", "They are fine", "ar", "en", False),
        ("أريد غرفة لليلتين", "A room for two nights", "ar", "en", True),
        ("أريد غرفة لليلتين", "I want a room for two nights", "ar", "en", False),
        ("سأذهب غداً", "Going tomorrow", "ar", "en", True),
        ("سأذهب غداً", "I will go tomorrow", "ar", "en", False),
        ("أشعر بالتعب", "Feeling tired", "ar", "en", True),
        ("أشعر بالتعب", "I feel tired", "ar", "en", False),
        ("لدي سؤال", "A question", "ar", "en", True),
        ("لدي سؤال", "I have a question", "ar", "en", False),
        ("أعتقد أنه جيد", "Think it is good", "ar", "en", True),
        ("أعتقد أنه جيد", "I think it is good", "ar", "en", False),
        # Branch 1: en -> ar (chat leak errors)
        ("How are you?", "أنا بخير، أنت؟", "en", "ar", True),
        ("How are you?", "كيف حالك؟", "en", "ar", False),
        ("I am fine, thank you.", "أنا بخير، شكراً لك.", "en", "ar", False),
        ("Are you okay?", "هل أنت بخير؟", "en", "ar", False),
        ("Do you feel well?", "هل تشعر أنك بخير؟", "en", "ar", False),
        ("Thank God you are here.", "الحمد لله أنك هنا.", "en", "ar", False),
        ("How can I help you?", "كيف يمكنني مساعدتك؟", "en", "ar", False),
        ("Where is the station?", "أين المحطة؟", "en", "ar", False),
        ("Can you help me?", "هل يمكنك مساعدتي؟", "en", "ar", False),
        ("I want to order food.", "أريد طلب الطعام.", "en", "ar", False),
        ("We need a taxi.", "نحتاج إلى سيارة أجرة.", "en", "ar", False),
        ("What is the time?", "كم الوقت؟", "en", "ar", False),
        ("Good morning.", "صباح الخير.", "en", "ar", False),
        ("See you tomorrow.", "أراك غداً.", "en", "ar", False),
    ]

    detector_fps = 0
    detector_fns = 0
    for src, tgt, sl, tl, expected in detector_cases:
        got = detect_person_mismatch(src, tgt, sl, tl)
        ok = got == expected
        if not ok:
            if got and not expected:
                detector_fps += 1
                failures.append(f"FALSE POSITIVE: {src!r} -> {tgt!r} tripped detector")
            else:
                detector_fns += 1
                failures.append(f"FALSE NEGATIVE: {src!r} -> {tgt!r} missed by detector")
        status = "ok" if ok else "FAIL"
        print(f"[{status}] mismatch({src[:25]!r} -> {tgt[:25]!r}) = {got} (expected {expected})")

    print(f"Detector benchmark: {len(detector_cases)} cases, {detector_fps} false positives, {detector_fns} false negatives")

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
