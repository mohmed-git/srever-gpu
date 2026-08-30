"""End-to-end verification against a running server.

    python tests/check_server.py [base_url]

Every audio check asserts a NON-EMPTY transcript. This is deliberate: a
synthetic tone makes Whisper return 0 segments in ~20 ms, which looks like a
fast success and is actually a rejection. A test that only checks HTTP 200
would pass on that and certify a server that transcribes nothing.

The checks also assert the *negative* cases -- silence, blank text, an
unknown language, a mislabelled audio format -- because a server that answers
everything with 200 is not a server that works, it is a server that cannot
tell you when it is broken.
"""

from __future__ import annotations

import asyncio
import json
import math
import struct
import sys
import wave
from pathlib import Path

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
WS_BASE = BASE.replace("http://", "ws://").replace("https://", "wss://")
SPEECH_WAV = Path("/tmp/speech3s.wav")

failures: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> bool:
    print(f"[{'ok' if condition else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""))
    if not condition:
        failures.append(label)
    return condition


def load_pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate()


def synthetic_buzz(seconds: float = 3.0, rate: int = 16000) -> bytes:
    """A formant-like buzz: sounds vaguely speechy, transcribes to nothing.

    Kept as an explicit control so the suite proves the hollow-detection path
    fires, instead of assuming it does.
    """
    frames = []
    for i in range(int(seconds * rate)):
        t = i / rate
        value = (
            0.35 * math.sin(2 * math.pi * 120 * t)
            + 0.25 * math.sin(2 * math.pi * 700 * t)
            + 0.15 * math.sin(2 * math.pi * 1220 * t)
        )
        frames.append(int(max(-1.0, min(1.0, value)) * 20000))
    return struct.pack(f"<{len(frames)}h", *frames)


# =====================================================================
def test_health(client: httpx.Client) -> None:
    r = client.get(f"{BASE}/health", timeout=30)
    check(r.status_code == 200, "GET /health is 200", f"got {r.status_code}")
    body = r.json()
    check(body.get("ready") is True, "server reports ready")
    check(
        isinstance(body.get("language_count"), int) and body["language_count"] >= 90,
        "language_count >= 90 (the brief's target)",
        f"got {body.get('language_count')}",
    )
    gpu = body.get("resources", {}).get("gpu", {})
    if gpu.get("available"):
        check(True, "GPU reported", f"{gpu.get('device_name')} {gpu.get('used_mb')} MB used")
    else:
        # The important part: absent must not be reported as zero.
        check(
            gpu.get("total_mb") is None and gpu.get("used_mb") is None,
            "absent GPU is null, not 0 MB",
            f"source={gpu.get('source')}",
        )
    engines = body.get("engines", {})
    print(
        f"       device={body.get('device')} asr={engines.get('asr', {}).get('model')} "
        f"mt={engines.get('mt', {}).get('backend')}"
    )


def test_rest(client: httpx.Client) -> None:
    cases = [
        ("Where is the nearest pharmacy?", "en", "ar"),
        ("أحتاج طبيباً الآن، من فضلك ساعدني.", "ar", "en"),
        ("I do not eat meat.", "en", "fr"),
        ("Thank you very much for your help today.", "en", "ja"),
    ]
    for text, src, dst in cases:
        r = client.post(
            f"{BASE}/translate",
            json={"text": text, "source": src, "target": dst},
            timeout=60,
        )
        if not check(r.status_code == 200, f"POST /translate {src}->{dst}", f"HTTP {r.status_code}"):
            continue
        body = r.json()
        translation = (body.get("translation") or "").strip()
        check(
            bool(translation),
            f"{src}->{dst} returned a non-empty translation",
            f"{translation!r} in {body.get('latency_ms')} ms",
        )
        check(
            body.get("hollow") is not True,
            f"{src}->{dst} not flagged hollow",
            str(body.get("hollow_reason")),
        )
        check(
            isinstance(body.get("latency_ms"), (int, float)),
            f"{src}->{dst} reports latency_ms",
        )


def test_rest_rejections(client: httpx.Client) -> None:
    # Blank text: must be refused, because the engine hallucinates on it.
    r = client.post(
        f"{BASE}/translate", json={"text": "   ", "source": "en", "target": "ar"}, timeout=30
    )
    check(r.status_code == 400, "blank text is refused (400)", f"HTTP {r.status_code}")
    if r.status_code == 400:
        check(r.json().get("error") == "empty_text", "blank text error code", str(r.json()))

    # Unknown language: must be refused, not silently routed.
    r = client.post(
        f"{BASE}/translate",
        json={"text": "hello", "source": "en", "target": "klingon"},
        timeout=30,
    )
    check(r.status_code == 400, "unknown target is refused (400)", f"HTTP {r.status_code}")

    # A language tag with a region should still work.
    r = client.post(
        f"{BASE}/translate", json={"text": "hello", "source": "en-US", "target": "ar-EG"}, timeout=60
    )
    check(r.status_code == 200, "regional tags (en-US -> ar-EG) are accepted", f"HTTP {r.status_code}")


def test_languages(client: httpx.Client) -> None:
    r = client.get(f"{BASE}/languages", timeout=30)
    check(r.status_code == 200, "GET /languages is 200")
    body = r.json()
    check(len(body.get("languages", [])) >= 90, "catalogue has >= 90 entries",
          f"{len(body.get('languages', []))}")
    check(bool(body.get("caveat")), "catalogue carries the quality caveat")
    arabic = [x for x in body.get("languages", []) if x["code"] == "ar"]
    check(bool(arabic) and arabic[0]["rtl"] is True, "Arabic is marked RTL")


# =====================================================================
async def _ws_utterance(
    url: str,
    payload: bytes,
    *,
    source: str,
    target: str,
    fmt: str | None,
    sample_rate: int = 16000,
    stream: bool | None = None,
) -> dict:
    """Return the TERMINAL frame of one utterance, plus the frames before it.

    WHY this drains instead of reading one frame: this helper is a client, and
    it caught a real break. When sentence streaming became the default, reading
    exactly one frame after flush returned a `type: "sentence"` frame, which
    carries no `asr_ms` and no `total_server_ms` -- the per-utterance totals
    only exist once every sentence is done. A client written against the old
    one-frame contract would therefore have read sentence 1 as the whole
    answer and acted on a partial translation.

    The terminal frame is returned with `_frames` attached so callers can
    assert on how many arrived, which is the difference between the two modes.
    """
    import websockets

    async with websockets.connect(url, max_size=32 * 1024 * 1024) as ws:
        hello = json.loads(await ws.recv())
        assert hello.get("event") == "ready", hello
        config: dict = {"source": source, "target": target, "sample_rate": sample_rate}
        if fmt:
            config["format"] = fmt
        if stream is not None:
            config["stream"] = stream
        await ws.send(json.dumps(config))
        await ws.recv()  # config ack
        # Send in 20 ms chunks, the way an earbud would.
        chunk = sample_rate * 2 // 50
        for offset in range(0, len(payload), chunk):
            await ws.send(payload[offset : offset + chunk])
        await ws.send(json.dumps({"action": "flush"}))

        frames: list[dict] = []
        while True:
            frame = json.loads(await ws.recv())
            frames.append(frame)
            kind = frame.get("type")
            # Terminal: the summary frame, an error, or -- in unified mode --
            # a payload with no "type" field at all.
            if kind in ("final", "error") or kind is None or frame.get("error"):
                break
        terminal = dict(frames[-1])
        terminal["_frames"] = frames
        return terminal


def test_websocket() -> None:
    url = f"{WS_BASE}/ws/v1/translate-stream"

    if not SPEECH_WAV.exists():
        check(False, "real-speech fixture exists", f"missing {SPEECH_WAV}")
        return
    pcm, rate = load_pcm(SPEECH_WAV)

    # 1) Real speech: the transcript MUST be non-empty. This is the check that
    #    a hollow measurement would fail.
    result = asyncio.run(
        _ws_utterance(url, pcm, source="en", target="ar", fmt="pcm_s16le", sample_rate=rate)
    )
    original = (result.get("original_text") or "").strip()
    translated = (result.get("translated_text") or "").strip()
    check(bool(original), "WS real speech produced a transcript", repr(original[:60]))
    check(bool(translated), "WS real speech produced a translation", repr(translated[:60]))
    check(result.get("hollow") is not True, "WS real speech not hollow", str(result.get("hollow_reason")))
    for field in ("original_text", "translated_text", "source_lang", "target_lang",
                  "asr_ms", "mt_ms", "total_server_ms"):
        check(field in result, f"WS response contains '{field}'")
    print(
        f"       asr={result.get('asr_ms')} mt={result.get('mt_ms')} "
        f"total={result.get('total_server_ms')} rtf={result.get('real_time_factor')} "
        f"within_budget={result.get('within_budget')}"
    )

    # 2) Language auto-detection.
    result = asyncio.run(
        _ws_utterance(url, pcm, source="auto", target="fr", fmt="pcm_s16le", sample_rate=rate)
    )
    check(
        bool((result.get("original_text") or "").strip()),
        "WS auto-detect produced a transcript",
        f"detected={result.get('source_lang')}",
    )
    check(result.get("language_detected") is True, "WS auto-detect flag is set")

    # 2b) The two contracts must BOTH work, and must agree on the fields an
    #     already-shipped client reads.
    #
    #     WHY this test exists: enabling sentence streaming by default broke
    #     test 1 above. The old helper read exactly one frame after flush and
    #     got a `type: "sentence"` frame, which has no `asr_ms` and no
    #     `total_server_ms` -- those totals cannot exist until the last
    #     sentence is done. That is precisely how a real client would have
    #     read sentence 1 as the finished translation. So the opt-out is
    #     locked here, and so is field parity between the two modes.
    shared = (
        "original_text",
        "translated_text",
        "source_lang",
        "target_lang",
        "asr_ms",
        "mt_ms",
        "total_server_ms",
        "within_budget",
        "real_time_factor",
        "language_detected",
    )

    unified = asyncio.run(
        _ws_utterance(
            url, pcm, source="en", target="ar", fmt="pcm_s16le",
            sample_rate=rate, stream=False,
        )
    )
    check(
        len(unified["_frames"]) == 1,
        "WS stream=false sends exactly ONE frame",
        f"got {len(unified['_frames'])}",
    )
    check(
        unified.get("type") is None,
        "WS stream=false frame carries no 'type' field (legacy shape)",
        f"type={unified.get('type')!r}",
    )
    for field in shared:
        check(field in unified, f"unified frame contains '{field}'")

    streamed = asyncio.run(
        _ws_utterance(
            url, pcm, source="en", target="ar", fmt="pcm_s16le",
            sample_rate=rate, stream=True,
        )
    )
    check(
        streamed.get("type") == "final",
        "WS stream=true ends with a 'final' frame",
        f"type={streamed.get('type')!r}",
    )
    for field in shared:
        check(
            field in streamed,
            f"streaming final frame also contains '{field}'",
        )
    sentence_frames = [f for f in streamed["_frames"] if f.get("type") == "sentence"]
    check(
        len(sentence_frames) >= 1,
        "WS stream=true sent at least one sentence frame before final",
        f"{len(sentence_frames)} sentence frame(s)",
    )
    check(
        all(f.get("is_last") is False for f in sentence_frames[:-1])
        and (not sentence_frames or sentence_frames[-1].get("is_last") is True),
        "exactly the LAST sentence frame is flagged is_last",
    )
    # The saving is time-to-first-word only. If first_sentence_ms ever exceeded
    # total_server_ms the field would be meaningless, so assert the direction.
    first_ms = streamed.get("first_sentence_ms")
    total_ms = streamed.get("total_server_ms")
    if first_ms is not None and total_ms is not None:
        check(
            first_ms <= total_ms + 1.0,
            "first_sentence_ms is not after total_server_ms",
            f"first={first_ms} total={total_ms}",
        )
        if len(sentence_frames) > 1:
            check(
                streamed.get("time_saved_to_first_word_ms", 0) > 0,
                "multi-sentence utterance reports a real time-to-first-word saving",
                f"saved={streamed.get('time_saved_to_first_word_ms')} ms "
                f"({len(sentence_frames)} sentences)",
            )
        else:
            print(
                "       [skip] saving assertion: ASR produced a single sentence, "
                "so there is nothing to overlap and no saving can exist"
            )

    # 3) Opus in an OGG container, declared wrongly as pcm: magic bytes must win.
    ogg = Path("/tmp/speech3s.ogg")
    if ogg.exists():
        result = asyncio.run(
            _ws_utterance(url, ogg.read_bytes(), source="en", target="ar", fmt="pcm_s16le")
        )
        check(
            bool((result.get("original_text") or "").strip()),
            "WS opus declared as pcm still transcribes (magic bytes win)",
            f"format={result.get('audio_format')}",
        )

    # 4) Silence: must be REFUSED, not answered with a fast empty success.
    silence = b"\x00\x00" * (16000 * 2)
    result = asyncio.run(
        _ws_utterance(url, silence, source="en", target="ar", fmt="pcm_s16le")
    )
    check(
        result.get("error") == "silent_audio",
        "WS silence is refused with 'silent_audio'",
        str(result.get("error") or result)[:80],
    )

    # 5) Synthetic buzz: either refused up front, or flagged hollow. What it
    #    must NOT be is a 200 with empty text and a flattering latency.
    result = asyncio.run(
        _ws_utterance(url, synthetic_buzz(), source="en", target="ar", fmt="pcm_s16le")
    )
    refused = result.get("error") is not None
    hollow = result.get("hollow") is True
    has_text = bool((result.get("original_text") or "").strip())
    check(
        refused or hollow or has_text,
        "WS synthetic buzz is refused or flagged hollow, never a silent empty success",
        f"error={result.get('error')} hollow={hollow} text={has_text}",
    )

    # 6) Missing target must be an explicit error.
    async def _no_target() -> dict:
        import websockets

        async with websockets.connect(url) as ws:
            await ws.recv()
            await ws.send(pcm[:6400])
            await ws.send(json.dumps({"action": "flush"}))
            return json.loads(await ws.recv())

    result = asyncio.run(_no_target())
    check(
        result.get("error") == "missing_target",
        "WS flush without a target is an explicit error",
        str(result.get("error")),
    )


def test_metrics(client: httpx.Client) -> None:
    r = client.get(f"{BASE}/metrics", timeout=30)
    check(r.status_code == 200, "GET /metrics is 200")
    body = r.json()
    total = body.get("latency", {}).get("total_server_ms", {})
    check(total.get("available") is True, "metrics report measured total latency",
          f"n={total.get('samples')} p50={total.get('p50_ms')}")
    if total.get("samples", 0) < 20:
        check(total.get("p95_ms") is None, "p95 withheld below 20 samples",
              str(total.get("p95_note")))
    budget = body.get("budget", {})
    check("within_budget_fraction" in budget, "metrics report budget compliance",
          f"{budget.get('within_budget_fraction')} of {budget.get('samples')}")
    queues = body.get("queues", {})
    for name in ("asr", "mt"):
        stats = queues.get(name, {})
        check(name in queues, f"queue stats present for '{name}'",
              f"capacity={stats.get('capacity_rps')} rps, "
              f"per_item={stats.get('measured_per_item_ms')} ms")


def test_capabilities(client: httpx.Client) -> None:
    r = client.get(f"{BASE}/capabilities", timeout=30)
    check(r.status_code == 200, "GET /capabilities is 200")
    body = r.json()
    check(
        body.get("concurrency_reality", {}).get("verdict") == "not reachable as specified",
        "capabilities states the concurrency verdict plainly",
    )
    check(
        "modelled_only" in body.get("measured_vs_modelled", {}),
        "capabilities separates measured from modelled",
    )


def main() -> int:
    with httpx.Client() as client:
        for name, fn in [
            ("health", lambda: test_health(client)),
            ("languages", lambda: test_languages(client)),
            ("rest", lambda: test_rest(client)),
            ("rest rejections", lambda: test_rest_rejections(client)),
            ("websocket", test_websocket),
            ("metrics", lambda: test_metrics(client)),
            ("capabilities", lambda: test_capabilities(client)),
        ]:
            print(f"\n--- {name} ---")
            try:
                fn()
            except Exception as exc:
                check(False, f"{name} raised", f"{type(exc).__name__}: {exc}")

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
