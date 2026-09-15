"""app.routing - Closed sink enums and deterministic Tri-Mode routing.

Implements the Protocol 2 Blueprint specified by Chief Architect:
- Strict closed enum: SINKS = ("earbud", "earbud_L", "earbud_R", "phone_speaker", "none")
- Attribution key: "capture_channel" (not "capture_source")
- Dynamic listener routing in share_b: sink = speakers[1 - speaker_id].sink, resolved via channel_map
"""
from __future__ import annotations

from typing import Any

# Contract: Closed sink enum agreed across Server and Mobile
SINKS: tuple[str, ...] = (
    "earbud",
    "earbud_L",
    "earbud_R",
    "phone_speaker",
    "none",
)


def resolve_route(
    state: Any,
    capture: str | None = None,
    lid_winner: str | None = None,
    lid_conf: float | None = None,
    speaker_id: int | None = None,
) -> dict[str, Any]:
    """Deterministically resolves routing and sinks for Tri-Mode (Protocol 2) and pair_auto.

    Blueprint Invariants:
    - Sinks strictly belong to SINKS: {"earbud", "earbud_L", "earbud_R", "phone_speaker", "none"}
    - Attribution key is 'capture_channel' for hybrid_a, 'lid' for share_b, 'pinned' for listen_c.
    - share_b routes strictly to the listener (1 - speaker_id) via channel_map.
    """
    mode = getattr(state, "mode", "pair_auto")
    languages = getattr(state, "languages", ["en", "ar"])
    channel_map = getattr(state, "channel_map", {"0": "L", "1": "R"})

    lang0 = languages[0] if languages else (getattr(state, "source", None) or "en")
    lang1 = languages[1] if len(languages) > 1 else (getattr(state, "target", None) or "ar")

    if mode == "hybrid_a":
        # Mode 1: Earbud User (spk 0) + Guest on Phone (spk 1)
        # Translation routes strictly to the other party's sink (the listener)
        if capture == "earbud_mic" or speaker_id == 0:
            spk = 0
            src = lang0
            dst = lang1
            sink = "phone_speaker"  # Guest hears translation on phone speaker
            attribution = "capture_channel"
        else:
            spk = 1
            src = lang1
            dst = lang0
            sink = "earbud"  # Earbud user hears translation in-ear
            attribution = "capture_channel"

        result = {
            "speaker_id": spk,
            "sink": sink,
            "attribution": attribution,
            "lid_conf": lid_conf,
            "source": src,
            "target": dst,
            "direction": f"{src}->{dst}",
            "channel": channel_map.get(str(spk), "L" if spk == 0 else "R").upper(),
        }

    elif mode == "share_b":
        # Mode 2: Shared Buds (L/R)
        # Attribution by constrained LID between the two languages.
        if lid_winner == lang1 or speaker_id == 1:
            spk = 1
            src = lang1
            dst = lang0
        else:
            spk = 0
            src = lang0
            dst = lang1

        # Blueprint rule: The sink is strictly the LISTENER'S, never the speaker's.
        # When speaker 0 speaks, translation goes into speaker 1's earbud.
        # When speaker 1 speaks, translation goes into speaker 0's earbud.
        listener = 1 - spk
        listener_chan = channel_map.get(str(listener), "R" if listener == 1 else "L").upper()
        sink = f"earbud_{listener_chan}"

        result = {
            "speaker_id": spk,
            "sink": sink,
            "attribution": "lid",
            "lid_conf": lid_conf,
            "source": src,
            "target": dst,
            "direction": f"{src}->{dst}",
            "channel": listener_chan,
        }

    elif mode == "listen_c":
        # Mode 3: Listen-only (Tour / Lecture)
        src = getattr(state, "source", None) or lang0
        dst = getattr(state, "target", None) or lang1
        result = {
            "speaker_id": None,
            "sink": "earbud",
            "attribution": "pinned",
            "lid_conf": lid_conf,
            "source": src,
            "target": dst,
            "direction": f"{src}->{dst}",
            "channel": "both",
        }

    else:
        # Default / pair_auto / single
        spk = speaker_id if speaker_id is not None else (0 if lid_winner == lang0 else 1)
        listener = 1 - spk
        listener_chan = channel_map.get(str(listener), "R" if listener == 1 else "L").upper()
        sink = f"earbud_{listener_chan}"
        src = lang0 if spk == 0 else lang1
        dst = lang1 if spk == 0 else lang0
        result = {
            "speaker_id": spk,
            "sink": sink,
            "attribution": "lid" if mode == "pair_auto" else "pinned",
            "lid_conf": lid_conf,
            "source": src,
            "target": dst,
            "direction": f"{src}->{dst}",
        }

    # Contract assertion: Sink must be a member of the canonical closed enum
    if result["sink"] not in SINKS:
        result["sink"] = "none"

    return result
