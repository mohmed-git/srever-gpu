"""Audio ingestion: PCM and compressed (Opus/WebM/OGG/WAV) -> float32 mono 16 kHz.

Whisper wants float32 mono at 16 kHz. The earbuds may send either raw
little-endian int16 PCM (cheapest, no decoder) or Opus in an OGG/WebM
container. PCM is handled with numpy alone; compressed input goes through PyAV
(already installed as a faster-whisper dependency), so no extra native codec
package is needed.

Everything here is pure CPU and runs off the event loop in a worker thread.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Final

import numpy as np

TARGET_RATE: Final[int] = 16000
_INT16_FULL_SCALE: Final[float] = 32768.0


class AudioError(ValueError):
    """Raised when the client's audio cannot be used. Never silently ignored."""


@dataclass(frozen=True)
class Decoded:
    samples: np.ndarray  # float32, mono, TARGET_RATE
    sample_rate: int
    duration_s: float
    source_format: str
    peak: float
    rms: float

    @property
    def looks_silent(self) -> bool:
        # A conservative floor: below this the ASR result would be an artefact
        # of noise, and reporting a latency for it would be a hollow number.
        return self.peak < 1e-4 or self.rms < 1e-5


def pcm16_to_float32(raw: bytes) -> np.ndarray:
    if len(raw) < 2:
        raise AudioError("pcm payload shorter than one sample")
    if len(raw) % 2:
        raw = raw[:-1]  # drop a torn trailing byte rather than misalign
    arr = np.frombuffer(raw, dtype="<i2").astype(np.float32) / _INT16_FULL_SCALE
    return arr


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    """Resample with soxr when available, else linear interpolation.

    soxr is present in this environment (1.1.0) and is far better than linear
    interpolation; the fallback exists so a slimmer image still runs.
    """
    if src_rate == dst_rate:
        return samples.astype(np.float32, copy=False)
    if samples.size == 0:
        return samples.astype(np.float32, copy=False)
    try:
        import soxr

        return np.asarray(soxr.resample(samples, src_rate, dst_rate), dtype=np.float32)
    except Exception:
        ratio = dst_rate / float(src_rate)
        n_out = int(round(samples.size * ratio))
        if n_out <= 1:
            raise AudioError("audio too short to resample")
        x_old = np.arange(samples.size, dtype=np.float64)
        x_new = np.linspace(0.0, samples.size - 1, n_out, dtype=np.float64)
        return np.interp(x_new, x_old, samples).astype(np.float32)


def _to_mono(samples: np.ndarray, channels: int) -> np.ndarray:
    if channels <= 1:
        return samples
    usable = (samples.size // channels) * channels
    if usable == 0:
        raise AudioError("audio shorter than one frame")
    return samples[:usable].reshape(-1, channels).mean(axis=1)


def decode_pcm(
    raw: bytes, *, sample_rate: int = TARGET_RATE, channels: int = 1
) -> Decoded:
    mono = _to_mono(pcm16_to_float32(raw), channels)
    out = resample_linear(mono, sample_rate)
    return _finish(out, "pcm_s16le")


def decode_compressed(raw: bytes) -> Decoded:
    """Decode Opus/OGG/WebM/WAV/MP3 via PyAV. Raises AudioError on failure."""
    try:
        import av
    except Exception as exc:  # pragma: no cover
        raise AudioError(f"no compressed-audio decoder available: {exc}") from exc

    try:
        with av.open(io.BytesIO(raw)) as container:
            streams = [s for s in container.streams if s.type == "audio"]
            if not streams:
                raise AudioError("container has no audio stream")
            stream = streams[0]
            resampler = av.audio.resampler.AudioResampler(
                format="s16", layout="mono", rate=TARGET_RATE
            )
            chunks: list[np.ndarray] = []
            codec = getattr(stream.codec_context, "name", "unknown")
            for frame in container.decode(stream):
                for out_frame in resampler.resample(frame):
                    chunks.append(out_frame.to_ndarray().reshape(-1))
            # Flush the resampler; PyAV buffers internally.
            for out_frame in resampler.resample(None):
                chunks.append(out_frame.to_ndarray().reshape(-1))
    except AudioError:
        raise
    except Exception as exc:
        raise AudioError(f"could not decode audio container: {exc}") from exc

    if not chunks:
        raise AudioError("decoder produced no samples")
    joined = np.concatenate(chunks).astype(np.int16, copy=False)
    return _finish(joined.astype(np.float32) / _INT16_FULL_SCALE, f"container:{codec}")


_MAGIC: Final[tuple[tuple[bytes, str], ...]] = (
    (b"OggS", "ogg"),
    (b"RIFF", "wav"),
    (b"\x1a\x45\xdf\xa3", "webm"),
    (b"fLaC", "flac"),
    (b"ID3", "mp3"),
)


def sniff(raw: bytes) -> str:
    """Return 'pcm' or a container hint. Magic bytes, not client claims."""
    head = raw[:4]
    for magic, name in _MAGIC:
        if raw.startswith(magic):
            return name
    if len(head) >= 2 and head[:2] == b"\xff\xfb":
        return "mp3"
    return "pcm"


def decode(
    raw: bytes,
    *,
    declared_format: str | None = None,
    sample_rate: int = TARGET_RATE,
    channels: int = 1,
) -> Decoded:
    """Decode client audio, trusting magic bytes over the declared format.

    A client that says "pcm" but sends OGG would otherwise be transcribed as
    noise and produce a plausible-looking latency for a garbage result.
    """
    if not raw:
        raise AudioError("empty audio payload")
    detected = sniff(raw)
    declared = (declared_format or "").strip().lower()
    if detected != "pcm":
        return decode_compressed(raw)
    if declared in {"opus", "ogg", "webm", "wav", "mp3", "flac"}:
        # Declared compressed but no magic match: try the decoder, and if it
        # fails, say so rather than reinterpreting the bytes as PCM.
        return decode_compressed(raw)
    return decode_pcm(raw, sample_rate=sample_rate, channels=channels)


def _finish(samples: np.ndarray, source_format: str) -> Decoded:
    if samples.size == 0:
        raise AudioError("no samples after decode")
    samples = np.ascontiguousarray(samples, dtype=np.float32)
    if not np.all(np.isfinite(samples)):
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if samples.size else 0.0
    return Decoded(
        samples=samples,
        sample_rate=TARGET_RATE,
        duration_s=samples.size / float(TARGET_RATE),
        source_format=source_format,
        peak=peak,
        rms=rms,
    )
