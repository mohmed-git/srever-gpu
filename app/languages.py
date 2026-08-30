"""Language catalogue.

The list is read from `faster_whisper.tokenizer._LANGUAGE_CODES` at import
time rather than hardcoded, so it can never drift from what the installed ASR
engine actually accepts. Measured on faster-whisper 1.2.1: 100 codes.

Honest caveat, stated in the API too: "present in the tokenizer" is not the
same as "usable in a commercial earbud". The Whisper paper reports WER above
50% on roughly 20 of the lowest-resource languages. Per-language quality is
NOT verified here — that needs a GPU and per-language test corpora.
"""

from __future__ import annotations

from typing import Final

try:
    from faster_whisper.tokenizer import _LANGUAGE_CODES as _WHISPER_CODES
except Exception:  # pragma: no cover - engine missing
    _WHISPER_CODES = frozenset()

# Right-to-left scripts, used by the client for text layout.
RTL_CODES: Final[frozenset[str]] = frozenset({"ar", "he", "fa", "ur", "ps", "sd", "yi"})

# English names for the codes Whisper accepts. Anything missing falls back to
# the bare code rather than inventing a name.
_NAMES: Final[dict[str, str]] = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese",
    "az": "Azerbaijani", "ba": "Bashkir", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish",
    "de": "German", "el": "Greek", "en": "English", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fo": "Faroese", "fr": "French", "gl": "Galician", "gu": "Gujarati",
    "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew", "hi": "Hindi",
    "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jw": "Javanese", "ka": "Georgian", "kk": "Kazakh", "km": "Khmer",
    "kn": "Kannada", "ko": "Korean", "la": "Latin", "lb": "Luxembourgish",
    "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian",
    "mg": "Malagasy", "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam",
    "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk",
    "no": "Norwegian", "oc": "Occitan", "pa": "Punjabi", "pl": "Polish",
    "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak",
    "sl": "Slovenian", "sn": "Shona", "so": "Somali", "sq": "Albanian",
    "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "tg": "Tajik", "th": "Thai",
    "tk": "Turkmen", "tl": "Tagalog", "tr": "Turkish", "tt": "Tatar",
    "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese",
    "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese",
}

# Common aliases the earbuds firmware may send.
ALIASES: Final[dict[str, str]] = {
    "iw": "he", "in": "id", "ji": "yi", "jv": "jw",
    "zh-cn": "zh", "zh-hans": "zh", "zh-tw": "zh", "zh-hant": "zh",
    "nb": "no", "nb-no": "no", "pt-br": "pt", "en-us": "en", "en-gb": "en",
    "ar-sa": "ar", "ar-eg": "ar", "es-419": "es", "fil": "tl",
}

ASR_CODES: Final[frozenset[str]] = frozenset(_WHISPER_CODES)
# Whisper cannot transcribe a language it does not know, but the MT stage can
# still translate *into* many targets. Keeping the two sets separate avoids
# claiming ASR coverage we do not have.
MT_TARGET_CODES: Final[frozenset[str]] = frozenset(_NAMES) | ASR_CODES


def normalise(code: str | None) -> str | None:
    """Normalise a client language tag. Returns None when unusable."""
    if not code:
        return None
    tag = code.strip().lower().replace("_", "-")
    if tag in {"auto", "detect", "automatic"}:
        return "auto"
    if tag in ALIASES:
        return ALIASES[tag]
    if tag in MT_TARGET_CODES:
        return tag
    base = tag.split("-", 1)[0]
    if base in ALIASES:
        return ALIASES[base]
    if base in MT_TARGET_CODES:
        return base
    return None


def name_of(code: str) -> str:
    return _NAMES.get(code, code)


def is_rtl(code: str) -> bool:
    return code in RTL_CODES


def catalogue() -> list[dict[str, object]]:
    codes = sorted(MT_TARGET_CODES)
    return [
        {
            "code": c,
            "name": name_of(c),
            "rtl": is_rtl(c),
            "asr": c in ASR_CODES,
            "mt_target": True,
        }
        for c in codes
    ]


ASR_LANGUAGE_COUNT: Final[int] = len(ASR_CODES)
MT_LANGUAGE_COUNT: Final[int] = len(MT_TARGET_CODES)
