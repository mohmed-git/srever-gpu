"""Machine-translation engines, selected by what the box can actually run.

Backends
--------
`qwen_vllm`   Qwen2.5-*-Instruct on vLLM. Continuous batching + PagedAttention.
              The intended production path on the RTX 3060.
`qwen_hf`     Qwen2.5-*-Instruct on transformers. Works anywhere, slower;
              used when vLLM is absent (e.g. a Windows box or this sandbox).
`m2m100_ct2`  M2M100-418M on CTranslate2 int8. A dedicated NMT model, not an
              LLM: ~15x smaller, no prompt to ignore, and it is the only
              backend that fits this sandbox (2 cores / 985 MB RAM / 2.4 GB
              free disk), so it is what the CPU verification runs on.

Why the default is AWQ int4, and why it is not as fast as first estimated
------------------------------------------------------------------------
MEASURED, by reading the real safetensors headers over HTTP range requests
(no download) rather than multiplying a parameter count:

    Qwen/Qwen2.5-1.5B-Instruct       model.safetensors = 3.087 GB  (fp16)
    Qwen/Qwen2.5-1.5B-Instruct-AWQ   model.safetensors = 1.614 GB  (int4)

The AWQ file breaks down as:

    qweight (int4 packed)   0.655 GB   196 tensors
    lm_head        (fp16)   0.467 GB     <- NOT quantised
    embed_tokens   (fp16)   0.467 GB     <- NOT quantised
    scales/qzeros/norms     0.025 GB
                            -------
                            1.614 GB

An earlier estimate in this repository said AWQ would be 0.72 GB and 16.0 ms.
That was WRONG: it quantised the whole parameter count, ignoring that AWQ
leaves the embedding and lm_head matrices in fp16 -- together 0.934 GB, more
than the quantised linear layers themselves. (Note also that config.json says
tie_word_embeddings=True yet lm_head ships as a separate tensor, so the
embeddings are stored twice on disk.)

Corrected roofline on an RTX 3060 (360 GB/s), at the MEASURED median of 8
output tokens per spoken sentence, ideal and at a realistic 70% of peak
bandwidth:

    fp16          3.087 GB -> 117 tok/s -> 68.6 ms ideal / 98.0 ms real
    AWQ int4      1.614 GB -> 223 tok/s -> 35.9 ms ideal / 51.2 ms real
    AWQ int4 tied 1.147 GB -> 314 tok/s -> 25.5 ms ideal / 36.4 ms real

So AWQ does NOT reach the brief's 20-35 ms; it reaches ~36-51 ms. It is still
the right default -- roughly 2x faster than fp16 and 1.5 GB lighter -- and it
leaves room inside the 150 ms whole-request budget once ASR is added. It is
chosen on that basis, not on a claim it cannot support.

Because `quantization="awq"` requires a pre-quantised checkpoint, asking for
int4/awq also switches the model id to the `-AWQ` repository (Apache-2.0,
ungated, verified reachable). Passing "awq" against the unquantised base repo
fails at load; see _resolve_awq_model.

The prompt is deliberately terse: an instruct model that explains itself
("Sure! Here is the translation:") doubles the token count, and token count is
what the latency budget is actually spent on.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Final

from .config import Settings
from .languages import name_of

log = logging.getLogger("lingua.mt")

def make_translation_messages(text: str, src: str, dst: str) -> list[dict[str, str]]:
    src_name = name_of(src) if src != "auto" else "the detected language"
    dst_name = name_of(dst)

    if dst == "ar" or dst.startswith("ar"):
        system_content = (
            "You are an expert real-time English-to-Arabic translator for live spoken conversation. "
            "Translate the input faithfully and literally into Modern Standard Arabic.\n"
            "Rules:\n"
            "- Always translate question words accurately: 'Where is' -> 'أين', 'How is' -> 'كيف', 'When is' -> 'متى'.\n"
            "- Translate 'weather' as 'الطقس' or 'الجو', 'train station' as 'محطة القطار', 'please' as 'من فضلك'.\n"
            "- Translate EVERY word into Arabic. Never leave English words untranslated.\n"
            "- Output ONLY the Arabic translation without quotes, notes, or explanations."
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "Where is the train station?"},
            {"role": "assistant", "content": "أين محطة القطار؟"},
            {"role": "user", "content": "The weather is nice today."},
            {"role": "assistant", "content": "الطقس جميل اليوم."},
            {"role": "user", "content": "I would like to order coffee please."},
            {"role": "assistant", "content": "أود طلب قهوة من فضلك."},
            {"role": "user", "content": text},
        ]
    elif src == "ar" or src.startswith("ar"):
        system_content = (
            f"You are an expert real-time Arabic-to-{dst_name} translator for live spoken conversation. "
            f"Translate the spoken Arabic text accurately and naturally into {dst_name}.\n"
            f"Output ONLY the direct {dst_name} translation with no notes, explanations, or quotes."
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "أين محطة القطار؟"},
            {"role": "assistant", "content": "Where is the train station?"},
            {"role": "user", "content": "الطقس جميل اليوم."},
            {"role": "assistant", "content": "The weather is nice today."},
            {"role": "user", "content": text},
        ]
    else:
        system_content = (
            f"You are a professional real-time translator. "
            f"Translate the text from {src_name} directly into {dst_name}.\n"
            f"Rules:\n"
            f"1. Output MUST be entirely in {dst_name}.\n"
            f"2. Never use Chinese or any unrelated language.\n"
            f"3. Output ONLY the direct translation without preamble, notes, or quotes."
        )
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": f"Translate to {dst_name}:\n{text}"},
        ]



# Preambles an instruct model may emit despite the prompt. Stripped so the
# earbuds never speak "Here is the translation:" out loud.
_PREAMBLE = re.compile(
    r"^\s*(?:sure[,!.]?\s*)?(?:here(?:'s| is)\s+(?:the\s+)?translation\s*[:\-]?"
    r"|translation\s*[:\-]|translated\s*(?:text)?\s*[:\-])\s*",
    re.IGNORECASE,
)


_CJK_TARGETS: Final[frozenset[str]] = frozenset(
    {"zh", "zh-cn", "zh-tw", "zh-hans", "zh-hant", "yue", "wuu", "ja", "japanese", "chinese", "cantonese"}
)


def clean_translation(text: str, target_lang: str = "") -> str:
    out = (text or "").strip()
    out = _PREAMBLE.sub("", out).strip()
    # A model that wrapped the whole answer in quotes should not make the
    # earbuds pronounce the quotes.
    if len(out) >= 2 and out[0] in "\"'“”«" and out[-1] in "\"'“”»":
        inner = out[1:-1].strip()
        if inner and inner[0] not in "\"'“”«":
            out = inner

    norm_target = (target_lang or "").strip().lower()
    base_target = norm_target.split("-")[0]

    # Never sanitize CJK scripts when target is a CJK language!
    if norm_target not in _CJK_TARGETS and base_target not in _CJK_TARGETS:
        # If translating to Arabic, strip any stray Chinese characters or prefix labels
        if norm_target == "ar" or base_target == "ar":
            out = re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf]+", "", out).strip()
            out = re.sub(r"^(?:الترجمة|الترجمة إلى العربية|النص المترجم)\s*[:\-]\s*", "", out).strip()

    return out.strip()



@dataclass(frozen=True)
class MtResult:
    text: str
    mt_ms: float
    backend: str
    model: str
    input_tokens: int | None
    output_tokens: int | None
    batch_size: int
    hollow: bool
    hollow_reason: str | None


class MtEngine:
    """Common interface: load(), translate_batch(), info()."""

    name = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.load_seconds: float | None = None
        self.error: str | None = None

    def load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    @property
    def ready(self) -> bool:  # pragma: no cover - overridden
        return False

    def translate_batch(
        self, items: list[tuple[str, str, str]]
    ) -> list[MtResult]:  # pragma: no cover
        """items: list of (text, source_lang, target_lang)."""
        raise NotImplementedError

    def translate(self, text: str, source: str, target: str) -> MtResult:
        return self.translate_batch([(text, source, target)])[0]

    def warmup(self) -> dict[str, Any]:
        if not self.ready:
            return {"available": False, "reason": "model not loaded"}
        started = time.perf_counter()
        try:
            result = self.translate("Hello.", "en", "ar")
        except Exception as exc:
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
        return {
            "available": True,
            "warmup_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "output": result.text,
            "hollow": result.hollow,
        }

    def info(self) -> dict[str, Any]:  # pragma: no cover - overridden
        return {"backend": self.name, "ready": self.ready, "error": self.error}


# A "word" in any script, not just Latin. Used to decide whether there is
# anything to translate at all.
_HAS_WORD = re.compile(r"\w", re.UNICODE)


def is_translatable(text: str) -> bool:
    """True when the text contains at least one word character.

    MEASURED, and the reason this guard exists: feeding the M2M100 CT2 engine
    the string "  " returned 'مـنـهـا' -- a confident hallucination, not an
    empty result. An engine given nothing does not reliably return nothing, so
    blank input must be rejected before inference rather than after it.
    """
    return bool(_HAS_WORD.search(text or ""))


def _hollow_check(text: str, source_text: str) -> tuple[bool, str | None]:
    """An empty translation is a failure, not a fast success."""
    if not is_translatable(source_text):
        return True, (
            "source text contains no word characters: any output here is a "
            "hallucination (measured: '  ' -> '\u0645\u0640\u0646\u0640\u0647\u0640\u0627')"
        )
    if not text.strip():
        return True, "translation engine returned empty text"
    return False, None


# =====================================================================
# M2M100 on CTranslate2 -- the dedicated NMT path.
# =====================================================================
class M2M100Ct2Engine(MtEngine):
    """CTranslate2 M2M100-418M int8.

    Chosen for the CPU verification path because it is the only translation
    model that fits this sandbox, and because the call pattern below is already
    proven in production on the standalone server (same tokenisation, same
    target_prefix forcing, same detokenisation).
    """

    name = "m2m100_ct2"

    def __init__(self, settings: Settings, model_path: str) -> None:
        super().__init__(settings)
        self.model_path = model_path
        self._translator: Any = None
        self._sp: Any = None
        self._supported: frozenset[str] = frozenset()

    def load(self) -> None:
        import os

        import ctranslate2
        import sentencepiece as spm

        spm_path = os.path.join(self.model_path, "sentencepiece.bpe.model")
        if not os.path.isdir(self.model_path):
            raise FileNotFoundError(f"MT model directory not found: {self.model_path}")
        if not os.path.exists(spm_path):
            raise FileNotFoundError(f"sentencepiece.bpe.model missing in {self.model_path}")

        started = time.perf_counter()
        try:
            device = "cuda" if self.settings.on_cuda else "cpu"
            compute = "float16" if self.settings.on_cuda else "int8"
            self._translator = ctranslate2.Translator(
                self.model_path,
                device=device,
                compute_type=compute,
                inter_threads=1,
                intra_threads=max(1, self.settings.resolved_asr_cpu_threads()),
            )
            self._sp = spm.SentencePieceProcessor(model_file=spm_path)
            # Which __xx__ tokens this checkpoint really has. Claiming a
            # language whose token is absent would produce silent garbage.
            self._supported = self._read_supported_tokens()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        self.load_seconds = time.perf_counter() - started
        log.info(
            "MT ready: m2m100_ct2 path=%s langs=%d load=%.2fs",
            self.model_path,
            len(self._supported),
            self.load_seconds,
        )

    def _read_supported_tokens(self) -> frozenset[str]:
        import json
        import os

        vocab_file = os.path.join(self.model_path, "shared_vocabulary.json")
        if not os.path.exists(vocab_file):
            return frozenset()
        try:
            with open(vocab_file, "r", encoding="utf-8") as fh:
                vocab = json.load(fh)
        except Exception:
            return frozenset()
        tokens = vocab.keys() if isinstance(vocab, dict) else vocab
        return frozenset(
            t[2:-2]
            for t in tokens
            if isinstance(t, str) and t.startswith("__") and t.endswith("__") and len(t) > 4
        )

    @property
    def ready(self) -> bool:
        return self._translator is not None and self._sp is not None

    def supports(self, code: str) -> bool:
        return (not self._supported) or (code in self._supported)

    def translate_batch(self, items: list[tuple[str, str, str]]) -> list[MtResult]:
        if not self.ready:
            raise RuntimeError("MT model not loaded")
        if not items:
            return []

        batch_pieces: list[list[str]] = []
        prefixes: list[list[str]] = []
        for text, src, dst in items:
            pieces = [f"__{src}__"] + self._sp.EncodeAsPieces(text) + ["</s>"]
            batch_pieces.append(pieces)
            prefixes.append([f"__{dst}__"])

        started = time.perf_counter()
        results = self._translator.translate_batch(
            batch_pieces,
            target_prefix=prefixes,
            beam_size=1 if self.settings.on_cuda else 2,
            max_decoding_length=self.settings.mt_max_new_tokens,
            max_input_length=512,
            repetition_penalty=1.1,
            no_repeat_ngram_size=4,
            replace_unknowns=True,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        # Batched work costs one wall-clock span for the whole batch; charging
        # each request the full span would triple-count. Report the shared span
        # and let the caller see batch_size next to it.
        per_item_ms = round(elapsed_ms, 2)

        out: list[MtResult] = []
        for (text, _src, _dst), pieces, result in zip(items, batch_pieces, results):
            hypothesis = list(result.hypotheses[0]) if result.hypotheses else []
            if hypothesis:
                hypothesis = hypothesis[1:]  # drop the forced target token
            hypothesis = [
                t
                for t in hypothesis
                if not (t.startswith("__") and t.endswith("__"))
                and t not in ("</s>", "<s>", "<pad>")
            ]
            decoded = clean_translation("".join(hypothesis).replace("\u2581", " "))
            hollow, reason = _hollow_check(decoded, text)
            out.append(
                MtResult(
                    text=decoded,
                    mt_ms=per_item_ms,
                    backend=self.name,
                    model=self.model_path.rsplit("/", 1)[-1],
                    input_tokens=len(pieces),
                    output_tokens=len(hypothesis),
                    batch_size=len(items),
                    hollow=hollow,
                    hollow_reason=reason,
                )
            )
        return out

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine": "ctranslate2",
            "model": self.model_path,
            "device": self.settings.device,
            "compute_type": "float16" if self.settings.on_cuda else "int8",
            "supported_language_tokens": len(self._supported) or None,
            "max_new_tokens": self.settings.mt_max_new_tokens,
            "load_seconds": round(self.load_seconds, 2) if self.load_seconds else None,
            "ready": self.ready,
            "error": self.error,
            "note": "dedicated NMT model, not an instruct LLM: no prompt tokens to pay for",
        }


# =====================================================================
# Qwen2.5-Instruct on vLLM -- the intended RTX 3060 path.
# =====================================================================
class QwenVllmEngine(MtEngine):
    """Qwen2.5-*-Instruct served in-process by vLLM.

    vLLM's continuous batching is the reason this backend exists: it lets
    concurrent requests share decode steps instead of queueing behind each
    other. `mt_gpu_mem_fraction` defaults to 0.45 so Whisper keeps its share of
    the 12 GB rather than vLLM's default 0.90 taking the card.
    """

    name = "qwen_vllm"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._llm: Any = None
        self._tokenizer: Any = None
        self._sampling: Any = None
        # The checkpoint actually loaded, which may differ from MT_MODEL when a
        # quantised sibling was substituted. Reported in info().
        self.resolved_model: str = settings.mt_model
        self.model_substitution_note: str | None = None

    def load(self) -> None:
        from vllm import LLM, SamplingParams

        started = time.perf_counter()
        quant = self.settings.mt_quant.strip().lower()
        self.resolved_model, self.model_substitution_note = _resolve_awq_model(
            self.settings.mt_model, quant
        )
        if self.model_substitution_note:
            log.warning("MT model resolution: %s", self.model_substitution_note)
        kwargs: dict[str, Any] = {
            "model": self.resolved_model,
            "dtype": "float16",
            "gpu_memory_utilization": self.settings.mt_gpu_mem_fraction,
            "max_model_len": 1024,  # a conversation turn, not a document
            "enforce_eager": False,
            "disable_log_stats": True,
        }
        # vLLM names in-flight weight quantisation differently from CT2.
        if quant in {"int8", "fp8", "w8a8"}:
            kwargs["quantization"] = "fp8"
        elif quant in {"awq", "int4", "awq_marlin"}:
            kwargs["quantization"] = "awq"
        elif quant in {"gptq", "gptq_marlin"}:
            kwargs["quantization"] = "gptq"
        try:
            self._llm = LLM(**kwargs)
            self._tokenizer = self._llm.get_tokenizer()
            self._sampling = SamplingParams(
                temperature=0.0,  # translation is not a creative task
                top_p=1.0,
                max_tokens=self.settings.mt_max_new_tokens,
                repetition_penalty=1.05,
            )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        self.load_seconds = time.perf_counter() - started
        log.info(
            "MT ready: qwen_vllm model=%s quant=%s load=%.2fs",
            self.resolved_model,
            quant,
            self.load_seconds,
        )

    @property
    def ready(self) -> bool:
        return self._llm is not None

    def _prompt(self, text: str, src: str, dst: str) -> str:
        messages = make_translation_messages(text, src, dst)
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def translate_batch(self, items: list[tuple[str, str, str]]) -> list[MtResult]:
        if not self.ready:
            raise RuntimeError("MT model not loaded")
        if not items:
            return []
        prompts = [self._prompt(t, s, d) for t, s, d in items]
        started = time.perf_counter()
        outputs = self._llm.generate(prompts, self._sampling, use_tqdm=False)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

        results: list[MtResult] = []
        for (text, _s, _d), output in zip(items, outputs):
            completion = output.outputs[0] if output.outputs else None
            decoded = clean_translation(completion.text if completion else "", target_lang=_d)
            hollow, reason = _hollow_check(decoded, text)
            results.append(
                MtResult(
                    text=decoded,
                    mt_ms=elapsed_ms,
                    backend=self.name,
                    model=self.resolved_model,
                    input_tokens=len(output.prompt_token_ids or []),
                    output_tokens=len(completion.token_ids) if completion else 0,
                    batch_size=len(items),
                    hollow=hollow,
                    hollow_reason=reason,
                )
            )
        return results

    def info(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "backend": self.name,
            "engine": "vllm",
            # `model` is the checkpoint actually serving requests, not the one
            # configured: with quant="awq" this is the -AWQ repo, not the base.
            "model": self.resolved_model,
            "device": "cuda",
            "quantization": self.resolved_quant,
            "max_new_tokens": self.settings.mt_max_new_tokens,
            "load_seconds": self.load_seconds,
            "ready": self.ready,
            "error": self.error,
        }
        if self.model_substitution_note:
            out["model_substitution_note"] = self.model_substitution_note
        if self.resolved_model in _AWQ_REPOS.values():
            out["weights_gb_measured"] = 1.614 if "1.5B" in self.resolved_model else None
            out["quant_note"] = (
                "AWQ leaves embed_tokens and lm_head in fp16 (0.467 GB each, "
                "measured), so int4 weights total 1.614 GB, not the 0.72 GB a "
                "naive parameter-count estimate gives. Expect ~36-51 ms per "
                "8-token sentence at 60-80% of the 360 GB/s bandwidth, not the "
                "20-35 ms in the original brief."
            )
        return out


# =====================================================================
# Qwen2.5-Instruct on transformers -- portable fallback.
# =====================================================================
class QwenHfEngine(MtEngine):
    """Qwen2.5-*-Instruct via transformers.

    No continuous batching, so throughput is far below the vLLM path; this
    exists so the same API works on a box without vLLM (Windows, or CPU).
    """

    name = "qwen_hf"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model: Any = None
        self._tokenizer: Any = None
        self.load_seconds: float = 0.0

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        started = time.perf_counter()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.settings.mt_model)
            dtype = torch.float16 if self.settings.on_cuda else torch.float32
            self._model = AutoModelForCausalLM.from_pretrained(
                self.settings.mt_model,
                dtype=dtype,
                device_map="cuda:0" if self.settings.on_cuda else "cpu",
                low_cpu_mem_usage=True,
            )
            self._model.eval()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            raise
        self.load_seconds = time.perf_counter() - started
        log.info("MT ready: qwen_hf model=%s load=%.2fs", self.settings.mt_model, self.load_seconds)

    @property
    def ready(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    def translate_batch(self, items: list[tuple[str, str, str]]) -> list[MtResult]:
        import torch

        if not self.ready:
            raise RuntimeError("MT model not loaded")
        if not items:
            return []

        texts = []
        for text, src, dst in items:
            messages = make_translation_messages(text, src, dst)
            texts.append(
                self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )

        self._tokenizer.padding_side = "left"
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        encoded = self._tokenizer(texts, return_tensors="pt", padding=True).to(
            self._model.device
        )
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self._model.generate(
                **encoded,
                max_new_tokens=self.settings.mt_max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 2)

        prompt_len = encoded["input_ids"].shape[1]
        results: list[MtResult] = []
        for (text, _s, _d), row in zip(items, generated):
            new_tokens = row[prompt_len:]
            decoded = clean_translation(
                self._tokenizer.decode(new_tokens, skip_special_tokens=True),
                target_lang=_d,
            )
            hollow, reason = _hollow_check(decoded, text)
            results.append(
                MtResult(
                    text=decoded,
                    mt_ms=elapsed_ms,
                    backend=self.name,
                    model=self.settings.mt_model,
                    input_tokens=int(prompt_len),
                    output_tokens=int((new_tokens != self._tokenizer.pad_token_id).sum()),
                    batch_size=len(items),
                    hollow=hollow,
                    hollow_reason=reason,
                )
            )
        return results

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine": "transformers",
            "model": self.settings.mt_model,
            "device": self.settings.device,
            "max_new_tokens": self.settings.mt_max_new_tokens,
            "load_seconds": round(self.load_seconds, 2) if self.load_seconds else None,
            "ready": self.ready,
            "error": self.error,
            "note": "no continuous batching: throughput well below the vllm backend",
        }


def _vllm_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("vllm") is not None


# Official pre-quantised Qwen2.5 checkpoints. VERIFIED reachable and
# Apache-2.0/ungated via the HF API at the time of writing; sizes are the real
# model.safetensors Content-Length, not estimates.
_AWQ_REPOS: Final[dict[str, str]] = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",  # 1.614 GB
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
    "Qwen/Qwen2.5-3B-Instruct": "Qwen/Qwen2.5-3B-Instruct-AWQ",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct-AWQ",
}
_GPTQ_REPOS: Final[dict[str, str]] = {
    "Qwen/Qwen2.5-1.5B-Instruct": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-0.5B-Instruct": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
}


def _resolve_awq_model(model: str, quant: str) -> tuple[str, str | None]:
    """Map (model, quant) to the checkpoint that can actually serve it.

    This exists because of a failure mode that is easy to ship and hard to
    read: vLLM's `quantization="awq"` does not quantise anything at load time,
    it *expects weights that are already AWQ*. Pointing it at the unquantised
    base repo raises deep inside the weight loader, and the traceback names a
    missing `qweight` tensor rather than saying "you asked for AWQ but gave me
    fp16 weights".

    So an int4/awq request is redirected to the official `-AWQ` repository, and
    the substitution is returned so /health and /capabilities can state which
    checkpoint is really loaded. A server that reports the model you asked for
    while serving a different one is worse than one that refuses.

    If the model is already a quantised repo, or has no known AWQ sibling, it
    is passed through untouched: guessing a repository name that may not exist
    would trade a clear error for a confusing 404.
    """
    q = quant.strip().lower()
    if q not in {"awq", "int4", "awq_marlin", "gptq", "gptq_marlin"}:
        return model, None
    upper = model.upper()
    if "AWQ" in upper or "GPTQ" in upper or "INT4" in upper:
        return model, None  # already a quantised checkpoint
    table = _GPTQ_REPOS if q.startswith("gptq") else _AWQ_REPOS
    target = table.get(model)
    if target is None:
        return model, (
            f"MT_QUANT={quant} needs a pre-quantised checkpoint, but no known "
            f"quantised sibling of {model!r} is on record. Loading it as given: "
            f"if the weights are not already quantised this will fail at load. "
            f"Set MT_MODEL to a quantised repository explicitly."
        )
    return target, (
        f"MT_QUANT={quant} requires pre-quantised weights, so MT_MODEL was "
        f"redirected from {model!r} to {target!r}. vLLM cannot quantise fp16 "
        f"weights at load time."
    )


def _discover_ct2_model(settings: Settings) -> str | None:
    """Find a local CT2 M2M100 directory. Explicit setting wins."""
    import os

    candidates: list[str] = []
    if settings.mt_cpu_model_path:
        candidates.append(settings.mt_cpu_model_path)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates += [
        os.path.join(here, "models", "m2m100_418M-ct2-int8"),
        os.path.join(os.path.dirname(here), "models", "ct2", "m2m100_418M-ct2-int8"),
        os.path.expanduser("~/convert/m2m100_418M-ct2-int8-selfconverted"),
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "model.bin")):
            return path
    return None


def build_mt_engine(settings: Settings) -> MtEngine:
    """Pick a backend. Explicit MT_BACKEND is honoured even if it then fails:
    a silent downgrade would make the reported model a lie."""
    backend = settings.mt_backend.strip().lower()

    if backend == "qwen_vllm":
        return QwenVllmEngine(settings)
    if backend == "qwen_hf":
        return QwenHfEngine(settings)
    if backend == "m2m100_ct2":
        path = _discover_ct2_model(settings)
        if path is None:
            raise FileNotFoundError(
                "MT_BACKEND=m2m100_ct2 but no CT2 model found; set MT_CPU_MODEL_PATH"
            )
        return M2M100Ct2Engine(settings, path)

    if backend not in {"auto", ""}:
        raise ValueError(f"unknown MT_BACKEND: {settings.mt_backend}")

    # auto: on a GPU prefer vLLM, else the local NMT model, else HF Qwen.
    if settings.on_cuda and _vllm_available():
        return QwenVllmEngine(settings)
    path = _discover_ct2_model(settings)
    if path is not None:
        return M2M100Ct2Engine(settings, path)
    if settings.on_cuda:
        return QwenHfEngine(settings)
    raise FileNotFoundError(
        "no usable MT backend: vLLM absent, no local CT2 model found, and running "
        "Qwen on CPU transformers would not meet any latency target. "
        "Set MT_CPU_MODEL_PATH to a CTranslate2 M2M100 directory, or MT_BACKEND=qwen_hf "
        "to force the slow portable path."
    )
