# Lingua Buds — Ultra-Low-Latency Translation Server

A GPU translation backend for the Lingua Buds earbuds: streaming ASR + MT behind
one WebSocket and one REST endpoint, with admission control, real latency
percentiles, and a hard rule that no number in this document is stated more
confidently than the evidence behind it.

---

## READ THIS FIRST: the two headline targets

The brief set two targets. One is achievable in specific configurations. The
other is not achievable on the specified hardware, and no amount of engineering
in this repository changes that. Both answers are below with the arithmetic.

### 1. `< 150 ms` server latency per request — **conditionally achievable**

Reachable **only** in specific model/quantization combinations, and **not** at
the configuration the brief named (`whisper-large-v3-turbo` + `Qwen2.5-1.5B`
at `fp16`).

The binding constraint is memory bandwidth, not compute. An autoregressive
decode step must read every weight once per generated token, so:

```
max tokens/sec  =  memory_bandwidth / weight_bytes
```

RTX 3060 12GB has **360 GB/s** (192-bit GDDR6). Measured median output length
for a spoken sentence in this build is **8 tokens** (max 16), using the real
Qwen tokenizer:

The weight sizes below are **MEASURED**, not estimated: the `safetensors`
header of each real checkpoint was read over HTTP range requests and the
per-tensor byte offsets summed. Nothing was downloaded.

| MT config | weights (measured) | ceiling tok/s | ms @ 8 ideal | ms @ 8 real (70 %) | meets 20–27 ms? |
|---|---|---|---|---|---|
| Qwen2.5-1.5B **fp16** | 3.087 GB | 116.6 | 68.6 ms | **98.0 ms** | ✗ no |
| Qwen2.5-1.5B **AWQ int4** (as shipped) | 1.614 GB | 223.0 | 35.9 ms | **51.2 ms** | ✗ no |
| Qwen2.5-1.5B **AWQ int4**, tied embeddings | 1.147 GB | 313.9 | 25.5 ms | **36.4 ms** | ✗ no |

#### Correction: an earlier version of this file was wrong

It claimed int4 was **0.72 GB** and **16.0 ms**. That was wrong. It quantised
the whole parameter count and ignored that AWQ leaves the embedding and
`lm_head` matrices in **fp16**. Reading the actual header, 731 tensors:

| part | size | note |
|---|---|---|
| `qweight` (int4 packed) | 0.655 GB | 196 tensors — the only quantised part |
| `lm_head` | 0.467 GB | **fp16** |
| `embed_tokens` | 0.467 GB | **fp16** |
| `scales` + `qzeros` + norms | 0.025 GB | |
| **total** | **1.614 GB** | 2.2× the retracted figure |

The two fp16 embedding copies (0.934 GB) **outweigh the quantised linear
layers** (0.655 GB). `config.json` says `tie_word_embeddings: true`, yet the
checkpoint ships `lm_head.weight` as a separate tensor anyway — trusting that
flag instead of the header is exactly how the wrong figure survived three
rounds of arithmetic.

#### What this means for the 20–27 ms target

**It is not reachable** on an RTX 3060 with this model at any quantisation.
The best imaginable case — dropping the duplicate embedding copy — is 36.4 ms,
and the shipped configuration is **~51 ms**.

`MT_QUANT` still defaults to **`awq`**, because it is the right default for
reasons that survive the correction: ~1.9× faster than fp16, 1.47 GB lighter,
and ~51 ms leaves real room inside the **150 ms whole-request** budget. It is
simply not the 20–27 ms that was asked for, and it is not sold as such.

**Status of the ms columns: MODELLED.** No CUDA device was available to this
build. The GB column is measured; the ms columns are a bandwidth ceiling, not
a benchmark, and must not be quoted as results.

### 2. `100+` concurrent users under 150 ms — **not reachable as specified**

This is queueing theory, not pessimism. If arrival rate × service time ≥ 1, the
queue grows without bound: latency **diverges**, it does not settle at 150 ms.

Taking the brief's own best-case service time of 127.5 ms → capacity 7.8 req/s:

| 100 users, one turn every… | demand | utilisation ρ |
|---|---|---|
| 10 s | 10.0 req/s | **1.28×** |
| 6 s | 16.7 req/s | **2.14×** |
| 3 s | 33.3 req/s | **4.25×** |

Every one of those is ρ ≥ 1. Batching raises throughput but spends latency to
buy it (MODELLED): **B=8** → 30.6 req/s at **p50 392 ms**; **B=32** →
44.4 req/s at **p50 1081 ms**. Batching reaches the required throughput — but
never at 150 ms.

**What this server does about it:** it refuses work it cannot finish inside the
budget (`503` + `Retry-After`) so the client degrades **visibly** instead of
silently. See [Admission control](#admission-control).

**What would actually be needed:** either 2–4 GPUs at the brief's per-request
budget, or a smaller MT model at int4 plus a relaxed budget, or a lower per-user
turn rate. Pick one deliberately; do not assume the number.

### VRAM: the brief's budget does not close

The brief allocated 1.5–2.0 GB for the whole MT stage. Qwen2.5-1.5B **fp16
weights alone are 3.087 GB (measured)** — over budget before KV cache,
activations, or the ~0.3–0.6 GB CUDA context.

| model | fp16 | AWQ int4 | how |
|---|---|---|---|
| Qwen2.5-1.5B | **3.087 GB** | **1.614 GB** | **MEASURED** from safetensors headers |
| Qwen2.5-0.5B | 0.91 GB | — | MODELLED |
| whisper-large-v3-turbo | 1.51 GB | 0.75 GB (int8) | MODELLED |
| whisper-large-v3 | 2.89 GB | — | MODELLED |

AWQ at 1.614 GB **fits the 1.5–2.0 GB budget, but only just** — and that is
weights only. Note the two figures for Qwen2.5-1.5B are measured; the rest are
still parameter-count arithmetic, which is exactly the method that produced the
retracted 0.72 GB above. Treat the MODELLED rows with the same suspicion.

---

## Measured vs modelled — the whole split in one table

| Claim | Status | Where measured |
|---|---|---|
| Language count = **100** | **MEASURED** | read from `faster_whisper.tokenizer._LANGUAGE_CODES` at import; 0 missing / 0 extra vs the name table |
| ASR latency on real speech (CPU) | **MEASURED** | `asr=2123.47 ms`, tiny/int8/2 cores, 3.00 s real speech |
| MT latency on real text (CPU) | **MEASURED** | `mt=1327.16 ms` en→ar; direct REST 766.51 ms (en→ar), 847.59 (ar→en), 518.93 (en→ja) |
| Silence costs ~20 ms and is flagged | **MEASURED** | 22.64 ms, `hollow=true` |
| Blank input makes MT **hallucinate** | **MEASURED** | `"  "` → `'مـنـهـا'` (7 tokens) — see [the hollow rule](#the-hollow-rule) |
| Magic bytes beat the declared format | **MEASURED** | real libopus OGG declared `pcm` → decoded `container:opus`, 3.00 s |
| Batching, admission, shedding | **MEASURED** | `tests/check_scheduler.py`, 5/5 pass |
| Load behaviour under 16-way concurrency | **MEASURED** | 48/48 served, 0 shed, 0 errors, 0 empty; p50 3571.3 ms |
| Median MT output = 8 tokens | **MEASURED** | real Qwen tokenizer |
| **Every GPU latency figure** | **MODELLED** | no CUDA device in the build environment |
| **Every VRAM figure** | **MODELLED** | parameter arithmetic |
| **Throughput at B=8 / B=32** | **MODELLED** | queueing arithmetic |
| **The Docker image builds** | **UNVERIFIED** | `docker` is not installed here; only the Dockerfile's structure and its healthcheck pattern were checked |

Verification hardware: Intel Xeon @ 2.50 GHz, **2 cores, 985 MB RAM**,
2.4 GB free disk, `torch 2.13.0+cpu`, **no CUDA**, `faster-whisper 1.2.1`,
`ctranslate2 4.8.1`, Python 3.13.

Because the box has no GPU and only 2.4 GB free disk, Qwen2.5-1.5B weights
(~3 GB) **could not be downloaded** — the HF cache holds its tokenizer/config
only (11 MB). So the code path is verified end-to-end with **whisper-tiny +
M2M100-418M-CT2-int8**, and the *same* code selects vLLM + Qwen on the 3060.
The backend factory is the portability seam; nothing else changes.

---

## Report card

| | |
|---|---|
| **Current model** | ASR `whisper-tiny` (CPU verify) → `large-v3-turbo` on CUDA; MT `M2M100-418M-CT2-int8` (CPU verify) → `Qwen2.5-1.5B-Instruct` on CUDA |
| **Previous model** | n/a — new subsystem, does not touch the existing app |
| **Backend** | faster-whisper (CTranslate2) + vLLM / CTranslate2 / HF-transformers, behind FastAPI + Uvicorn |
| **Quantization** | ASR `int8` (CPU) / `float16` (CUDA); MT **`awq` (int4) default** — ~1.9× faster than fp16 and 1.47 GB lighter, but **not** the 20–27 ms requested (real ~51 ms) |
| **Model size** | 75 MB + 477 MB (CPU verify) · ~1.5 GB ASR + **1.614 GB MT (AWQ, MEASURED)** on CUDA |
| **RAM usage** | reported live in `/metrics` → `resources.cpu`; VRAM via torch/pynvml, else `available:false` **with a reason** |
| **CPU** | 2 cores in verification; `ASR_CPU_THREADS=0` uses all |
| **Median latency** | **MEASURED CPU:** p50 **751.79 ms** (e2e, n=12) · under 16-way load p50 **3574.9 ms**, p95 **3679.4 ms**, p99 withheld (41 < 100) · **within budget 0/41 (0.0 %) — honest, not a defect** |
| **Time to first word** | **MEASURED, n=5 each, same 5.39 s audio:** streaming **1587 ms** vs unified **2228 ms** → **641 ms sooner**. Total server time went the other way: 2373 ms vs 2228 ms, i.e. **145 ms slower overall** |
| **Supported languages** | **100**, read from the engine's own table so it cannot drift |
| **Files changed** | none — the existing app is untouched |
| **Files added** | 23 (below) |
| **How to run** | `python run.py` — or `docker build -t lingua-server:gpu . && docker run --gpus all -p 8080:8080 --env-file .env lingua-server:gpu` |
| **How to test** | `python tests/check_mt_guard.py` · `check_scheduler.py` · `check_server.py` · `load_test.py <url> <conc> <n>` |

---

## Quick start

```bash
cd gpu_server
pip install -r requirements.txt      # add: pip install vllm>=0.6.0  on the 3060
cp .env.example .env                 # every knob is documented in there
python run.py                        # one command
```

Docker:

```bash
docker build -t lingua-server:gpu .                  # default target = gpu
docker run --gpus all -p 8080:8080 \
  -v hf-cache:/cache/huggingface --env-file .env lingua-server:gpu

docker build -t lingua-server:cpu --target cpu .     # CI / verification
```

The GPU image pins `DEVICE=cuda` deliberately. Without it, a container started
without `--gpus all` would auto-detect CPU, come up **healthy**, and serve ~10×
slower — the single most likely silent misconfiguration of this image. Pinned,
it fails loudly at model load and `/health` reports `ready:false`.

> **⚠️ The Dockerfile has NOT been built.** `docker` is not installed in the
> verification sandbox, so neither image was ever produced. What *was* verified:
> both stages parse, the default target is `gpu` (last stage), all four `COPY`
> sources exist, both stages carry `EXPOSE`/`USER`/`CMD`/`HEALTHCHECK`, and the
> `HEALTHCHECK` grep pattern was tested against the live server's real `/health`
> output **including a negative control** proving it rejects `ready:false`.
> Everything else about the image — that the CUDA base resolves, that
> `pip install vllm` succeeds against it, that cuDNN satisfies CTranslate2 — is
> **unverified**. Expect to iterate on the first real build; do not treat it as
> tested.

---

## API

### `WS /ws/v1/translate-stream`

Send binary audio chunks (Opus / OGG / WebM / WAV / raw PCM 16 kHz mono) and
JSON control frames. On connect the server sends a `ready` frame documenting the
protocol. Control keys: `source`, `target`, `format`, `sample_rate`, `channels`,
`stream`, `action` (`flush` | `reset` | `close`). Unknown keys come back in
`ignored_keys` rather than being dropped in silence.

There are **two response shapes**, and the `ready` frame states which one this
connection will get in its `sentence_streaming` field. Send `{"stream": false}`
to force the single-JSON shape.

#### Shape A — sentence streaming (default)

One frame per sentence, then exactly one `final`:

```json
{"type":"sentence","index":0,"sentence_count":2,"is_last":false,
 "original_text":"Where is the nearest pharmacy?",
 "translated_text":"أين هي أقرب صيدلية؟",
 "elapsed_ms":1634.03,"mt_ms":733.14,"output_tokens":9}
{"type":"sentence","index":1,"sentence_count":2,"is_last":true, "...":"..."}
{"type":"final","original_text":"...","translated_text":"...",
 "asr_ms":874.21,"mt_ms":1502.89,"total_server_ms":2410.17,
 "sentence_count":2,"streamed":true,
 "first_sentence_ms":1634.03,"time_saved_to_first_word_ms":776.13}
```

Play each sentence as it arrives. **`asr_ms` and `total_server_ms` exist only
on the `final` frame** — a per-utterance total cannot exist before the last
sentence is done. A client that reads frame 1 as the finished answer will act
on a partial translation; that is not hypothetical, it is what broke the test
suite when this became the default, which is why the mode is announced up
front and why `{"stream": false}` exists.

Errors after the first frame have their own shape, because by then the earbuds
have **already started speaking**: `{"type":"error", "frames_sent":n,
"partial":true, "terminal":true, ...}`. "This request failed" and "this request
half happened" need different client behaviour.

**What streaming buys — MEASURED**, n=5 per mode, identical 5.39 s audio, CPU:

| | streaming | unified | delta |
|---|---|---|---|
| time to first word (client-measured) | **1587 ms** | 2228 ms | **−641 ms** ✓ |
| total time to last word | 2373 ms | 2228 ms | **+145 ms** ✗ |

Both rows are reported because either one alone is misleading. Streaming does
**not** make the server faster — `n` small MT calls carry more total overhead
than one large call. It makes the *first word arrive sooner*, which is the
number a person wearing earbuds actually feels. This is why
`first_sentence_ms` is its own field and is never folded into
`total_server_ms`.

Single-sentence utterances take the unsplit path: nothing to overlap, so
`time_saved_to_first_word_ms` is `0.0` and says so.

#### Shape B — one unified JSON (`{"stream": false}`)

```json
{
  "original_text":   "Where is the nearest pharmacy?",
  "translated_text": "أين أقرب صيدلية؟",
  "source_lang":     "en",
  "target_lang":     "ar",
  "asr_ms":          2123.47,
  "mt_ms":           1327.16,
  "total_server_ms": 3490.83
}
```

…plus diagnostics: `hollow`, `hollow_reason`, `within_budget`, `rtf`,
`detected_language`, `audio_format`, `unaccounted_ms`.

`source` may be omitted for auto-detection (verified: returns `detected=en`).
`format` is a *hint*: magic bytes win, because a client claiming `pcm` while
sending Opus would otherwise be transcribed as noise and produce a
plausible-looking latency for garbage.

### `POST /translate`

```bash
curl -s localhost:8080/translate -H 'content-type: application/json' \
  -d '{"text":"Where is the nearest pharmacy?","source":"en","target":"ar"}'
```

```json
{ "translation": "أين أقرب صيدلية؟", "latency_ms": 766.51, "...": "diagnostics" }
```

### Observability

| endpoint | purpose |
|---|---|
| `/health` | readiness, `startup_error`, per-engine state, warmup result, `language_count`, `pair_count`, live budget compliance |
| `/metrics` | per-stage percentiles with **withholding rules**, queue stats, `admission_mode`, CPU/GPU resources |
| `/languages` | the 100 codes + names + RTL flags, **with the WER caveat stated** |
| `/capabilities` | the concurrency verdict, batching trade-off, VRAM arithmetic, decode roofline, and the measured/modelled split — machine-readable |

---

## Three design rules, each forced by a measurement

### The hollow rule

A synthetic formant buzz makes Whisper return `segments=0, text=''` in **~20 ms**
— a beautiful latency that measures **rejection, not transcription**. Any
benchmark that averages such runs is reporting a fiction.

So every ASR/MT result carries `hollow: bool` + `hollow_reason`, every timed
test asserts non-empty output, and `tests/check_server.py` keeps the buzz as an
explicit **control** so the hollow path is *proved* to fire rather than assumed.
Verified: buzz → `hollow=True`; silence → `silent_audio`.

Then measurement contradicted an assumption I had not questioned: *"an engine
given nothing returns nothing."* Feeding the M2M100 CT2 engine `"  "` returned
**`'مـنـهـا'`** — a confident hallucination, 7 tokens. Untranslatable input must
be rejected **before** inference, so `is_translatable()` (a `\w` unicode search)
now gates it, and `tests/check_mt_guard.py` locks the regression.

### Admission control

`ADMISSION_ENABLED=1` shed load with `503` + `Retry-After` rather than let
latency diverge. But the first version of that check **broke the server**, and
the load test caught it: **46 of 48 requests rejected at `peak_queue_depth=1`**.

The cause was arithmetic, not a bug in the loop: measured service time was
1447 ms against a 150 ms budget, so *any* queue depth ≥ 1 projects over budget.
That is not load shedding — that is a server that refuses to work.

Admission therefore has two modes, and `/metrics` says which is live:

- **budget enforcement** — the device *can* meet the budget when idle; shed when
  projected wait exceeds `budget × OVERLOAD_FACTOR`.
- **queue-growth protection** — the budget is unreachable even at zero queue
  depth; shed only above `8×` the measured per-item time, and report
  `budget_unreachable_on_this_device: true` with the reason in plain text.

After the fix, measured on the restarted server: **48/48 served, 0 shed,
0 errors, 0 empty translations**, with `admission_mode` reporting
`"queue-growth protection: the budget is unreachable on this device, shedding
above 8x the 367 ms service time"`. The unit test still proves shedding
happens (16 shed / 4 served of 20).

The EWMA per-item time is `None` until a real batch completes, so admission
never refuses a request based on a guess — the first request is always admitted.

### Percentile hygiene

`p95` is **withheld below 20 samples**, `p99` below 100, returning `null` plus a
note (`"withheld: 12 samples < 20 needed"`). A p95 from 2 samples is the maximum
wearing a label.

`tests/load_test.py` then committed exactly that error — it printed
`p95 = 2662.0 ms` from 2 samples — and now enforces the same `_MIN_SAMPLES` rule
the server does. A tool that violates the standard it is auditing is worse than
no tool.

Related: `/metrics` never reports "unavailable" as **zero**. With no GPU,
`_gpu_memory()` returns `available:false` with
`reason: "no CUDA device visible … (not the same as 0 MB used)"`.

### Sentence splitting: two defects found by test, not by design

Splitting the transcript so sentence 1 can be spoken early is the one change
here that helps the *perceived* latency. Both of its bugs were found by the
test file, not by reading the code:

1. **Chinese never split.** The first regex required `\s+` after the
   terminator. Chinese and Japanese do not put a space after `。`, so
   `"这是第一句话。这是第二句话。"` came back as **one piece**. The claim to
   support those languages was hollow. Fixed with a separate `_CJK_SPLIT` rule.
2. **Quotations split mid-sentence.** `He said "stop!" and left.` broke at the
   `!` inside the quotes. Fixed with an `_inside_quotes()` parity check.

Then a third, subtler one I caught myself: `len()` is script-biased.
`"这是第一句话。"` is a *complete* sentence in 7 characters, so under
`min_chars=12` it was judged a fragment and merged into its neighbour —
silently disabling splitting for exactly the languages defect 1 was fixed to
support. `_length_units()` now weights CJK characters 2.5×.

And a fourth: the `reason` field always blamed `min_chars`, so an utterance
kept whole by an abbreviation reported an untrue reason. It now names the rule
that actually fired (`"abbreviation"`, `"quoted speech"`).

`tests/check_sentences.py` holds ~35 assertions across 6 scripts and locks all
four.

---

## Files added (23)

```
gpu_server/
├── Dockerfile                  # gpu (default) + cpu targets, DEVICE pinned -- NOT built (no docker here)
├── .dockerignore               # keeps a real .env and model weights out of image layers
├── .env.example                # every knob, each with the arithmetic behind its default
├── requirements.txt            # vLLM intentionally opt-in (2 GB CUDA-only wheel)
├── run.py                      # one command; single worker on purpose
├── README.md                   # this file
├── app/
│   ├── config.py               # frozen Settings, env-overridable, torch-free device detection
│   ├── languages.py            # 100 langs read from the engine + names/RTL/aliases
│   ├── audio.py                # magic-byte sniffing, resample, silence detect, always raises
│   ├── metrics.py              # withheld percentiles, budget report, "unavailable ≠ zero"
│   ├── asr.py                  # faster-whisper; no silent model downgrade; hollow detection
│   ├── mt.py                   # 3 backends + is_translatable() guard + preamble stripping
│   ├── scheduler.py            # batching, two-mode admission control, Overloaded
│   ├── sentences.py            # multi-script splitter; CJK + quote + abbreviation rules
│   ├── pipeline.py             # orchestration, passthrough, unaccounted_ms, coverage refusal
│   └── server.py               # FastAPI: WS + REST + health/metrics/languages/capabilities
└── tests/
    ├── check_mt_guard.py       # locks the 'مـنـهـا' hallucination regression
    ├── check_sentences.py      # ~35 assertions, 6 scripts; locks the 4 splitter defects
    ├── check_scheduler.py      # 5 behaviours incl. shedding + failure isolation
    ├── check_server.py         # e2e with the synthetic-buzz control + 5 negative cases
    └── load_test.py            # concurrent load, honest headline, percentile withholding
```

Single uvicorn worker is deliberate: two workers would each load their own
Whisper **and** Qwen, doubling VRAM for zero throughput gain. Concurrency comes
from the in-process async batching scheduler.

Cross-request ASR batches are timed **item by item** in `_run_asr_batch`,
because `BatchedInferencePipeline` batches *within* one utterance — reporting a
shared span across requests would understate per-request latency.

---

## Test results (last run, verbatim)

```
tests/check_sentences.py  ALL PASSED   (~35 assertions, 6 scripts, 4 locked defects)
tests/check_mt_guard.py   ALL PASSED   ('  ' → hollow 'مـنـهـا';  'Hello there.' → 'مرحبا هناك' 383.18 ms)
tests/check_scheduler.py  ALL PASSED   (5 behaviours incl. shedding + failure isolation)
tests/check_server.py     ALL PASSED   (buzz → hollow=True; opus-as-pcm → container:opus;
                                        both WS shapes; is_last on exactly the last frame)
tests/load_test.py        41 served, 7 shed, 0 errors, 0 empty | p50 3574.9 | budget 0/41
```

Sentence-streaming A/B, n=5 per mode, same 5.39 s audio:

```
STREAMING  first frame median 1586.5 ms  (range 1498.8-1604.4)   last 2373.3 ms
UNIFIED    first frame median 2228.0 ms  (range 2151.9-2301.2)   last 2228.0 ms
→ time-to-first-word  -641.5 ms   (the gain)
→ total time          +145.2 ms   (the cost, as designed)
```

The 7 shed requests are the correct behaviour, not a failure: they were refused
with an explicit `503` + `Retry-After` instead of being absorbed into a growing
queue. `0 errors, 0 empty` is the line that matters.

Note the audio in that A/B was `espeak-ng` synthetic speech, and ASR `tiny`
misheard "Hello there." as "The lone heir," — a comma, not a full stop — so the
splitter saw **2** sentences where 3 were spoken. Checked directly: the splitter
returns 3 on the correctly-transcribed text. The 2 came from ASR, not from the
splitter. On a better ASR model the saving would be larger, not smaller.

`0/48` inside budget on a 2-core CPU with no GPU is the correct and expected
measurement. `run.py` prints this before it happens, so nobody reads it as a
regression:

> *no CUDA device detected … a CPU cannot meet the 150 ms budget: expect
> `/health` to show `within_budget_fraction` near 0. That is the measurement,
> not a bug.*

---

## Next step on real hardware

Everything above marked **MODELLED** should be replaced with a measurement on
the 3060. Order of work, highest information first:

1. Run the GPU image, then `curl /metrics` — this converts the roofline estimate
   into an observation.
2. Sweep `MT_QUANT` int8 → int4/AWQ and record p50 per config. The roofline says
   int4 has the only comfortable headroom; confirm or refute it.
3. Run `load_test.py` at rising concurrency until `admission_mode` flips to
   budget enforcement, and record the concurrency at which p95 crosses 150 ms.
   **That number is the real per-GPU user capacity** — and it will be far below
   100.
4. Decide explicitly between more GPUs, a smaller MT model, or a relaxed budget.

Licensing: `whisper-large-v3-turbo` (MIT) and `Qwen2.5-*-Instruct` (Apache-2.0)
are both commercially usable; M2M100 (MIT) is used only for CPU verification.
