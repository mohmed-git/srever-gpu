FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    HF_HOME=/cache/huggingface \
    VLLM_USE_V1=0 \
    DEVICE=cuda

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt accelerate autoawq

# Build-time pre-conversion of Qwen2.5-1.5B-Instruct to CTranslate2 int8_float16
RUN ct2-transformers-converter --model Qwen/Qwen2.5-1.5B-Instruct --quantization int8_float16 --output_dir /models/qwen2.5-1.5b-ct2

ENV MT_MODEL=/models/qwen2.5-1.5b-ct2 \
    MT_TOKENIZER=Qwen/Qwen2.5-1.5B-Instruct \
    MT_BACKEND=qwen_ct2

COPY app ./app
COPY tests ./tests
COPY run.py .

EXPOSE 8080

CMD ["python3", "run.py"]

