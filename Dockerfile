# syntax=docker/dockerfile:1.7

# ---------------------------------------------------------------------------
# Stage 1: build a self-contained virtualenv
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements.txt .
# CPU-only torch keeps the image ~2 GB smaller; drop the index URL for CUDA images.
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.2" \
 && pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .

# Bake the processor (and base weights, unless a fine-tuned model is mounted) into the image.
ARG PRELOAD_MODEL=microsoft/layoutlmv3-base
ENV HF_HOME=/opt/hf
RUN python -c "from transformers import LayoutLMv3Processor, LayoutLMv3Model as M; \
LayoutLMv3Processor.from_pretrained('${PRELOAD_MODEL}', apply_ocr=False); M.from_pretrained('${PRELOAD_MODEL}')"

# ---------------------------------------------------------------------------
# Stage 2: slim runtime with Tesseract and Poppler
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng poppler-utils libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 docai
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder --chown=docai:docai /opt/hf /opt/hf

ENV PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    DOCAI_UPLOAD_DIR=/data/uploads

RUN mkdir -p /data/uploads && chown -R docai:docai /data
USER docai
WORKDIR /home/docai
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)" || exit 1

CMD ["uvicorn", "docai.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
