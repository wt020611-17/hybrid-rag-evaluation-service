FROM python:3.11.9-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/.cache \
    USE_TF=0 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY data ./data
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0 \
    && pip install ".[production]"

COPY models/bge-small-zh-v1.5 /models/bge-small-zh-v1.5

RUN useradd --create-home --uid 10001 appuser && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000
CMD ["uvicorn", "hybrid_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]

