FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Сначала зависимости: этот слой кэшируется, пока не меняется pyproject.toml
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY configs ./configs
COPY models ./models

RUN useradd --create-home appuser
USER appuser

ENV MODEL_PATH=models/model.joblib
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "mlproject.api:app", "--host", "0.0.0.0", "--port", "8000"]
