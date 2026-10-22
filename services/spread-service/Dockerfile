# Contexte de build : racine du dépôt (le code ml/ est partagé avec l'entraînement)
FROM python:3.12-slim AS base
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /srv
COPY services/spread-service/requirements.txt /tmp/req.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/req.txt
COPY ml/__init__.py ml/__init__.py
COPY ml/common ml/common
COPY ml/spread_tf ml/spread_tf
COPY services/spread-service/app app
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
