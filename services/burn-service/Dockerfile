FROM python:3.12-slim
# Connexions lentes : délai et relances élevés ; le cache pip survit aux builds ratés
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10
WORKDIR /srv
COPY services/burn-service/requirements.txt /tmp/req.txt
# torch ET torchvision depuis le même index CPU
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/req.txt
COPY ml/__init__.py ml/__init__.py
COPY ml/common ml/common
COPY ml/burn_torch ml/burn_torch
COPY services/burn-service/app app
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
