FROM python:3.12-slim
ENV PIP_DEFAULT_TIMEOUT=300 PIP_RETRIES=10 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY services/api/requirements.txt /tmp/req.txt
RUN --mount=type=cache,target=/root/.cache/pip pip install -r /tmp/req.txt
COPY services/api/app app
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/api/v1/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
