FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY services/burn-service/requirements.txt /tmp/req.txt
# Roue PyTorch CPU : image ~4x plus légère que la version CUDA
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r /tmp/req.txt
COPY ml/__init__.py ml/__init__.py
COPY ml/common ml/common
COPY ml/burn_torch ml/burn_torch
COPY services/burn-service/app app
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
