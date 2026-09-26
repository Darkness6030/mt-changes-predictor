# ML module: HTTP inference service by default, offline CLI on demand.
FROM python:3.12-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.lock
COPY src ./src
RUN pip install --no-cache-dir --no-deps . \
    && useradd --create-home --uid 10001 ml \
    && mkdir -p /app/artifacts && chown ml:ml /app/artifacts
USER ml

ENV ML_MODEL_DIR=/app/model \
    ML_PORT=8001 \
    ML_HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1
EXPOSE 8001

# The model directory is mounted read-only, so the image contains no data and no artifact.
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=5 \
    CMD python -c "import urllib.request,os,sys; \
url='http://127.0.0.1:'+os.environ.get('ML_PORT','8001')+'/health/ready'; \
sys.exit(0 if urllib.request.urlopen(url, timeout=2).status == 200 else 1)"

CMD ["transport-ml-serve"]
