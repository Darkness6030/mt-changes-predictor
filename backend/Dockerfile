# Backend module: NDTP ingestion, causal features and the dispatcher API.
# Build context is the repository root, because the shared feature library lives in ml/.
FROM python:3.12-slim

WORKDIR /app
COPY backend/requirements.lock ./requirements.lock
RUN pip install --no-cache-dir -r requirements.lock
# The shared feature implementation is installed without CatBoost: inference is a separate
# service, so the Backend image stays small and cannot train or load a model itself.
COPY ml/pyproject.toml ./ml/pyproject.toml
COPY ml/src ./ml/src
RUN pip install --no-cache-dir --no-deps ./ml
COPY backend/pyproject.toml ./backend/pyproject.toml
COPY backend/src ./backend/src
RUN pip install --no-cache-dir --no-deps ./backend \
    && useradd --create-home --uid 10002 backend
USER backend

ENV BACKEND_HOST=0.0.0.0 \
    BACKEND_PORT=8000 \
    BACKEND_DATA_ROOT=/app/dataset \
    BACKEND_ML_URL=http://ml:8001 \
    PYTHONUNBUFFERED=1
EXPOSE 8000 9201

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 \
    CMD python -c "import urllib.request,os,sys; \
url='http://127.0.0.1:'+os.environ.get('BACKEND_PORT','8000')+'/health/ready'; \
sys.exit(0 if urllib.request.urlopen(url, timeout=2).status == 200 else 1)"

CMD ["transport-backend", "serve"]
