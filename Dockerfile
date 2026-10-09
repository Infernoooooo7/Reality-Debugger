# Single-service image: the FastAPI backend also serves the built frontend,
# so the whole app runs behind one URL (used for the Render deployment).
#
#   docker build -t reality-debugger .
#   docker run -p 8000:8000 --env-file .env reality-debugger
#
# No API key is needed: detection, tracking and diagnostics run locally (in
# the browser and the backend's rule engine), and /api/vision runs YOLOX and
# the reference-comparison model on the server's CPU. GEMINI_API_KEY (or an
# explicit AI_PROVIDER=claude + ANTHROPIC_API_KEY) only adds the optional AI layer.

# Model weights. The fast detector is committed; the deep detector (YOLOX-S,
# 36 MB) is downloaded and verified against the SHA-256 in models/registry/.
# The anomaly feature extractor is cut from the registered ResNet-50 (also
# checksum-verified) and must reproduce its recorded SHA-256 exactly.
FROM python:3.13-slim AS models
WORKDIR /src
RUN pip install --no-cache-dir onnx==1.23.2
COPY tools/fetch_models.py tools/build_feature_extractor.py tools/
COPY models/registry/ models/registry/
COPY frontend/public/models/ frontend/public/models/
RUN python tools/fetch_models.py --fetch \
 && python tools/build_feature_extractor.py --strict \
 && rm -f models/weights/resnet50-v1-12.onnx

FROM node:22-slim AS frontend
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
# Shared parameter files (<repo>/config) are imported by the frontend build.
COPY config/ /app/config/
COPY frontend/ ./
COPY --from=models /src/frontend/public/models/ ./public/models/
RUN npm run build

FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    FRONTEND_DIST=/app/frontend/dist
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install -r requirements.txt
# The backend reads the same parameter files from <repo>/config (/app/config)
# and the model registry from /app/models/registry.
COPY config/ /app/config/
COPY models/registry/ /app/models/registry/
COPY --from=models /src/models/weights/resnet50_v1_patch_features.onnx /app/models/weights/
COPY backend/app ./app
COPY --from=frontend /app/frontend/dist /app/frontend/dist
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
# Hosting platforms (Render, Fly, Railway...) pass the port in $PORT.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
