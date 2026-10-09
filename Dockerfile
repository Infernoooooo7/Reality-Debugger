# Single-service image: the FastAPI backend also serves the built frontend,
# so the whole app runs behind one URL (used for the Render deployment).
#
#   docker build -t reality-debugger .
#   docker run -p 8000:8000 --env-file .env reality-debugger
#
# Without ANTHROPIC_API_KEY (or OPENAI_* settings) it runs in DEMO MODE.

FROM node:22-slim AS frontend
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
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
COPY backend/app ./app
COPY --from=frontend /app/frontend/dist /app/frontend/dist
RUN useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
# Hosting platforms (Render, Fly, Railway...) pass the port in $PORT.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips '*'"]
