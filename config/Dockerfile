# Multi-stage Docker build for Enterprise Document Processing API
# Optimized for production deployment with security and performance

# =============================================================================
# Build Stage - Compile dependencies and prepare application
# =============================================================================
FROM python:3.11-slim as builder

LABEL maintainer="Enterprise Document Processing Team"
LABEL description="Enterprise-grade document processing API with multi-tenant architecture"
LABEL version="1.0.0"

# Build arguments
ARG ENVIRONMENT=production
ARG BUILD_DATE
ARG GIT_COMMIT
ARG VERSION=1.0.0

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    pkg-config \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create build user
RUN groupadd -r builduser && useradd -r -g builduser builduser

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn uvicorn[standard]

# Copy application code
COPY . .

# Remove dev dependencies and unnecessary files
RUN pip uninstall -y -r requirements-dev.txt || true && \
    find . -type f -name "*.pyc" -delete && \
    find . -type d -name "__pycache__" -delete && \
    rm -rf .git .github docs tests *.md requirements-dev.txt

# =============================================================================
# Runtime Stage - Minimal production image
# =============================================================================
FROM python:3.11-slim as runtime

# Runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    PORT=8000 \
    WORKERS=4 \
    LOG_LEVEL=info \
    TIMEOUT=120 \
    KEEPALIVE=5

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    libpq5 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create application user
RUN groupadd -r appuser && \
    useradd -r -g appuser -d /app -s /bin/bash appuser

# Create necessary directories
RUN mkdir -p /app/data/uploads /app/data/processing /app/data/archive /app/logs && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code from builder
COPY --from=builder --chown=appuser:appuser /app .

# Create startup script
COPY --chown=appuser:appuser <<EOF /app/start.sh
#!/bin/bash
set -e

# Wait for database if needed
if [ -n "\$DATABASE_URL" ]; then
    echo "Waiting for database..."
    python -c "
import asyncio
import sys
from api.database.session import test_database_connection

async def wait_for_db():
    max_retries = 30
    for i in range(max_retries):
        try:
            if await test_database_connection():
                print('Database is ready!')
                return
        except Exception as e:
            print(f'Database not ready ({i+1}/{max_retries}): {e}')
            if i < max_retries - 1:
                await asyncio.sleep(2)
            else:
                print('Database connection failed after all retries')
                sys.exit(1)

asyncio.run(wait_for_db())
"
fi

# Run database migrations if needed
echo "Running database migrations..."
# python -m alembic upgrade head || echo "No migrations to run"

# Start the application
echo "Starting Enterprise Document Processing API..."
exec gunicorn api.main:app \
    --bind 0.0.0.0:\${PORT} \
    --workers \${WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout \${TIMEOUT} \
    --keepalive \${KEEPALIVE} \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --log-level \${LOG_LEVEL} \
    --access-logfile - \
    --error-logfile - \
    --log-config logging.conf
EOF

# Make startup script executable
RUN chmod +x /app/start.sh

# Create logging configuration
COPY --chown=appuser:appuser <<EOF /app/logging.conf
[loggers]
keys=root,gunicorn.error,gunicorn.access,uvicorn.error,uvicorn.access

[handlers]
keys=console

[formatters]
keys=generic,access

[logger_root]
level=INFO
handlers=console

[logger_gunicorn.error]
level=INFO
handlers=console
propagate=1
qualname=gunicorn.error

[logger_gunicorn.access]
level=INFO
handlers=console
propagate=0
qualname=gunicorn.access

[logger_uvicorn.error]
level=INFO
handlers=console
propagate=1
qualname=uvicorn.error

[logger_uvicorn.access]
level=INFO
handlers=console
propagate=0
qualname=uvicorn.access

[handler_console]
class=StreamHandler
formatter=generic
args=(sys.stdout, )

[formatter_generic]
format=%(asctime)s [%(process)d] [%(levelname)s] %(message)s
datefmt=%Y-%m-%d %H:%M:%S
class=logging.Formatter

[formatter_access]
format=%(message)s
class=logging.Formatter
EOF

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Expose port
EXPOSE ${PORT}

# Labels for metadata
LABEL build.date="${BUILD_DATE}" \
      build.version="${VERSION}" \
      build.commit="${GIT_COMMIT}" \
      build.environment="${ENVIRONMENT}"

# Set entrypoint
ENTRYPOINT ["/app/start.sh"]

# =============================================================================
# Development Stage (optional) - For development with hot reload
# =============================================================================
FROM runtime as development

USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Override entrypoint for development
COPY --chown=appuser:appuser <<EOF /app/dev-start.sh
#!/bin/bash
set -e

echo "Starting development server with hot reload..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port \${PORT:-8000} \
    --reload \
    --log-level debug \
    --access-log
EOF

RUN chmod +x /app/dev-start.sh

USER appuser

ENTRYPOINT ["/app/dev-start.sh"]