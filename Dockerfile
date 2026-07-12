# Multi-stage build for Redubber v2.0
# Stage 1: Build React frontend with Vite
# Stage 2: Python runtime with FastAPI backend

# =============================================================================
# Stage 1: Frontend Builder
# =============================================================================
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files and install dependencies
COPY frontend/package*.json ./
RUN npm ci --prefer-offline --no-audit

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# Verify build output
RUN ls -la dist/ && echo "Frontend build complete"

# =============================================================================
# Stage 2: Python Runtime
# =============================================================================
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (ffmpeg for video processing)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry for dependency management
RUN pip install --no-cache-dir poetry==1.8.0

# Configure Poetry to not create virtual environments (use system Python)
RUN poetry config virtualenvs.create false

# Copy Python dependency files
COPY pyproject.toml poetry.lock ./

# Install Python dependencies (production only)
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy application code
COPY app/ ./app/
COPY redubber.py database.py file_scanner.py video_analyzer.py ./
COPY reproj.py seg_postprocessor.py utils.py pipeline_status.py ./

# Copy frontend build artifacts from stage 1
COPY --from=frontend-builder /app/frontend/dist ./app/static

# Create required directories with proper permissions
RUN mkdir -p /mounted-storage && chmod 755 /mounted-storage

# Expose FastAPI port
EXPOSE 8000

# Health check using the FastAPI health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Run application using uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
