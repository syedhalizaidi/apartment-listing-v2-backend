# Stage 1: Builder stage for creating wheels
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create wheels directory
WORKDIR /wheels

# Copy requirements (without heavy libs)
COPY requirements.txt .
# Optional: create a "light-requirements.txt" without sentence-transformers/chromadb
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# Stage 2: Final image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies from wheels
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt

# Clean up
RUN rm -rf /wheels \
    && find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + \
    && find /opt/venv -type d -name 'tests' -exec rm -rf {} + \
    && find /opt/venv -type d -name 'test' -exec rm -rf {} + \
    && find /opt/venv -name '*.pyc' -delete

# Create app directory and switch to non-root user
WORKDIR /app
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Copy application code
COPY --chown=appuser:appuser . .

# Expose port and set healthcheck
EXPOSE 5000
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -fsS http://localhost:5000/health || exit 1

# Run the application
CMD ["python", "run.py"]
