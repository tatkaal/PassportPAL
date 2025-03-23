# Builder stage
FROM python:3.10.16-slim as builder

WORKDIR /app

# Install only essential build dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY backend/requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# Install CPU-only PyTorch using requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} +

# Final stage
FROM python:3.10.16-slim

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# Create directory structure
RUN mkdir -p /app/models /app/frontend/public/samples

# Copy application code (only what's needed)
COPY backend/main.py backend/dataset.py ./
COPY backend/models/ /app/models/

# Set environment variables for performance and CPU-only mode
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV PYTHONDONTWRITEBYTECODE=1
# Explicitly disable CUDA
ENV CUDA_VISIBLE_DEVICES="-1"
ENV TORCH_CUDA_ARCH_LIST="None"

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 5000

# Run with uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port 5000
