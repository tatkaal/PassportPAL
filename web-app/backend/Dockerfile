# Builder stage
FROM python:3.12.9-slim as builder

WORKDIR /app

# Install only essential build dependencies 
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt && \
    find /opt/venv -name "*.pyc" -delete && \
    find /opt/venv -name "__pycache__" -type d -exec rm -rf {} +

# Final stage
FROM python:3.12.9-slim

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install only runtime dependencies needed for OpenCV and network tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/* /var/tmp/*

# Create directory structure
RUN mkdir -p /app/models /app/frontend/public/samples

# Copy application code (only what's needed)
COPY main.py dataset.py download_models.sh ./
COPY models/ /app/models/

# Add GPU support detection code
RUN echo '#!/bin/bash\n\
echo "Checking for GPU..."\n\
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then\n\
  echo "No GPU available, using CPU"\n\
else\n\
  python -c "import torch; print(f\"GPU available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}, Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '\''None'\''}\")"\n\
fi' > /app/check_gpu.sh && chmod +x /app/check_gpu.sh

# Set proper environment variables for PyTorch
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV VECLIB_MAXIMUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create a non-root user and switch to it
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Expose API port
EXPOSE 5000

# Run with uvicorn
CMD /app/check_gpu.sh && uvicorn main:app --host 0.0.0.0 --port 5000
