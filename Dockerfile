# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Prevent Python from writing .pyc and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only wheel FIRST to prevent downloading 3.5GB+ of NVIDIA CUDA packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, configurations, data, and reports
COPY src/ ./src/
COPY app/ ./app/
COPY config/ ./config/
COPY data/processed/ ./data/processed/
COPY data/evaluation/ ./data/evaluation/
COPY reports/ ./reports/
COPY .env.example ./.env

EXPOSE 8000

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
