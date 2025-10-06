# Coordinator Service Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
COPY ../requirements.txt ./shared-requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .
COPY ../proto ./proto
COPY ../config ./config

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose gRPC port
EXPOSE 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import grpc; import sys; \
    try: \
        channel = grpc.insecure_channel('localhost:50051'); \
        grpc.channel_ready_future(channel).result(timeout=5); \
        print('OK'); \
    except: \
        sys.exit(1)"

# Run the application
CMD ["python", "main.py"]