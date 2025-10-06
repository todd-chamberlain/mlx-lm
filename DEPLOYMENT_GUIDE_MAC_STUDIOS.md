# 🚀 Deployment Guide for Mac Studio 1 & 2

This guide will help you deploy the MLX Distributed Inference System across your two Mac Studios.

## 📋 Prerequisites

### On Both Mac Studios

1. **macOS with Apple Silicon** (M1/M2 Ultra)
2. **Python 3.9+** installed
3. **MLX installed**: `pip install mlx mlx-lm`
4. **Docker Desktop** (only needed on Mac Studio 1 for containerized services)
5. **Network connectivity** between the two machines

### Check Prerequisites

```bash
# Check Python
python3 --version

# Check MLX
python3 -c "import mlx.core; print('MLX OK')"

# Check Docker (Mac Studio 1 only)
docker --version
docker-compose --version
```

---

## 🏗️ Architecture Overview

```
Mac Studio 1 (Coordinator Host)
├── Docker Containers
│   ├── API Server (port 8000)
│   ├── Coordinator (port 50051)
│   ├── Redis (port 6379)
│   ├── Prometheus (port 9090)
│   ├── Grafana (port 3000)
│   └── Nginx Load Balancer (port 80)
└── Native MLX Worker 1

Mac Studio 2 (Worker Host)
└── Native MLX Worker 2
```

---

## 📦 Step 1: Setup on Mac Studio 1 (Coordinator Host)

### 1.1 Clone and Setup Project

```bash
# Clone the repository (if not already done)
cd ~/projects
git clone <your-repo-url> mlx-distributed-inference
cd mlx-distributed-inference

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 1.2 Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

**Edit `.env` with these values:**

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_KEY=your-secret-api-key-here

# Coordinator Configuration
COORDINATOR_HOST=0.0.0.0
COORDINATOR_PORT=50051

# Worker Configuration
WORKER_HEARTBEAT_INTERVAL=10
WORKER_TIMEOUT=30

# Model Configuration
MODEL_PATH=/Users/Shared/models
DEFAULT_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
```

### 1.3 Download Models

```bash
# Create model directory
mkdir -p /Users/Shared/models

# Download a model (example with Llama 3.2 3B)
cd /Users/Shared/models
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir Llama-3.2-3B-Instruct-4bit

# Or use any MLX-compatible model you prefer
```

### 1.4 Deploy Containerized Services

```bash
cd ~/projects/mlx-distributed-inference

# Start all containerized services
docker-compose up -d

# Check services are running
docker-compose ps

# View logs
docker-compose logs -f
```

**Expected output:**
```
NAME                STATUS              PORTS
api                 Up                  0.0.0.0:8000->8000/tcp
coordinator         Up                  0.0.0.0:50051->50051/tcp
redis               Up                  0.0.0.0:6379->6379/tcp
prometheus          Up                  0.0.0.0:9090->9090/tcp
grafana             Up                  0.0.0.0:3000->3000/tcp
nginx               Up                  0.0.0.0:80->80/tcp
```

### 1.5 Start Native Worker on Mac Studio 1

```bash
# Get the IP address of Mac Studio 1
export MAC_STUDIO_1_IP=$(ipconfig getifaddr en0)
echo "Mac Studio 1 IP: $MAC_STUDIO_1_IP"

# Start worker
./scripts/start_worker_macos.sh \
  --worker-id worker-studio1 \
  --coordinator localhost:50051 \
  --model-path /Users/Shared/models \
  --model-name Llama-3.2-3B-Instruct-4bit \
  --port 50052
```

---

## 📦 Step 2: Setup on Mac Studio 2 (Worker Host)

### 2.1 Clone and Setup Project

```bash
# Clone the repository
cd ~/projects
git clone <your-repo-url> mlx-distributed-inference
cd mlx-distributed-inference

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2.2 Download Models

```bash
# Create model directory
mkdir -p /Users/Shared/models

# Download the same model as Studio 1
cd /Users/Shared/models
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit --local-dir Llama-3.2-3B-Instruct-4bit
```

### 2.3 Start Native Worker on Mac Studio 2

```bash
# Get Mac Studio 1's IP address (replace with actual IP)
export MAC_STUDIO_1_IP="192.168.1.100"  # Replace with actual IP

# Start worker pointing to Studio 1's coordinator
./scripts/start_worker_macos.sh \
  --worker-id worker-studio2 \
  --coordinator ${MAC_STUDIO_1_IP}:50051 \
  --model-path /Users/Shared/models \
  --model-name Llama-3.2-3B-Instruct-4bit \
  --port 50052
```

---

## ✅ Step 3: Verify Deployment

### 3.1 Check System Health (from Mac Studio 1)

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {
#   "status": "healthy",
#   "workers": 2,
#   "active_requests": 0,
#   "timestamp": "2025-10-06T15:00:00Z"
# }

# Check worker health
./scripts/health_check.sh --all

# Check coordinator logs
docker-compose logs coordinator | tail -20
```

### 3.2 Test Inference

```bash
# Test API endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -d '{
    "model": "Llama-3.2-3B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello! Tell me a short joke."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### 3.3 Test Streaming

```bash
# Test streaming response
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -d '{
    "model": "Llama-3.2-3B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Count from 1 to 10"}
    ],
    "max_tokens": 100,
    "stream": true
  }'
```

---

## 📊 Step 4: Access Monitoring Dashboards

### Grafana Dashboard

```bash
# Open in browser
open http://localhost:3000

# Default credentials:
# Username: admin
# Password: admin
```

**Available Dashboards:**
- System Overview
- Worker Performance
- Request Latency
- Throughput Metrics

### Prometheus Metrics

```bash
# Open in browser
open http://localhost:9090

# Query examples:
# - inference_requests_total
# - inference_latency_seconds
# - worker_active_requests
# - worker_tokens_per_second
```

### Jaeger Tracing

```bash
# Open in browser
open http://localhost:16686

# View distributed traces across services
```

---

## 🧪 Step 5: Run Performance Tests

### Basic Performance Test

```bash
cd ~/projects/mlx-distributed-inference

# Run performance benchmarks
python tests/test_performance_benchmarking.py

# Run load tests
python tests/load_testing_scenarios.py --workers 2 --duration 60
```

### Load Testing with Multiple Concurrent Requests

```bash
# Install Apache Bench (if not installed)
brew install httpd

# Run load test
ab -n 100 -c 10 -p request.json -T application/json \
  -H "Authorization: Bearer your-secret-api-key-here" \
  http://localhost:8000/v1/chat/completions
```

**Create `request.json`:**
```json
{
  "model": "Llama-3.2-3B-Instruct-4bit",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 50
}
```

---

## 🔧 Management Commands

### Check Status

```bash
# Check all services
docker-compose ps

# Check worker processes
ps aux | grep worker

# Check system health
./scripts/health_check.sh --all --format json
```

### View Logs

```bash
# API logs
docker-compose logs -f api

# Coordinator logs
docker-compose logs -f coordinator

# Worker logs
tail -f logs/worker/*.log

# All logs
tail -f logs/*/*.log
```

### Stop Services

```bash
# Stop workers
./scripts/stop_worker.sh --all

# Stop containerized services
docker-compose down

# Stop everything
docker-compose down && ./scripts/stop_worker.sh --all
```

### Restart Services

```bash
# Restart containerized services
docker-compose restart

# Restart specific service
docker-compose restart coordinator

# Restart worker (stop and start)
./scripts/stop_worker.sh --worker-id worker-studio1
./scripts/start_worker_macos.sh --worker-id worker-studio1 --coordinator localhost:50051
```

---

## 🐛 Troubleshooting

### Workers Not Registering

```bash
# Check coordinator is accessible
nc -zv <MAC_STUDIO_1_IP> 50051

# Check firewall settings
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# Check coordinator logs
docker-compose logs coordinator | grep -i register
```

### High Latency

```bash
# Check network latency between machines
ping <MAC_STUDIO_1_IP>

# Check worker load
./scripts/health_check.sh --all

# Check system resources
top -l 1 | head -20
```

### Model Loading Issues

```bash
# Verify model files exist
ls -lh /Users/Shared/models/Llama-3.2-3B-Instruct-4bit/

# Check worker logs for errors
tail -f logs/worker/worker-studio*.log

# Test model loading manually
python3 -c "
from mlx_lm import load
model, tokenizer = load('/Users/Shared/models/Llama-3.2-3B-Instruct-4bit')
print('Model loaded successfully')
"
```

### Docker Issues

```bash
# Restart Docker Desktop
killall Docker && open /Applications/Docker.app

# Clean up Docker
docker system prune -a

# Check Docker resources
docker stats
```

---

## 📈 Performance Optimization Tips

### 1. Tune Worker Concurrency

Edit `config/worker.yaml`:
```yaml
worker:
  max_concurrent_requests: 4  # Adjust based on your GPU memory
  batch_size: 1
  cache_enabled: true
```

### 2. Optimize Model Loading

```yaml
model:
  preload: true  # Load model at startup
  quantization: "4bit"  # Use 4-bit quantization
  cache_size: 2048  # KV cache size
```

### 3. Network Optimization

```bash
# Use wired connection between Mac Studios
# Enable jumbo frames if supported
sudo ifconfig en0 mtu 9000
```

### 4. Monitor Resource Usage

```bash
# Watch GPU usage
sudo powermetrics --samplers gpu_power -i 1000

# Watch memory
vm_stat 1

# Watch network
nettop -m tcp
```

---

## 🎯 Quick Start Script

Save this as `quick_deploy.sh`:

```bash
#!/bin/bash

# Quick deployment script for Mac Studios

echo "🚀 MLX Distributed Inference - Quick Deploy"
echo

# Mac Studio 1 (Coordinator + Worker)
if [[ "$(hostname)" == "mac-studio-1" ]]; then
    echo "📍 Deploying on Mac Studio 1 (Coordinator Host)"
    
    # Start containerized services
    docker-compose up -d
    sleep 10
    
    # Start local worker
    ./scripts/start_worker_macos.sh \
        --worker-id worker-studio1 \
        --coordinator localhost:50051 \
        --model-path /Users/Shared/models
    
    echo "✅ Mac Studio 1 deployed"
    echo "🌐 API: http://localhost:8000"
    echo "📊 Grafana: http://localhost:3000"
    
# Mac Studio 2 (Worker only)
elif [[ "$(hostname)" == "mac-studio-2" ]]; then
    echo "📍 Deploying on Mac Studio 2 (Worker Host)"
    
    # Get Studio 1 IP (update this!)
    STUDIO1_IP="192.168.1.100"
    
    # Start worker
    ./scripts/start_worker_macos.sh \
        --worker-id worker-studio2 \
        --coordinator ${STUDIO1_IP}:50051 \
        --model-path /Users/Shared/models
    
    echo "✅ Mac Studio 2 deployed"
    echo "🔗 Connected to: ${STUDIO1_IP}:50051"
fi

echo
echo "✨ Deployment complete!"
echo "Run './scripts/health_check.sh --all' to verify"
```

Make it executable:
```bash
chmod +x quick_deploy.sh
```

---

## 📞 Support

If you encounter issues:

1. Check logs: `docker-compose logs` and `tail -f logs/*/*.log`
2. Verify network connectivity: `ping <other-mac-ip>`
3. Check firewall settings
4. Review worker registration in coordinator logs
5. Test API health: `curl http://localhost:8000/health`

---

## 🎉 Success Checklist

- [ ] Both Mac Studios have MLX installed
- [ ] Models downloaded on both machines
- [ ] Docker running on Mac Studio 1
- [ ] Containerized services started
- [ ] Worker 1 registered with coordinator
- [ ] Worker 2 registered with coordinator
- [ ] API health check returns 2 workers
- [ ] Test inference request succeeds
- [ ] Monitoring dashboards accessible
- [ ] Performance tests pass

---

**Ready to deploy? Start with Mac Studio 1, then Mac Studio 2!** 🚀
