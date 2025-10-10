# MLX Distributed Inference - Complete User Guide

**Everything you need to know to run and operate the system**

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Installation Guide](#installation-guide)
3. [Configuration Guide](#configuration-guide)
4. [Deployment Scenarios](#deployment-scenarios)
5. [Usage Patterns](#usage-patterns)
6. [Monitoring & Operations](#monitoring--operations)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [FAQ](#faq)

---

## Getting Started

### What is MLX Distributed Inference?

A production-ready system that distributes large language model inference across multiple Mac machines with Apple Silicon, enabling:

- **50% memory reduction** per node through layer partitioning
- **2-3x throughput improvement** with intelligent batching
- **30-55% latency reduction** with multi-level caching
- **Linear scalability** by adding more Mac Studios

### Who Should Use This?

- Organizations running LLM inference on Mac hardware
- Developers needing to run models larger than single-machine memory
- Teams requiring high-throughput inference
- Anyone wanting OpenAI-compatible API on local hardware

### Prerequisites

Before starting, ensure you have:
- [ ] 2+ Mac machines with Apple Silicon (M1/M2/M3/M4)
- [ ] 16GB+ RAM per machine (32GB+ recommended)
- [ ] macOS 12.0+ (Monterey or later)
- [ ] Python 3.9+
- [ ] Network connection between machines (Thunderbolt or 10GbE)
- [ ] Basic command-line knowledge

---

## Installation Guide

### Step 1: System Preparation

**On each Mac:**

```bash
# Update macOS
softwareupdate --list
softwareupdate --install --all

# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.9+
brew install python@3.11

# Verify Python version
python3 --version  # Should be 3.9 or higher
```


### Step 2: Clone Repository

```bash
# Clone the repository
git clone <repository-url>
cd mlx-distributed-inference

# Verify structure
ls -la
# Should see: api/, coordinator/, worker/, common/, tests/, docs/, scripts/
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux

# Verify activation (should show venv path)
which python
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Verify MLX installation
python -c "import mlx; print(f'MLX version: {mlx.__version__}')"

# Verify all imports
python -c "import mlx, numpy, grpc, fastapi; print('All dependencies OK')"
```

### Step 5: Download Models

```bash
# Create models directory
mkdir -p models

# Download a model (example: Llama 3.2 1B)
python scripts/download_model.py mlx-community/Llama-3.2-1B-Instruct-4bit

# Verify model files
ls -lh models/mlx-community/Llama-3.2-1B-Instruct-4bit/
# Should see: config.json, tokenizer.json, weights.npz, etc.
```

### Step 6: Verify Installation

```bash
# Run test suite
python -m pytest tests/ -v --tb=short

# Should see: 300+ tests passed

# Run quick smoke test
python scripts/smoke_test.py
```

---

## Configuration Guide

### Basic Configuration

Create `config/distributed_config.yaml`:

```yaml
# Minimal configuration for getting started
coordinator:
  host: "0.0.0.0"
  port: 50051

api:
  host: "0.0.0.0"
  port: 8000
  api_keys:
    - "dev-key-12345"  # Change in production!

worker:
  coordinator_host: "localhost:50051"
  port: 50052
  model_name: "mlx-community/Llama-3.2-1B-Instruct-4bit"
```

### Network Configuration

**For Thunderbolt Network:**

```yaml
# On Mac Studio 1 (Coordinator)
coordinator:
  host: "0.0.0.0"  # Listen on all interfaces
  port: 50051

# On Mac Studio 2 (Worker)
worker:
  coordinator_host: "169.254.x.x:50051"  # Thunderbolt IP
  port: 50052
```

**For Ethernet Network:**

```yaml
# On Mac Studio 1
coordinator:
  host: "192.168.1.10"  # Your LAN IP
  port: 50051

# On Mac Studio 2
worker:
  coordinator_host: "192.168.1.10:50051"
  port: 50052
```

### Security Configuration

```yaml
api:
  # Authentication
  api_keys:
    - "prod-key-abc123"
    - "backup-key-xyz789"
  
  # Rate limiting
  rate_limit:
    requests_per_minute: 60
    burst: 10
  
  # TLS (optional)
  tls:
    enabled: true
    cert_file: "/path/to/cert.pem"
    key_file: "/path/to/key.pem"
```

### Performance Configuration

```yaml
# For maximum throughput
batch:
  enabled: true
  min_size: 4
  max_size: 32
  timeout_ms: 200
  adaptive_sizing: true

cache:
  activation_cache:
    enabled: true
    max_entries: 2000
    ttl_seconds: 600
  kv_cache:
    enabled: true
    max_entries: 1000
    ttl_seconds: 1800

# For minimum latency
batch:
  enabled: true
  min_size: 1
  max_size: 8
  timeout_ms: 50

cache:
  activation_cache:
    ttl_seconds: 60
  kv_cache:
    ttl_seconds: 300
```

---

## Deployment Scenarios

### Scenario 1: Single Machine (Development)

**Use Case:** Testing, development, small models

```bash
# Terminal 1: Start coordinator
python -m coordinator.main

# Terminal 2: Start worker
python -m worker.main

# Terminal 3: Start API
python -m api.main

# Terminal 4: Test
curl http://localhost:8000/health
```

### Scenario 2: Two Mac Studios (Production)

**Use Case:** Production deployment, large models

**Mac Studio 1 (Coordinator + Worker + API):**
```bash
# Deploy everything
./scripts/deploy_production_studio1.sh

# Verify
tail -f logs/worker-studio1.log
curl http://localhost:8000/health
```

**Mac Studio 2 (Worker only):**
```bash
# Set coordinator address
export COORDINATOR_HOST="192.168.1.10:50051"

# Deploy worker
./scripts/deploy_production_studio2.sh

# Verify
tail -f logs/worker-studio2.log
```

### Scenario 3: Docker Deployment

**Use Case:** Containerized deployment, easy scaling

```bash
# Start all services
docker-compose up -d

# View status
docker-compose ps

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale worker=4

# Stop
docker-compose down
```

### Scenario 4: Hybrid Deployment

**Use Case:** Docker for services, native for workers

```bash
# Start coordinator and API in Docker
docker-compose up -d coordinator api

# Start native workers on each Mac
./scripts/start_worker_macos.sh

# Verify
curl http://localhost:8000/health
```

---

## Usage Patterns

### Pattern 1: Simple Request-Response

```python
import requests

def simple_inference(prompt):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": "Bearer your-api-key"},
        json={
            "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100
        }
    )
    return response.json()["choices"][0]["message"]["content"]

# Use it
result = simple_inference("What is machine learning?")
print(result)
```

### Pattern 2: Streaming Responses

```python
import requests
import json

def streaming_inference(prompt):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Authorization": "Bearer your-api-key"},
        json={
            "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        },
        stream=True
    )
    
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                data = line[6:]
                if data != '[DONE]':
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        yield content

# Use it
for token in streaming_inference("Write a story"):
    print(token, end="", flush=True)
```

### Pattern 3: Batch Processing

```python
import asyncio
import aiohttp

async def batch_inference(prompts):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for prompt in prompts:
            task = session.post(
                "http://localhost:8000/v1/chat/completions",
                headers={"Authorization": "Bearer your-api-key"},
                json={
                    "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                    "messages": [{"role": "user", "content": prompt}]
                }
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        results = []
        for r in responses:
            data = await r.json()
            results.append(data["choices"][0]["message"]["content"])
        return results

# Use it
prompts = ["Question 1", "Question 2", "Question 3"]
results = asyncio.run(batch_inference(prompts))
```

### Pattern 4: Multi-Turn Conversation

```python
class Conversation:
    def __init__(self, api_url, api_key):
        self.api_url = api_url
        self.api_key = api_key
        self.messages = []
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def send(self, user_message):
        self.add_message("user", user_message)
        
        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": "mlx-community/Llama-3.2-1B-Instruct-4bit",
                "messages": self.messages
            }
        )
        
        assistant_message = response.json()["choices"][0]["message"]["content"]
        self.add_message("assistant", assistant_message)
        return assistant_message

# Use it
conv = Conversation("http://localhost:8000", "your-api-key")
print(conv.send("Hello!"))
print(conv.send("What's the weather like?"))
print(conv.send("Thanks!"))
```

---

## Monitoring & Operations

### Health Monitoring

```bash
# Check overall health
curl http://localhost:8000/health

# Check specific worker
curl http://localhost:50052/health

# Check coordinator
curl http://localhost:50051/health
```

### Metrics Collection

```bash
# View all metrics
curl http://localhost:8000/metrics

# Filter specific metrics
curl http://localhost:8000/metrics | grep latency
curl http://localhost:8000/metrics | grep cache
curl http://localhost:8000/metrics | grep batch
```

### Log Monitoring

```bash
# View all logs
tail -f logs/*.log

# View specific component
tail -f logs/worker-studio1.log
tail -f logs/coordinator.log
tail -f logs/api.log

# Filter by level
grep "ERROR" logs/*.log
grep "WARNING" logs/*.log

# Follow errors in real-time
tail -f logs/*.log | grep "ERROR"
```

### Grafana Dashboards

1. Open Grafana: `http://localhost:3000`
2. Login: admin/admin
3. Navigate to Dashboards
4. Select "MLX Distributed Inference"

**Key Panels:**
- Request Rate & Latency
- Worker Health & Load
- Cache Hit Rates
- Memory Usage
- GPU Utilization
- Batch Efficiency

### Alerting

Configure alerts in `config/alerts.yaml`:

```yaml
alerts:
  - name: "High Latency"
    condition: "p95_latency > 200ms"
    action: "email"
    recipients: ["ops@example.com"]
  
  - name: "Worker Down"
    condition: "healthy_workers < 2"
    action: "slack"
    channel: "#alerts"
  
  - name: "High Error Rate"
    condition: "error_rate > 5%"
    action: "pagerduty"
```

---

## Performance Optimization

### Optimization Checklist

- [ ] Enable batch processing
- [ ] Enable caching (activation + KV)
- [ ] Enable memory pooling
- [ ] Tune batch sizes for workload
- [ ] Adjust cache TTLs
- [ ] Configure GC thresholds
- [ ] Use memory-aware partitioning
- [ ] Monitor and tune based on metrics

### Tuning for Throughput

```yaml
# Maximize throughput
batch:
  min_size: 8
  max_size: 32
  timeout_ms: 200

worker:
  max_concurrent_requests: 20

cache:
  activation_cache:
    max_entries: 2000
  kv_cache:
    max_entries: 1000
```

### Tuning for Latency

```yaml
# Minimize latency
batch:
  min_size: 1
  max_size: 4
  timeout_ms: 20

worker:
  max_concurrent_requests: 5

cache:
  activation_cache:
    ttl_seconds: 60
  kv_cache:
    ttl_seconds: 300
```

### Tuning for Memory

```yaml
# Optimize memory usage
memory:
  pool_enabled: true
  gc_threshold: 5000
  eager_deallocation: true
  pressure_threshold: 0.7

cache:
  activation_cache:
    max_entries: 500
  kv_cache:
    max_entries: 250
    compression: true
```

---

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for detailed troubleshooting guide.

### Quick Diagnostics

```bash
# Run diagnostic script
./scripts/diagnose.sh

# Check system health
curl http://localhost:8000/health

# View recent errors
grep "ERROR" logs/*.log | tail -20

# Check resource usage
top -l 1 | grep -E "CPU|PhysMem"
```

---

## Best Practices

### Security

1. **Change default API keys** in production
2. **Enable TLS** for production deployments
3. **Use firewall rules** to restrict access
4. **Rotate API keys** regularly
5. **Monitor authentication failures**

### Performance

1. **Enable all optimizations** (batch, cache, memory pool)
2. **Monitor metrics** continuously
3. **Tune based on workload** patterns
4. **Use appropriate batch sizes**
5. **Set cache TTLs** based on query patterns

### Reliability

1. **Deploy multiple workers** for redundancy
2. **Monitor worker health** continuously
3. **Set up alerting** for critical issues
4. **Regular log rotation**
5. **Backup configurations**

### Operations

1. **Document your configuration**
2. **Version control configs**
3. **Test changes in staging**
4. **Monitor after deployments**
5. **Keep logs for debugging**

---

## Automatic Model Sharding

### Overview

Automatic model sharding distributes large models across multiple workers, reducing per-node memory usage by ~50%. This enables running models larger than single-machine capacity.

### When to Use Sharding

- Model size exceeds single-node memory (70B+ parameters)
- You have 2+ workers available
- Network latency between workers is < 10ms

### Download Model with Sharding

**Endpoint:** `POST /admin/download_model`

**Request:**
```bash
curl -X POST http://localhost:8000/admin/download_model \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model_name": "meta-llama/Llama-2-70b-chat-hf",
    "enable_sharding": true,
    "force_redownload": false
  }'
```

**Parameters:**
- `model_name` (required): HuggingFace model identifier
- `enable_sharding` (optional): Enable automatic sharding (default: false)
- `force_redownload` (optional): Force re-download if model exists (default: false)

**Response:**
```json
{
  "success": true,
  "message": "Sharded download initiated across 2 workers",
  "download_id": "dl-abc123",
  "workers_notified": 2,
  "is_sharded": true,
  "shard_plan": {
    "model_name": "meta-llama/Llama-2-70b-chat-hf",
    "total_layers": 80,
    "assignments": [
      {
        "worker_id": "worker-studio1",
        "start_layer": 0,
        "end_layer": 40,
        "next_worker_id": "worker-studio2"
      },
      {
        "worker_id": "worker-studio2",
        "start_layer": 40,
        "end_layer": 80,
        "next_worker_id": null
      }
    ]
  }
}
```

### Check Shard Status

**Endpoint:** `GET /admin/models/{model_name}/shards`

**Request:**
```bash
curl http://localhost:8000/admin/models/meta-llama/Llama-2-70b-chat-hf/shards
```

**Response:**
```json
{
  "model_name": "meta-llama/Llama-2-70b-chat-hf",
  "is_sharded": true,
  "total_layers": 80,
  "shard_assignments": [
    {
      "worker_id": "worker-studio1",
      "start_layer": 0,
      "end_layer": 40,
      "status": "READY",
      "progress_percent": 100.0,
      "error": null
    },
    {
      "worker_id": "worker-studio2",
      "start_layer": 40,
      "end_layer": 80,
      "status": "READY",
      "progress_percent": 100.0,
      "error": null
    }
  ],
  "overall_status": "READY",
  "inference_count": 156,
  "avg_latency_ms": 142.3
}
```

**Status Values:**
- `DOWNLOADING`: Download in progress
- `READY`: Shard loaded and ready for inference
- `FAILED`: Download or loading failed
- `UNLOADING`: Shard being unloaded

### Health Check with Sharded Models

**Endpoint:** `GET /health?include_sharded_models=true`

**Request:**
```bash
curl "http://localhost:8000/health?include_sharded_models=true"
```

**Response:**
```json
{
  "status": "healthy",
  "total_workers": 2,
  "healthy_workers": 2,
  "active_requests": 3,
  "timestamp": 1704067200,
  "sharded_model_count": 1,
  "sharded_models": [
    {
      "model_name": "meta-llama/Llama-2-70b-chat-hf",
      "total_shards": 2,
      "healthy_shards": 2,
      "status": "ready"
    }
  ]
}
```

### Inference with Sharded Models

Inference works identically for sharded and non-sharded models:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "meta-llama/Llama-2-70b-chat-hf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

The system automatically routes sharded models through the pipeline inference path.

### Delete Sharded Model

**Endpoint:** `DELETE /admin/models/{model_name}`

Deleting a sharded model removes partial weights from all workers:

```bash
curl -X DELETE http://localhost:8000/admin/models/meta-llama/Llama-2-70b-chat-hf
```

### Python Example

```python
import requests
import time

# Download with sharding
response = requests.post(
    "http://localhost:8000/admin/download_model",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "model_name": "meta-llama/Llama-2-70b-chat-hf",
        "enable_sharding": True
    }
)

print(f"Download initiated: {response.json()['download_id']}")

# Monitor progress
model_name = "meta-llama/Llama-2-70b-chat-hf"
while True:
    status = requests.get(
        f"http://localhost:8000/admin/models/{model_name}/shards"
    ).json()
    
    if status["overall_status"] == "READY":
        print("Download complete!")
        break
    
    print(f"Status: {status['overall_status']}")
    time.sleep(5)

# Run inference
result = requests.post(
    "http://localhost:8000/v1/chat/completions",
    headers={"Authorization": "Bearer your-api-key"},
    json={
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello!"}]
    }
).json()

print(result["choices"][0]["message"]["content"])
```

### Performance Characteristics

**Memory Savings:**
- Each worker uses ~50% of full model memory
- Example: 70B model requires 30GB per worker instead of 60GB

**Latency Overhead:**
- Typical overhead: 15-30% compared to non-sharded
- Activation transfer: ~15ms per forward pass
- Compression overhead: ~5ms (ZSTD)

**Network Requirements:**
- Minimum: 1 Gbps
- Recommended: 10 Gbps or Thunderbolt
- Latency: < 10ms between workers

For detailed information, see [AUTOMATIC_SHARDING_GUIDE.md](AUTOMATIC_SHARDING_GUIDE.md).

---

## FAQ

**Q: How many Mac Studios do I need?**
A: Minimum 2 for distributed inference. More nodes = better performance and larger model support.

**Q: What network speed do I need?**
A: Thunderbolt 3/4 (40 Gbps) or 10GbE minimum. Lower latency is more important than bandwidth.

**Q: Can I mix different Mac models?**
A: Yes! Use memory-aware partitioning to handle heterogeneous clusters.

**Q: How do I update models?**
A: Download new model, update config, restart workers. No downtime needed with multiple workers.

**Q: What's the maximum model size?**
A: Limited by total cluster memory. 70B models work well on 2x Mac Studio with 64GB each.

**Q: How do I scale beyond 2 nodes?**
A: Add more workers, they auto-register. System supports 8+ nodes.

**Q: Can I run multiple models?**
A: Currently one model per cluster. Multi-model support is on the roadmap.

**Q: How do I backup/restore?**
A: Backup `config/` and `models/` directories. Restore and restart services.

**Q: What is automatic model sharding?**
A: Sharding splits large models across workers, reducing per-node memory by ~50%. Enable with `enable_sharding: true` when downloading models. Requires 2+ workers.

**Q: When should I use sharding?**
A: Use sharding for models > 30GB that exceed single-node memory. Don't use for small models or when you need absolute minimum latency.

**Q: Does sharding affect inference quality?**
A: No! Sharding is a memory optimization. Models produce identical outputs whether sharded or not.

---

**For more help, see:**
- [README.md](../README.md) - Quick start and overview
- [PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md) - Detailed deployment
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Problem solving
- [API_REFERENCE.md](API_REFERENCE.md) - Complete API docs
