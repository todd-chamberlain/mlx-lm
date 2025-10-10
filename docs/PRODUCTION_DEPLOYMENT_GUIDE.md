# Production Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the MLX Distributed Inference System to production on Mac Studio hardware. It covers deployment validation, testing, monitoring setup, and optimization.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Deployment Architecture](#deployment-architecture)
3. [Phase 1: Mac Studio 1 Deployment](#phase-1-mac-studio-1-deployment)
4. [Phase 2: Mac Studio 2 Deployment](#phase-2-mac-studio-2-deployment)
5. [Phase 3: Distributed Pipeline Configuration](#phase-3-distributed-pipeline-configuration)
6. [Phase 4: API Server and Monitoring](#phase-4-api-server-and-monitoring)
7. [Phase 5: Load Testing](#phase-5-load-testing)
8. [Phase 6: Optimization](#phase-6-optimization)
9. [Phase 7: Documentation](#phase-7-documentation)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

- **Mac Studio 1**: M1/M2 Ultra with 64GB+ RAM
- **Mac Studio 2**: M1/M2 Ultra with 64GB+ RAM
- **Network**: Thunderbolt or 10GbE connection between studios
- **Storage**: 100GB+ available on each machine

### Software Requirements

- **macOS**: 13.0 (Ventura) or later
- **Python**: 3.9 or later
- **MLX**: Latest version (`pip install mlx mlx-lm`)
- **Docker Desktop**: Latest version (Mac Studio 1 only)
- **Git**: For repository management

### Network Configuration

- Both Mac Studios on same network
- Firewall rules allowing:
  - gRPC: Port 50051 (coordinator)
  - Worker: Port 50052 (workers)
  - API: Port 8000 (API server)
  - Monitoring: Ports 3000, 9090, 16686

### Automatic Model Sharding Prerequisites

For deploying large models with automatic sharding:

- **Minimum Workers**: 2+ workers required for sharding
- **Network Latency**: < 10ms between workers (< 5ms recommended)
  - Test with: `ping -c 100 <other-worker-ip>`
  - Thunderbolt 3/4: Typically < 1ms
  - 10 Gigabit Ethernet: Typically < 2ms
- **Network Bandwidth**: 1 Gbps minimum, 10 Gbps recommended
  - Test with: `iperf3 -c <other-worker-ip>`
- **Disk Space**: Sufficient for partial model on each worker
  - Example: 70B model requires ~35GB per worker
- **Memory**: Sufficient for partial model + activations
  - Example: 70B model requires ~30GB per worker

**Network Performance Test:**

```bash
# Install iperf3 if needed
brew install iperf3

# On Mac Studio 1
iperf3 -s

# On Mac Studio 2
iperf3 -c <studio1-ip>

# Expected results for sharding:
# - Bandwidth: > 1 Gbps (10+ Gbps ideal)
# - Latency: < 10ms (< 5ms ideal)
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Mac Studio 1 (Coordinator Host)                             │
├─────────────────────────────────────────────────────────────┤
│ Docker Containers:                                          │
│   • API Server (FastAPI) - Port 8000                        │
│   • Coordinator Service - Port 50051                        │
│   • Redis Cache - Port 6379                                 │
│   • Prometheus - Port 9090                                  │
│   • Grafana - Port 3000                                     │
│   • Jaeger - Port 16686                                     │
│   • Nginx Load Balancer - Port 80                           │
│                                                             │
│ Native Process:                                             │
│   • MLX Worker 1 - Port 50052                               │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Mac Studio 2 (Worker Host)                                  │
├─────────────────────────────────────────────────────────────┤
│ Native Process:                                             │
│   • MLX Worker 2 - Port 50052                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Mac Studio 1 Deployment

### Step 1.1: Prepare Environment

```bash
# On Mac Studio 1
cd ~/projects/mlx-distributed-inference

# Activate virtual environment
source venv/bin/activate

# Verify MLX installation
python3 -c "import mlx.core; print('MLX OK')"

# Check system resources
./scripts/health_check.sh --all
```

### Step 1.2: Download Models

```bash
# Create model directory
sudo mkdir -p /Users/Shared/models
sudo chown $(whoami) /Users/Shared/models

# Download model (example: Llama 3.2 3B)
cd /Users/Shared/models
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit \
    --local-dir Llama-3.2-3B-Instruct-4bit

# Verify model files
ls -lh Llama-3.2-3B-Instruct-4bit/
```

### Step 1.3: Run Deployment Script

```bash
# Run production deployment script for Studio 1
./scripts/deploy_production_studio1.sh

# Expected output:
# ╔════════════════════════════════════════════════════════════════════╗
# ║  MLX Distributed Inference - Mac Studio 1 Production Deployment   ║
# ╚════════════════════════════════════════════════════════════════════╝
#
# [PASS] prerequisites: Apple Silicon detected, Python 3.11, MLX 0.x...
# [PASS] system_resources: Model: Mac14,13, CPUs: 20, Memory: 64GB...
# [PASS] worker_start: Started successfully (PID: 12345)
# [PASS] worker_registration: Registered successfully after 5 attempts
# [PASS] model_loading: Model loaded in 12.34s
# [PASS] local_inference: Inference completed in 2.45s, generated 50 chars
# [PASS] memory_usage: RSS: 4096MB, VSZ: 8192MB, %MEM: 6.4%
# [PASS] performance_metrics: CPU: 15.2%, Uptime: 00:01:23
#
# ✓ Mac Studio 1 deployment completed successfully!
```

### Step 1.4: Verify Deployment

```bash
# Check worker process
ps aux | grep worker-studio1

# Check worker logs
tail -f logs/worker-studio1.log

# Test worker health
curl http://localhost:50052/health

# Check memory usage
./scripts/health_check.sh --worker-id worker-studio1
```

### Step 1.5: Review Deployment Report

```bash
# View deployment report
cat logs/deployment_report_studio1_*.txt

# Check validation log
less logs/deployment_validation_studio1.log
```

### Expected Results

- ✅ Worker process running
- ✅ Model loaded successfully
- ✅ Local inference working
- ✅ Memory usage < 8GB
- ✅ Worker registered with coordinator (if coordinator running)

---

## Phase 2: Mac Studio 2 Deployment

### Step 2.1: Prepare Environment

```bash
# On Mac Studio 2
cd ~/projects/mlx-distributed-inference

# Activate virtual environment
source venv/bin/activate

# Verify MLX installation
python3 -c "import mlx.core; print('MLX OK')"

# Test network connectivity to Studio 1
ping -c 3 10.201.20.10  # Replace with Studio 1 IP
nc -zv 10.201.20.10 50051  # Test coordinator port
```

### Step 2.2: Download Models

```bash
# Create model directory
sudo mkdir -p /Users/Shared/models
sudo chown $(whoami) /Users/Shared/models

# Download same model as Studio 1
cd /Users/Shared/models
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit \
    --local-dir Llama-3.2-3B-Instruct-4bit

# Verify model files
ls -lh Llama-3.2-3B-Instruct-4bit/
```

### Step 2.3: Configure Coordinator Connection

```bash
# Set Studio 1 IP address
export STUDIO1_IP="10.201.20.10"  # Replace with actual IP
export COORDINATOR_HOST="${STUDIO1_IP}:50051"

# Verify connectivity
nc -zv $STUDIO1_IP 50051
```

### Step 2.4: Run Deployment Script

```bash
# Run production deployment script for Studio 2
STUDIO1_IP="10.201.20.10" ./scripts/deploy_production_studio2.sh

# Expected output:
# ╔════════════════════════════════════════════════════════════════════╗
# ║  MLX Distributed Inference - Mac Studio 2 Production Deployment   ║
# ╚════════════════════════════════════════════════════════════════════╝
#
# [PASS] network_connectivity: Ping successful, latency: 0.5ms...
# [PASS] prerequisites: Apple Silicon detected, Python 3.11, MLX 0.x...
# [PASS] system_resources: Model: Mac14,13, CPUs: 20, Memory: 64GB...
# [PASS] worker_start: Started successfully (PID: 23456)
# [PASS] worker_registration: Registered successfully after 8 attempts
# [PASS] model_loading: Model loaded in 11.89s
# [PASS] local_inference: Inference completed in 2.38s, generated 48 chars
# [PASS] memory_usage: RSS: 3984MB, VSZ: 8096MB, %MEM: 6.2%
# [PASS] performance_metrics: CPU: 14.8%, Uptime: 00:01:18
#
# ✓ Mac Studio 2 deployment completed successfully!
```

### Step 2.5: Verify Deployment

```bash
# Check worker process
ps aux | grep worker-studio2

# Check worker logs
tail -f logs/worker-studio2.log

# Test worker health
curl http://localhost:50052/health

# Check memory usage
./scripts/health_check.sh --worker-id worker-studio2
```

### Step 2.6: Verify Cross-Node Communication

```bash
# From Studio 1, check if Studio 2 worker is accessible
nc -zv <STUDIO2_IP> 50052

# From Studio 2, check if coordinator is accessible
nc -zv <STUDIO1_IP> 50051
```

### Expected Results

- ✅ Worker process running
- ✅ Model loaded successfully
- ✅ Local inference working
- ✅ Memory usage < 8GB
- ✅ Worker registered with coordinator
- ✅ Network connectivity to Studio 1 verified

---

## Phase 3: Distributed Pipeline Configuration

### Step 3.1: Verify Both Workers Registered

```bash
# On Mac Studio 1, check coordinator logs
docker-compose logs coordinator | grep -i register

# Or check API health endpoint
curl http://localhost:8000/health | jq

# Expected output:
# {
#   "status": "healthy",
#   "workers": 2,
#   "healthy_workers": 2,
#   "active_requests": 0,
#   "timestamp": "2025-10-07T12:00:00Z"
# }
```

### Step 3.2: Configure Layer Partitioning

```bash
# Edit worker configurations for distributed inference
# On Studio 1:
cat > config/worker_studio1.yaml << EOF
worker_id: "worker-studio1"
distributed:
  enabled: true
  layer_range: [0, 16]  # First 16 layers
  pipeline_stage: 0
EOF

# On Studio 2:
cat > config/worker_studio2.yaml << EOF
worker_id: "worker-studio2"
distributed:
  enabled: true
  layer_range: [16, 32]  # Last 16 layers
  pipeline_stage: 1
EOF
```

### Step 3.3: Test Distributed Inference

```bash
# Run distributed inference test
python3 test_distributed_layers.py

# Expected output:
# Testing distributed layer inference...
# ✓ Layer partitioning configured
# ✓ Activation transfer working
# ✓ Pipeline coordination successful
# ✓ End-to-end inference completed
```

### Step 3.4: Verify Activation Transfer

```bash
# Check activation transfer logs
grep "activation transfer" logs/worker-studio1.log
grep "activation transfer" logs/worker-studio2.log

# Monitor network traffic during inference
# On Studio 1:
nettop -m tcp -t wifi0 -P -L 1

# Look for traffic between studios on port 50052
```

### Expected Results

- ✅ Both workers registered with coordinator
- ✅ Layer partitioning configured correctly
- ✅ Activation transfer working between nodes
- ✅ Distributed inference completing successfully
- ✅ Network latency < 5ms between nodes

---

## Phase 3.5: Automatic Model Sharding Configuration (Optional)

### Overview

Automatic model sharding enables running models larger than single-node memory by distributing layers across workers during download. This is optional but recommended for large models (70B+ parameters).

### Step 3.5.1: Enable Sharding in Coordinator

Edit `config/coordinator.yaml`:

```yaml
sharding:
  # Enable automatic sharding
  enabled: true
  
  # Minimum model size (GB) to trigger auto-sharding
  threshold_gb: 30
  
  # Partitioning strategy
  # - 'balanced': Equal layers per worker (recommended)
  # - 'memory_aware': Distribute based on worker memory
  strategy: balanced
  
  # Activation compression algorithm
  # - 'zstd': Best compression, ~5ms overhead (recommended)
  # - 'lz4': Faster, less compression, ~2ms overhead
  # - 'none': No compression (not recommended)
  compression: zstd
  
  # Maximum time to wait for activation transfer (ms)
  activation_timeout_ms: 5000
  
  # Maximum pipeline depth (number of workers in chain)
  max_pipeline_depth: 4
```

Or use environment variables:

```bash
# Add to .env file
export ENABLE_AUTO_SHARDING=true
export SHARD_THRESHOLD_GB=30
export ACTIVATION_COMPRESSION=zstd
export ACTIVATION_TIMEOUT_MS=5000
```

### Step 3.5.2: Verify Network Performance

Before enabling sharding, verify network meets requirements:

```bash
# Test latency (should be < 10ms, ideally < 5ms)
ping -c 100 <studio2-ip>

# Test bandwidth (should be > 1 Gbps, ideally 10+ Gbps)
# On Studio 1:
iperf3 -s

# On Studio 2:
iperf3 -c <studio1-ip>

# Expected results:
# - Bandwidth: 10+ Gbps (Thunderbolt) or 1+ Gbps (Ethernet)
# - Latency: < 5ms average
```

### Step 3.5.3: Download Model with Sharding

```bash
# Download large model with automatic sharding
curl -X POST http://localhost:8000/admin/download_model \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "meta-llama/Llama-2-70b-chat-hf",
    "enable_sharding": true
  }'

# Response includes shard plan:
# {
#   "success": true,
#   "is_sharded": true,
#   "workers_notified": 2,
#   "shard_plan": {
#     "total_layers": 80,
#     "assignments": [
#       {"worker_id": "worker-studio1", "start_layer": 0, "end_layer": 40},
#       {"worker_id": "worker-studio2", "start_layer": 40, "end_layer": 80}
#     ]
#   }
# }
```

### Step 3.5.4: Monitor Download Progress

```bash
# Check shard status
curl http://localhost:8000/admin/models/meta-llama/Llama-2-70b-chat-hf/shards | jq

# Monitor worker logs
tail -f logs/worker-studio1.log | grep -i "download\|shard"
tail -f logs/worker-studio2.log | grep -i "download\|shard"

# Wait for overall_status: "READY"
```

### Step 3.5.5: Test Sharded Inference

```bash
# Run inference on sharded model
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-2-70b-chat-hf",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# Check inference metrics
curl http://localhost:8000/admin/models/meta-llama/Llama-2-70b-chat-hf/shards | jq '.avg_latency_ms'
```

### Step 3.5.6: Monitor Sharding Metrics

```bash
# View sharding-specific metrics
curl http://localhost:8000/metrics | grep -E "shard|activation_transfer"

# Key metrics to monitor:
# - sharded_inference_latency_seconds
# - activation_transfer_duration_seconds
# - activation_transfer_bytes_total
# - shard_download_status
```

### Expected Results for Sharded Models

- ✅ Model downloaded and split across 2 workers
- ✅ Each worker uses ~50% of full model memory
- ✅ Shard status shows "READY" for all workers
- ✅ Inference completes successfully through pipeline
- ✅ Latency overhead < 30% compared to non-sharded
- ✅ Activation transfer time < 20ms per forward pass

### Monitoring Sharded Models

Add to Grafana dashboard:

```yaml
# Sharding panel configuration
panels:
  - title: "Sharded Model Status"
    targets:
      - expr: shard_download_status
  
  - title: "Activation Transfer Latency"
    targets:
      - expr: histogram_quantile(0.95, activation_transfer_duration_seconds)
  
  - title: "Sharded Inference Latency"
    targets:
      - expr: histogram_quantile(0.95, sharded_inference_latency_seconds)
  
  - title: "Memory per Shard"
    targets:
      - expr: shard_memory_bytes / 1024 / 1024 / 1024
```

### Troubleshooting Sharding

**Issue: High activation transfer latency**

```bash
# Check network latency
ping -c 100 <other-worker-ip>

# Try faster compression
# Edit config/coordinator.yaml:
sharding:
  compression: lz4  # Faster than zstd
```

**Issue: Shard download stuck**

```bash
# Check worker logs
tail -f logs/worker-studio*.log | grep -i error

# Check disk space
df -h /Users/Shared/models

# Retry download
curl -X POST http://localhost:8000/admin/download_model \
  -d '{"model_name": "model-name", "enable_sharding": true, "force_redownload": true}'
```

**Issue: One shard offline**

```bash
# Check shard health
curl http://localhost:8000/health?include_sharded_models=true

# Restart offline worker
./scripts/stop_worker.sh --worker-id worker-studio2
./scripts/start_worker_macos.sh --worker-id worker-studio2
```

For detailed sharding documentation, see [AUTOMATIC_SHARDING_GUIDE.md](AUTOMATIC_SHARDING_GUIDE.md).

---

## Phase 4: API Server and Monitoring

### Step 4.1: Deploy Containerized Services

```bash
# On Mac Studio 1
cd ~/projects/mlx-distributed-inference

# Start all containerized services
docker-compose up -d

# Verify services are running
docker-compose ps

# Expected output:
# NAME                STATUS              PORTS
# api                 Up                  0.0.0.0:8000->8000/tcp
# coordinator         Up                  0.0.0.0:50051->50051/tcp
# redis               Up                  0.0.0.0:6379->6379/tcp
# prometheus          Up                  0.0.0.0:9090->9090/tcp
# grafana             Up                  0.0.0.0:3000->3000/tcp
# jaeger              Up                  0.0.0.0:16686->16686/tcp
# nginx               Up                  0.0.0.0:80->80/tcp
```

### Step 4.2: Test API Endpoints

```bash
# Test health endpoint
curl http://localhost:8000/health | jq

# Test models endpoint
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer your-api-key" | jq

# Test chat completions
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "Llama-3.2-3B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }' | jq

# Test streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "model": "Llama-3.2-3B-Instruct-4bit",
    "messages": [
      {"role": "user", "content": "Count to 5"}
    ],
    "max_tokens": 50,
    "stream": true
  }'
```

### Step 4.3: Configure Prometheus

```bash
# Verify Prometheus is scraping metrics
open http://localhost:9090

# Check targets
# Navigate to Status > Targets
# Verify all services are UP:
#   - api:8000/metrics
#   - coordinator:50051/metrics
#   - worker-studio1:50052/metrics
#   - worker-studio2:50052/metrics

# Test queries:
# - inference_requests_total
# - inference_latency_seconds
# - worker_active_requests
# - worker_tokens_per_second
```

### Step 4.4: Configure Grafana Dashboards

```bash
# Open Grafana
open http://localhost:3000

# Login with default credentials:
# Username: admin
# Password: admin

# Import dashboards:
# 1. System Overview Dashboard
# 2. Worker Performance Dashboard
# 3. Request Latency Dashboard
# 4. Throughput Metrics Dashboard

# Configure alerts:
# - High latency (P95 > 500ms)
# - High error rate (> 5%)
# - Worker unavailable
# - Memory pressure (> 90%)
```

### Step 4.5: Configure Jaeger Tracing

```bash
# Open Jaeger UI
open http://localhost:16686

# Test distributed tracing:
# 1. Send a request through the API
# 2. Search for traces in Jaeger
# 3. Verify trace spans across services:
#    - API Server
#    - Coordinator
#    - Worker 1
#    - Worker 2

# Check trace details:
# - Request flow
# - Latency breakdown
# - Error propagation
```

### Expected Results

- ✅ All containerized services running
- ✅ API endpoints responding correctly
- ✅ Streaming working
- ✅ Prometheus collecting metrics from all services
- ✅ Grafana dashboards displaying data
- ✅ Jaeger showing distributed traces

---

## Phase 5: Load Testing

### Step 5.1: Sustained Load Test

```bash
# Run 1-hour sustained load test
python3 tests/load_testing_scenarios.py \
  --scenario sustained \
  --duration 3600 \
  --concurrent-users 10 \
  --requests-per-second 50

# Monitor during test:
# - Grafana dashboards
# - System resources (top, vm_stat)
# - Worker logs
# - API logs

# Expected metrics:
# - P50 latency: < 100ms
# - P95 latency: < 200ms
# - P99 latency: < 500ms
# - Throughput: 30-50 tokens/sec per worker
# - Error rate: < 1%
# - Memory stable (no leaks)
```

### Step 5.2: Spike Load Test

```bash
# Run 10-minute spike test
python3 tests/load_testing_scenarios.py \
  --scenario spike \
  --duration 600 \
  --concurrent-users 20 \
  --requests-per-second 100

# Monitor system behavior:
# - Request queuing
# - Worker load distribution
# - Error handling
# - Recovery time

# Expected behavior:
# - System handles spike gracefully
# - Requests queued appropriately
# - No crashes or failures
# - Recovery within 30 seconds
```

### Step 5.3: Stress Test

```bash
# Run 30-minute stress test
python3 tests/load_testing_scenarios.py \
  --scenario stress \
  --duration 1800 \
  --concurrent-users 50

# Monitor breaking points:
# - Maximum concurrent requests
# - Memory limits
# - CPU saturation
# - Network bandwidth

# Document limits:
# - Max concurrent users: ?
# - Max requests/sec: ?
# - Memory ceiling: ?
# - CPU ceiling: ?
```

### Step 5.4: Stability Test

```bash
# Run 24-hour stability test
python3 tests/load_testing_scenarios.py \
  --scenario stability \
  --duration 86400 \
  --concurrent-users 10 \
  --requests-per-second 20

# Monitor for:
# - Memory leaks
# - Performance degradation
# - Error accumulation
# - Resource exhaustion

# Expected results:
# - No memory leaks
# - Stable performance over time
# - Error rate remains < 1%
# - No service crashes
```

### Step 5.5: Collect and Analyze Results

```bash
# Generate load test report
python3 scripts/analyze_load_test_results.py \
  --results-dir logs/load_tests \
  --output-format html

# Review metrics:
# - Latency percentiles (P50, P95, P99)
# - Throughput (tokens/sec, requests/sec)
# - Error rates by type
# - Resource utilization trends
# - Cache hit rates
# - Batch efficiency

# Compare with baseline:
python3 scripts/compare_with_baseline.py \
  --baseline docs/PERFORMANCE_BASELINE.md \
  --results logs/load_tests/latest
```

### Expected Results

- ✅ Sustained load: P95 < 200ms, error rate < 1%
- ✅ Spike load: System handles gracefully, recovers quickly
- ✅ Stress test: Limits documented, no crashes
- ✅ Stability test: No memory leaks, stable performance
- ✅ All metrics meet or exceed targets

---

## Phase 6: Optimization

### Step 6.1: Analyze Bottlenecks

```bash
# Review load test results
cat logs/load_tests/analysis_report.html

# Identify bottlenecks:
# - High latency components
# - Resource constraints
# - Network overhead
# - Inefficient operations

# Common bottlenecks:
# - Model loading time
# - Activation transfer latency
# - GC pauses
# - Memory allocation
# - Network bandwidth
```

### Step 6.2: Tune Batch Processing

```bash
# Edit batch configuration
vim config/coordinator.yaml

# Adjust parameters:
batch_manager:
  max_batch_size: 32  # Increase for higher throughput
  max_wait_time_ms: 50  # Decrease for lower latency
  adaptive_sizing: true
  target_gpu_utilization: 0.85

# Restart coordinator
docker-compose restart coordinator

# Re-run load tests
python3 tests/load_testing_scenarios.py --scenario sustained --duration 600

# Compare results
```

### Step 6.3: Optimize Memory Management

```bash
# Edit memory configuration
vim config/worker.yaml

# Adjust parameters:
memory_pool:
  max_pool_size_gb: 8
  max_idle_seconds: 600
  enable_auto_cleanup: true

gc_optimizer:
  manual_gc_enabled: true
  gc_threshold: [10000, 20, 20]

# Restart workers
./scripts/restart_workers.sh

# Monitor memory usage
watch -n 1 './scripts/health_check.sh --all | grep memory'
```

### Step 6.4: Tune Caching

```bash
# Edit cache configuration
vim config/coordinator.yaml

# Adjust parameters:
activation_cache:
  max_size_gb: 4
  ttl_seconds: 3600
  eviction_policy: lru

kv_cache:
  enable_sharing: true
  enable_compression: true
  compression_level: 3

# Restart services
docker-compose restart coordinator

# Monitor cache effectiveness
curl http://localhost:9090/api/v1/query?query=cache_hit_rate
```

### Step 6.5: Optimize Worker Pool

```bash
# Edit coordinator configuration
vim config/coordinator.yaml

# Adjust parameters:
coordinator:
  max_concurrent_per_worker: 4
  health_check_interval_seconds: 10
  unhealthy_threshold_seconds: 30
  removal_threshold_seconds: 120
  
  worker_selection:
    load_weight: 0.5
    latency_weight: 0.3
    reliability_weight: 0.2

# Restart coordinator
docker-compose restart coordinator

# Monitor worker distribution
watch -n 1 'curl -s http://localhost:8000/health | jq'
```

### Step 6.6: Validate Improvements

```bash
# Re-run all load tests
./scripts/run_all_load_tests.sh

# Compare with baseline
python3 scripts/compare_with_baseline.py \
  --baseline logs/load_tests/baseline \
  --optimized logs/load_tests/optimized

# Expected improvements:
# - Latency: 10-20% reduction
# - Throughput: 15-25% increase
# - Memory: 15-25% reduction
# - Cache hit rate: 40-60%
# - Batch efficiency: 95%+
```

### Expected Results

- ✅ Bottlenecks identified and documented
- ✅ Configuration tuned for production workload
- ✅ Performance improvements validated
- ✅ Resource utilization optimized
- ✅ All metrics meet or exceed targets

---

## Phase 7: Documentation

### Step 7.1: Create Deployment Guide

Document the complete deployment process including:

- Prerequisites and requirements
- Step-by-step deployment instructions
- Configuration recommendations
- Validation procedures
- Expected results at each step

### Step 7.2: Create Troubleshooting Guide

Document common issues and solutions:

- Worker registration failures
- Network connectivity issues
- Model loading errors
- Memory pressure
- Performance degradation
- Service crashes

### Step 7.3: Create Monitoring Guide

Document monitoring setup and usage:

- Prometheus queries
- Grafana dashboard usage
- Jaeger trace analysis
- Alert configuration
- Log analysis

### Step 7.4: Create Operations Guide

Document day-to-day operations:

- Starting/stopping services
- Checking system health
- Viewing logs
- Updating configuration
- Scaling workers
- Backup and recovery

### Step 7.5: Create Best Practices Guide

Document production best practices:

- Configuration recommendations
- Performance tuning tips
- Security considerations
- Capacity planning
- Disaster recovery
- Upgrade procedures

---

## Troubleshooting

### Worker Not Starting

**Symptoms**: Worker process fails to start or exits immediately

**Diagnosis**:
```bash
# Check worker logs
tail -100 logs/worker-studio1.log

# Check for errors
grep -i error logs/worker-studio1.log

# Check system resources
./scripts/health_check.sh --all
```

**Common Causes**:
- MLX not installed
- Model files missing
- Insufficient memory
- Port already in use
- Python environment issues

**Solutions**:
```bash
# Reinstall MLX
pip install --upgrade mlx mlx-lm

# Verify model files
ls -lh /Users/Shared/models/

# Check port availability
lsof -i :50052

# Check memory
vm_stat | head -10
```

### Worker Not Registering

**Symptoms**: Worker starts but doesn't register with coordinator

**Diagnosis**:
```bash
# Check coordinator connectivity
nc -zv localhost 50051

# Check coordinator logs
docker-compose logs coordinator | grep -i register

# Check worker logs
grep -i register logs/worker-studio1.log
```

**Common Causes**:
- Coordinator not running
- Network connectivity issues
- Firewall blocking connection
- Incorrect coordinator address

**Solutions**:
```bash
# Start coordinator
docker-compose up -d coordinator

# Check firewall
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# Test connectivity
telnet localhost 50051

# Update coordinator address in config
vim config/worker.yaml
```

### High Latency

**Symptoms**: Inference requests taking longer than expected

**Diagnosis**:
```bash
# Check Grafana latency dashboard
open http://localhost:3000

# Check worker load
./scripts/health_check.sh --all

# Check system resources
top -l 1 | head -20
```

**Common Causes**:
- High worker load
- Memory pressure
- Network latency
- Inefficient batching
- GC pauses

**Solutions**:
```bash
# Reduce batch size
vim config/coordinator.yaml
# Set max_batch_size: 16

# Tune GC
vim config/worker.yaml
# Set gc_threshold: [10000, 20, 20]

# Add more workers
./scripts/start_worker_macos.sh --worker-id worker-studio3

# Check network latency
ping -c 10 <other-studio-ip>
```

### Memory Leaks

**Symptoms**: Memory usage continuously increasing over time

**Diagnosis**:
```bash
# Monitor memory over time
watch -n 5 'ps aux | grep worker | grep -v grep'

# Check for memory leaks
python3 -m memory_profiler worker/main.py

# Review memory metrics
curl http://localhost:9090/api/v1/query?query=process_resident_memory_bytes
```

**Common Causes**:
- Tensor not released
- Cache not evicting
- Reference cycles
- Resource leaks

**Solutions**:
```bash
# Enable tensor lifecycle tracking
vim config/worker.yaml
# Set tensor_lifecycle_tracking: true

# Reduce cache size
vim config/coordinator.yaml
# Set max_size_gb: 2

# Enable auto cleanup
vim config/worker.yaml
# Set enable_auto_cleanup: true

# Restart workers
./scripts/restart_workers.sh
```

### Service Crashes

**Symptoms**: Services unexpectedly stopping or restarting

**Diagnosis**:
```bash
# Check service status
docker-compose ps

# Check logs for crashes
docker-compose logs --tail=100 | grep -i "error\|exception\|crash"

# Check system logs
log show --predicate 'process == "python3"' --last 1h
```

**Common Causes**:
- Out of memory
- Unhandled exceptions
- Resource exhaustion
- Configuration errors

**Solutions**:
```bash
# Increase memory limits
vim docker-compose.yml
# Set mem_limit: 8g

# Add error handling
# Review and update error handling in code

# Check configuration
python3 -m config.validation

# Restart services
docker-compose restart
```

---

## Summary

This production deployment guide provides comprehensive instructions for deploying, validating, testing, and optimizing the MLX Distributed Inference System. Follow each phase sequentially to ensure a successful production deployment.

### Deployment Checklist

- [ ] Phase 1: Mac Studio 1 deployed and validated
- [ ] Phase 2: Mac Studio 2 deployed and validated
- [ ] Phase 3: Distributed pipeline configured and tested
- [ ] Phase 4: API server and monitoring deployed
- [ ] Phase 5: Load tests completed successfully
- [ ] Phase 6: System optimized based on test results
- [ ] Phase 7: Documentation completed

### Success Criteria

- ✅ Both workers operational on physical hardware
- ✅ Distributed inference working across nodes
- ✅ API endpoints responding correctly
- ✅ Monitoring stack operational
- ✅ Load tests passing with target metrics
- ✅ System optimized for production workload
- ✅ Complete documentation available

### Next Steps

After successful deployment:

1. Monitor system performance continuously
2. Review and adjust configuration as needed
3. Scale workers based on demand
4. Implement backup and disaster recovery
5. Plan for capacity expansion
6. Schedule regular maintenance windows

---

**Document Version**: 1.0  
**Last Updated**: October 7, 2025  
**Status**: Production Ready
