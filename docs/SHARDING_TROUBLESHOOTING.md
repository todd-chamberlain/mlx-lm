# Sharding Troubleshooting Guide

**Diagnose and resolve common sharding issues**

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Common Issues](#common-issues)
3. [Debugging Steps](#debugging-steps)
4. [Verification Commands](#verification-commands)
5. [Log Analysis](#log-analysis)
6. [Performance Issues](#performance-issues)

---

## Quick Diagnostics

### Check System Status

```bash
# Check coordinator logs
tail -100 logs/coordinator/coordinator.log | grep -i "error\|shard"

# Check worker logs
tail -100 logs/worker/*.log | grep -i "error\|shard"

# Check if workers are registered
# (Use gRPC client to check coordinator health)
```

### Verify Configuration

```bash
# Check coordinator configuration
cat config/coordinator.yaml | grep -A 10 "sharding:"

# Check environment variables
env | grep -i shard
```

### Check Network

```bash
# Test latency between workers
ping -c 10 <other-worker-ip>

# Test bandwidth
iperf3 -c <other-worker-ip>
```

---

## Common Issues

### Issue 1: "Sharding requires at least 2 workers"

**Symptom:**
```
gRPC error: UNAVAILABLE
Details: "Sharding requires at least 2 workers, but only 1 available"
```

**Cause:**
- Only one worker is registered with coordinator
- Workers failed to register
- Workers are unhealthy

**Diagnosis:**

```python
# Check worker count via gRPC
import grpc
from proto.inference_pb2 import HealthCheckRequest
from proto.inference_pb2_grpc import CoordinatorServiceStub

channel = grpc.insecure_channel('localhost:50051')
stub = CoordinatorServiceStub(channel)

# Check health (you'd need to implement this RPC)
# Or check coordinator logs for worker registration
```

```bash
# Check coordinator logs for worker registration
grep "worker_registered" logs/coordinator/coordinator.log

# Expected output:
# {"operation": "worker_registered", "worker_id": "worker-1", ...}
# {"operation": "worker_registered", "worker_id": "worker-2", ...}
```

**Solution:**

1. **Start additional workers:**
   ```bash
   # On another machine or terminal
   python -m worker.main --worker-id worker-2 --coordinator localhost:50051
   ```

2. **Check worker connectivity:**
   ```bash
   # Test coordinator port
   nc -zv localhost 50051
   
   # Check firewall
   sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
   ```

3. **Wait for registration:**
   - Workers take 10-30 seconds to register
   - Check coordinator logs for registration messages

4. **Verify worker health:**
   ```bash
   # Check worker logs for errors
   tail -f logs/worker/*.log | grep -i error
   ```

---

### Issue 2: Download Stuck at "DOWNLOADING"

**Symptom:**
- GetShardStatus shows `overall_status: "DOWNLOADING"` for extended period
- No progress after 5+ minutes

**Cause:**
- Network connectivity issues
- HuggingFace Hub access problems
- Insufficient disk space
- Worker crashed during download

**Diagnosis:**

```python
# Check shard status
from proto.inference_pb2 import ShardStatusRequest

request = ShardStatusRequest(model_name="model-name")
response = stub.GetShardStatus(request)

for shard in response.shard_assignments:
    print(f"{shard.worker_id}: {shard.download_status} ({shard.progress_percent}%)")
    if shard.error:
        print(f"  Error: {shard.error}")
```

```bash
# Check worker logs for download errors
grep "download" logs/worker/*.log | grep -i error

# Check network connectivity
ping -c 5 huggingface.co

# Check disk space
df -h /Users/Shared/models
```

**Solution:**

1. **Check worker logs:**
   ```bash
   tail -100 logs/worker/worker-1.log | grep -i "download\|error"
   tail -100 logs/worker/worker-2.log | grep -i "download\|error"
   ```

2. **Verify HuggingFace access:**
   ```bash
   # Test HuggingFace Hub access
   curl -I https://huggingface.co
   
   # Check HuggingFace token (if required)
   echo $HF_TOKEN
   ```

3. **Check disk space:**
   ```bash
   df -h /Users/Shared/models
   
   # Clean up if needed
   rm -rf /Users/Shared/models/old-model
   ```

4. **Restart stuck worker:**
   ```bash
   # Find worker process
   ps aux | grep "worker.main"
   
   # Kill stuck worker
   kill <pid>
   
   # Restart worker
   python -m worker.main --worker-id worker-1 --coordinator localhost:50051
   ```

5. **Force re-download:**
   ```python
   request = ModelDownloadRequest(
       model_name="model-name",
       force_redownload=True  # Force fresh download
   )
   response = stub.DownloadModel(request)
   ```

---

### Issue 3: High Inference Latency

**Symptom:**
- Sharded inference takes > 500ms per token
- Significantly slower than expected

**Cause:**
- High network latency between workers
- Network congestion
- Worker overload
- Inefficient activation transfer

**Diagnosis:**

```bash
# Check network latency
ping -c 100 <other-worker-ip>
# Expected: avg < 10ms

# Check activation transfer time in logs
grep "activation_transfer" logs/worker/*.log | grep "duration"

# Check worker CPU/memory
top -l 1 | grep -E "CPU|PhysMem"
```

```python
# Check shard status for inference metrics
request = ShardStatusRequest(model_name="model-name")
response = stub.GetShardStatus(request)

# Note: Inference metrics not currently exposed in GetShardStatus
# Check coordinator logs instead
```

**Solution:**

1. **Improve network:**
   ```bash
   # Use Thunderbolt or 10GbE connection
   # Avoid WiFi
   
   # Test network performance
   iperf3 -c <other-worker-ip>
   # Expected: > 1 Gbps
   ```

2. **Reduce worker load:**
   ```yaml
   # Edit config/worker.yaml
   worker:
     max_concurrent_requests: 2  # Reduce from default
   ```

3. **Check for bottlenecks:**
   ```bash
   # Monitor worker during inference
   top -l 1
   
   # Check for memory pressure
   vm_stat | head -10
   ```

4. **Verify pipeline configuration:**
   ```bash
   # Check coordinator logs for pipeline creation
   grep "pipeline" logs/coordinator/coordinator.log
   
   # Verify layer assignments are balanced
   grep "shard_plan_created" logs/coordinator/coordinator.log
   ```

---

### Issue 4: Inference Fails with "Model not found"

**Symptom:**
```
gRPC error: NOT_FOUND
Details: "Model not found: model-name"
```

**Cause:**
- Model not downloaded
- Download failed
- Model unloaded from worker
- Wrong model name

**Diagnosis:**

```python
# Check if model is sharded
request = ShardStatusRequest(model_name="model-name")
try:
    response = stub.GetShardStatus(request)
    print(f"Is sharded: {response.is_sharded}")
    print(f"Status: {response.overall_status}")
except grpc.RpcError as e:
    print(f"Error: {e.details()}")
```

```bash
# Check worker logs for model loading
grep "model_loaded\|model_name" logs/worker/*.log

# Check coordinator logs for download
grep "download.*model-name" logs/coordinator/coordinator.log
```

**Solution:**

1. **Verify model name:**
   ```python
   # Use exact HuggingFace identifier
   model_name = "meta-llama/Llama-2-70b-chat-hf"  # Correct
   # Not: "llama-2-70b" or "Llama-2-70b"
   ```

2. **Check download status:**
   ```python
   request = ShardStatusRequest(model_name=model_name)
   response = stub.GetShardStatus(request)
   
   if response.overall_status != "READY":
       print(f"Model not ready: {response.overall_status}")
       # Wait or re-download
   ```

3. **Re-download model:**
   ```python
   request = ModelDownloadRequest(
       model_name=model_name,
       force_redownload=True
   )
   response = stub.DownloadModel(request)
   ```

4. **Check worker model list:**
   ```bash
   # Check what models are loaded on workers
   grep "loaded_models" logs/worker/*.log
   ```

---

### Issue 5: One Worker Offline, Model Unavailable

**Symptom:**
- GetShardStatus shows one shard as "FAILED" or missing
- Inference fails even though some shards are healthy

**Cause:**
- Worker crashed or went offline
- Network connectivity lost
- Worker process killed

**Diagnosis:**

```python
# Check shard status
request = ShardStatusRequest(model_name="model-name")
response = stub.GetShardStatus(request)

for shard in response.shard_assignments:
    print(f"{shard.worker_id}: {shard.download_status}")
    if shard.download_status != "COMPLETED":
        print(f"  Problem with {shard.worker_id}")
```

```bash
# Check if worker process is running
ps aux | grep "worker.main"

# Check worker logs
tail -100 logs/worker/worker-2.log
```

**Solution:**

1. **Identify offline worker:**
   ```bash
   # Check coordinator logs
   grep "worker.*offline\|worker.*unhealthy" logs/coordinator/coordinator.log
   ```

2. **Restart offline worker:**
   ```bash
   # On the worker machine
   python -m worker.main --worker-id worker-2 --coordinator <coordinator-ip>:50051
   ```

3. **Verify worker registration:**
   ```bash
   # Check coordinator logs for registration
   tail -f logs/coordinator/coordinator.log | grep "worker_registered"
   ```

4. **Re-download model if needed:**
   ```python
   # If worker lost model data
   request = ModelDownloadRequest(
       model_name="model-name",
       force_redownload=True
   )
   response = stub.DownloadModel(request)
   ```

---

### Issue 6: Shard Status Shows "FAILED"

**Symptom:**
```python
response.overall_status == "FAILED"
# One or more shards show download_status: "FAILED"
```

**Cause:**
- Download error (network, disk, permissions)
- Model loading error (incompatible format, corruption)
- Worker error during download

**Diagnosis:**

```python
# Check which shard failed
request = ShardStatusRequest(model_name="model-name")
response = stub.GetShardStatus(request)

for shard in response.shard_assignments:
    if shard.download_status == "FAILED":
        print(f"Failed: {shard.worker_id}")
        print(f"Error: {shard.error}")
```

```bash
# Check worker logs for detailed error
grep "error.*download\|failed.*download" logs/worker/*.log
```

**Solution:**

1. **Check error message:**
   ```python
   # Error message should indicate the problem
   if shard.error:
       print(f"Error: {shard.error}")
       # Common errors:
       # - "Connection timeout" → network issue
       # - "Disk full" → insufficient space
       # - "Permission denied" → file permissions
       # - "Model not found" → invalid model name
   ```

2. **Fix underlying issue:**
   ```bash
   # Network issue
   ping -c 10 huggingface.co
   
   # Disk space
   df -h /Users/Shared/models
   
   # Permissions
   ls -la /Users/Shared/models
   chmod 755 /Users/Shared/models
   ```

3. **Retry download:**
   ```python
   request = ModelDownloadRequest(
       model_name="model-name",
       force_redownload=True
   )
   response = stub.DownloadModel(request)
   ```

---

## Debugging Steps

### Step 1: Check Configuration

```bash
# Verify sharding is enabled
cat config/coordinator.yaml | grep -A 5 "sharding:"

# Expected:
# sharding:
#   enabled: true
#   threshold_gb: 30
#   min_workers_for_sharding: 2
```

### Step 2: Check Worker Registration

```bash
# Check coordinator logs
grep "worker_registered" logs/coordinator/coordinator.log | tail -5

# Should see at least 2 workers
```

### Step 3: Check Sharding Decision

```bash
# Check if model should be sharded
grep "sharding_decision" logs/coordinator/coordinator.log | tail -5

# Look for:
# {"operation": "sharding_decision", "should_shard": true, ...}
```

### Step 4: Check Shard Plan Creation

```bash
# Check if shard plan was created
grep "shard_plan_created" logs/coordinator/coordinator.log | tail -5

# Should show layer assignments
```

### Step 5: Check Worker Download

```bash
# Check worker logs for download
grep "download.*started\|loading_layers" logs/worker/*.log | tail -10

# Should see layer assignments
```

### Step 6: Check Inference Routing

```bash
# Check coordinator logs for inference routing
grep "inference_routing" logs/coordinator/coordinator.log | tail -5

# Should show routing to first worker for sharded models
```

---

## Verification Commands

### Verify Sharding is Working

```python
import grpc
from proto.inference_pb2 import (
    ModelDownloadRequest,
    ShardStatusRequest,
    InferenceRequest
)
from proto.inference_pb2_grpc import CoordinatorServiceStub

channel = grpc.insecure_channel('localhost:50051')
stub = CoordinatorServiceStub(channel)

# 1. Download model
print("1. Downloading model...")
download_req = ModelDownloadRequest(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
download_resp = stub.DownloadModel(download_req)
print(f"   Sharded: {download_resp.is_sharded}")
print(f"   Workers: {download_resp.workers_notified}")

# 2. Check shard status
print("\n2. Checking shard status...")
status_req = ShardStatusRequest(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
status_resp = stub.GetShardStatus(status_req)
print(f"   Is sharded: {status_resp.is_sharded}")
print(f"   Status: {status_resp.overall_status}")
print(f"   Shards: {len(status_resp.shard_assignments)}")

# 3. Run inference
print("\n3. Running inference...")
infer_req = InferenceRequest(
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    prompt="Hello",
    max_tokens=10
)
infer_resp = stub.Infer(infer_req)
print(f"   Output: {infer_resp.generated_text[:50]}...")

print("\n✓ Sharding is working!")
```

### Verify Memory Savings

```bash
# Check worker memory usage
ps aux | grep "worker.main" | awk '{print $2, $4, $11}'

# Compare with non-sharded model
# Sharded: Each worker should use ~50% of full model memory
# Non-sharded: Each worker uses 100% of model memory
```

### Verify Network Performance

```bash
# Test latency
ping -c 100 <other-worker-ip> | tail -1

# Expected: avg < 10ms

# Test bandwidth
iperf3 -c <other-worker-ip>

# Expected: > 1 Gbps
```

---

## Log Analysis

### Coordinator Logs

**Key log patterns:**

```bash
# Sharding decision
grep "sharding_decision" logs/coordinator/coordinator.log

# Shard plan creation
grep "shard_plan_created" logs/coordinator/coordinator.log

# Worker registration
grep "worker_registered" logs/coordinator/coordinator.log

# Inference routing
grep "inference_routing" logs/coordinator/coordinator.log

# Shard status updates
grep "shard_status_update" logs/coordinator/coordinator.log
```

**Example log entries:**

```json
{"operation": "sharding_decision", "model": "llama-70b", "should_shard": true, "reason": "model_size_exceeds_threshold"}
{"operation": "shard_plan_created", "model": "llama-70b", "total_layers": 80, "num_workers": 2}
{"operation": "worker_registered", "worker_id": "worker-1", "address": "localhost:50052"}
{"operation": "inference_routing", "model": "llama-70b", "is_sharded": true, "target_worker": "worker-1"}
```

### Worker Logs

**Key log patterns:**

```bash
# Layer loading
grep "loading_layers" logs/worker/*.log

# Shard metadata
grep "shard_metadata" logs/worker/*.log

# Inference routing
grep "inference_routing" logs/worker/*.log

# Distributed inference
grep "sharded_inference" logs/worker/*.log

# Activation transfer
grep "activation_transfer" logs/worker/*.log
```

**Example log entries:**

```json
{"operation": "loading_layers", "worker_id": "worker-1", "start_layer": 0, "end_layer": 40}
{"operation": "model_loaded", "model": "llama-70b", "is_sharded": true, "memory_gb": 30.5}
{"operation": "inference_routing", "model": "llama-70b", "is_sharded": true, "routing": "sharded_pipeline"}
{"operation": "sharded_inference", "pipeline_id": "abc123", "stage": 1, "latency_ms": 45.2}
```

---

## Performance Issues

### High Latency

**Target:** < 200ms per token for sharded inference

**Diagnosis:**
```bash
# Check activation transfer time
grep "activation_transfer.*duration" logs/worker/*.log

# Check network latency
ping -c 100 <other-worker-ip>

# Check worker load
top -l 1
```

**Solutions:**
1. Improve network (use Thunderbolt or 10GbE)
2. Reduce concurrent requests per worker
3. Check for CPU/memory bottlenecks

### High Memory Usage

**Target:** ~50% of full model memory per worker

**Diagnosis:**
```bash
# Check worker memory
ps aux | grep "worker.main" | awk '{print $4, $11}'

# Check if layers are actually sharded
grep "loading_layers" logs/worker/*.log
```

**Solutions:**
1. Verify shard plan is correct
2. Check if workers loaded only assigned layers
3. Restart workers if memory leak suspected

### Low Throughput

**Target:** Similar to non-sharded inference

**Diagnosis:**
```bash
# Check batch processing
grep "batch" logs/coordinator/coordinator.log

# Check worker utilization
top -l 1
```

**Solutions:**
1. Enable batch processing
2. Increase max_concurrent_requests
3. Add more workers

---

## Getting Help

If issues persist:

1. **Collect logs:**
   ```bash
   tar -czf sharding-logs.tar.gz logs/
   ```

2. **Collect configuration:**
   ```bash
   tar -czf sharding-config.tar.gz config/
   ```

3. **Document issue:**
   - What you're trying to do
   - What's happening instead
   - Error messages
   - Steps to reproduce

4. **Check documentation:**
   - Sharding Guide: `docs/SHARDING_GUIDE.md`
   - Architecture: `docs/ARCHITECTURE_DIAGRAMS.md`
   - API Reference: `docs/SHARDING_API_REFERENCE.md`

---

**Remember:** Most sharding issues are related to network connectivity, worker registration, or configuration. Start with the quick diagnostics and work through the common issues systematically.
