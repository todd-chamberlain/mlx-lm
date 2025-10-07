# Distributed Layer Inference Validation

This guide helps you validate that both Mac Studios can see each other and test distributed layer inference where a model is split across multiple nodes.

## Quick Start

### 1. Ensure Both Nodes Are Running

Make sure both Mac Studios have their workers running:

```bash
# On Mac Studio 1 (this machine)
./check_status.sh

# Should show:
# - Coordinator running
# - Worker running on Studio 1
# - Worker running on Studio 2 (via SSH)
```

If services aren't running, start them:

```bash
# Start both studios
./deploy_both_studios.sh

# Or just native workers
./deploy_native_only.sh
```

### 2. Run Validation

```bash
# Quick validation
./validate_nodes.sh

# Or with custom addresses
NODE1=localhost:50052 NODE2=mac-studio-2:50052 ./validate_nodes.sh
```

### 3. Run Detailed Python Validation

```bash
# Activate virtual environment
source venv/bin/activate

# Run validation with default settings
python3 validate_distributed_layers.py

# Or specify custom addresses
python3 validate_distributed_layers.py \
    --node1 localhost:50052 \
    --node2 mac-studio-2:50052 \
    --coordinator localhost:50051
```

## What Gets Tested

### Step 1: Node Connectivity
- ✓ Can connect to Node 1 (Mac Studio 1)
- ✓ Can connect to Node 2 (Mac Studio 2)
- ✓ Can connect to Coordinator
- ✓ Checks how many workers are registered

### Step 2: Node-to-Node Communication
- ✓ Node 1 can reach Node 2 directly
- ✓ Node 2 can reach Node 1 directly
- Tests the distributed inference service endpoints

### Step 3: Worker Registration
- ✓ Both workers are registered with coordinator
- ✓ Both workers are healthy
- Shows worker status and load

### Step 4: Basic Inference
- ✓ Can route inference requests through coordinator
- ✓ Measures latency
- Shows which worker handled the request

### Step 5: Distributed Layer Inference
- ✓ Tests pipeline configuration
- ✓ Tests layer partitioning (layers 0-15 on Node 1, 16-31 on Node 2)
- ✓ Tests activation transfer between nodes
- Measures stage latency, transfer latency, and activation size

### Step 6: Performance Testing
- Runs 10 concurrent requests
- Measures average, min, and max latency
- Calculates success rate

## Expected Output

```
╔════════════════════════════════════════════════════════════════════╗
║     MLX Distributed Layer Inference Validation                    ║
╚════════════════════════════════════════════════════════════════════╝

Configuration:
  Node 1: localhost:50052
  Node 2: mac-studio-2:50052
  Coordinator: localhost:50051

══════════════════════════════════════════════════════════════════════
                        STEP 1: Node Connectivity                        
══════════════════════════════════════════════════════════════════════

  ✓ Node 1 (localhost:50052)............................ PASS
     Status: healthy
  ✓ Node 2 (mac-studio-2:50052)......................... PASS
     Status: healthy
  ✓ Coordinator (localhost:50051)....................... PASS
     Workers: 2/2

══════════════════════════════════════════════════════════════════════
                   STEP 2: Node-to-Node Communication                   
══════════════════════════════════════════════════════════════════════

  ⚠ Node 1 → Node 2..................................... WARN
     Distributed service not yet implemented
  ⚠ Node 2 → Node 1..................................... WARN
     Distributed service not yet implemented

...
```

## Troubleshooting

### No Workers Registered

If you see 0 workers registered:

```bash
# Check if workers are running
./check_status.sh

# Restart workers
./deploy_both_studios.sh
```

### Cannot Connect to Node 2

If Node 2 is unreachable:

```bash
# Test SSH connection
ssh mac-studio-2 "echo 'Connected'"

# Check if worker is running on Node 2
ssh mac-studio-2 "pgrep -f 'python3 -m worker.main'"

# Check Thunderbolt network
ping 10.201.1.101
```

### Distributed Service Not Implemented

This is expected! The validation script will show warnings for distributed layer inference features that haven't been implemented yet. The proto definitions are ready, but the actual implementation is pending.

Current status:
- ✓ Proto definitions ready (`proto/distributed.proto`)
- ✓ gRPC stubs generated
- ⚠ Service implementation pending
- ⚠ Layer partitioning pending
- ⚠ Activation transfer pending

## Next Steps

### To Implement Distributed Layer Inference:

1. **Implement DistributedInferenceService** in workers
   - Add service to `worker/main.py`
   - Implement `ProcessLayers()` method
   - Implement `TransferActivation()` method
   - Implement `CheckPipelineHealth()` method

2. **Create Model Partitioner**
   - `worker/model_partitioner.py`
   - Split model layers across nodes
   - Support BALANCED, MEMORY_AWARE strategies

3. **Create Layer Worker**
   - `worker/layer_worker.py`
   - Load only assigned layers
   - Process activations through layers
   - Send to next node in pipeline

4. **Create Activation Transfer**
   - `worker/activation_transfer.py`
   - Efficient tensor serialization
   - gRPC streaming for large tensors
   - Optional compression

5. **Create Pipeline Coordinator**
   - `coordinator/pipeline_coordinator.py`
   - Setup distributed pipeline
   - Execute inference through pipeline
   - Handle failures and recovery

See `DISTRIBUTED_LAYER_INFERENCE_DESIGN.md` for detailed design.

## Manual Testing

### Test Basic Connectivity

```bash
# Test Node 1
grpcurl -plaintext localhost:50052 inference.WorkerService/HealthCheck

# Test Node 2
grpcurl -plaintext mac-studio-2:50052 inference.WorkerService/HealthCheck

# Test Coordinator
grpcurl -plaintext localhost:50051 inference.CoordinatorService/HealthCheck
```

### Test Inference

```bash
# Via API
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 20
  }'
```

## Configuration

### Environment Variables

```bash
# Node addresses
export NODE1="localhost:50052"
export NODE2="mac-studio-2:50052"
export COORDINATOR="localhost:50051"

# Run validation
./validate_nodes.sh
```

### Distributed Config (Future)

When distributed layer inference is implemented, configure it in `config/distributed.yaml`:

```yaml
distributed:
  enabled: true
  mode: pipeline
  
  nodes:
    - id: node-1
      address: "10.201.1.100:50052"
      layers: [0, 15]
      
    - id: node-2
      address: "10.201.1.101:50052"
      layers: [16, 31]
  
  pipeline:
    micro_batch_size: 4
    activation_compression: true
```

## Performance Expectations

### Current (Request-Level Distribution)
- Latency: ~50-100ms per token
- Throughput: ~20 tokens/sec per worker
- Scalability: Linear with number of workers

### Future (Layer-Level Distribution)
- Latency: ~60-120ms per token (< 20% overhead)
- Throughput: ~16+ tokens/sec (80%+ efficiency)
- Enables models larger than single-node memory

## References

- [Distributed Layer Inference Design](DISTRIBUTED_LAYER_INFERENCE_DESIGN.md)
- [Deployment Guide](DEPLOYMENT_GUIDE_MAC_STUDIOS.md)
- [Quick Start](QUICK_START.md)
- [Proto Definitions](proto/distributed.proto)
