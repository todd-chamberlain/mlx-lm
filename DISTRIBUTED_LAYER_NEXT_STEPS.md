# Distributed Layer-Level Inference - Implementation Plan

## 🎯 The Core Innovation

**Split a large model across multiple Mac Studios and pipeline activations through them over Thunderbolt.**

This enables:
- Running models larger than single-machine memory (e.g., 70B models across 2-4 Mac Studios)
- True distributed inference with layer-level partitioning
- High-speed activation transfer over Thunderbolt (10.201.20.10 network)

## ✅ What's Already Working

1. **Network Infrastructure**
   - Thunderbolt networking validated (10.201.20.10)
   - Mac Studio 1 ↔ Mac Studio 2 communication working
   - gRPC over Thunderbolt tested

2. **Base System**
   - Coordinator service (manages workers)
   - Worker service (runs MLX inference)
   - API server (OpenAI-compatible)
   - Model loading (MLX models)

3. **Code Structure**
   - 30,000+ lines of production code
   - Async gRPC architecture
   - Comprehensive error handling
   - Monitoring and metrics

## 🚀 What We Need to Build

### Phase 1: Model Layer Partitioning (2-3 hours)

**Goal:** Split model layers across nodes

```python
# Example: 32-layer model across 2 nodes
partition = {
    "studio-1": {"layers": [0-15], "memory": "32GB"},
    "studio-2": {"layers": [16-31], "memory": "32GB"}
}
```

**Files to create:**
- `worker/layer_partition.py` - Partitioning logic
- `worker/model_shard.py` - Load specific layers only
- `proto/distributed.proto` - Distributed inference messages

### Phase 2: Pipeline Execution (3-4 hours)

**Goal:** Flow activations through nodes

```
Request → Studio-1 (layers 0-15) → Studio-2 (layers 16-31) → Response
```

**Files to create:**
- `coordinator/pipeline_manager.py` - Pipeline orchestration
- `worker/activation_transfer.py` - Send/receive activations
- `worker/pipeline_executor.py` - Execute assigned layers

### Phase 3: Activation Transfer (2-3 hours)

**Goal:** Efficient tensor transfer over Thunderbolt

**Optimizations:**
- Zero-copy serialization (pickle/msgpack)
- Compression (zstd for large tensors)
- Streaming for large activations
- Metrics tracking

**Files to create:**
- `common/tensor_transfer.py` - Serialization/compression
- `common/activation_protocol.py` - Transfer protocol

### Phase 4: Integration & Testing (2-3 hours)

**Goal:** End-to-end distributed inference

**Tests:**
- Single request through pipeline
- Multiple concurrent requests
- Node failure handling
- Performance benchmarks

## 📊 Expected Performance

**Target Metrics:**
- Activation transfer: < 10ms over Thunderbolt
- Distributed overhead: < 20% vs single-node
- Throughput efficiency: > 80%
- Support 2x larger models

**Real-world example:**
- Llama-70B (140GB) split across 2 Mac Studios (64GB each)
- ~50-100ms additional latency for activation transfers
- Still faster than swapping to disk!

## 🎯 Immediate Next Steps

1. **Create proto/distributed.proto**
   - Define ActivationTransfer message
   - Define PipelineRequest/Response
   - Define LayerAssignment

2. **Implement worker/layer_partition.py**
   - Parse model architecture
   - Calculate layer splits
   - Assign layers to nodes

3. **Implement worker/activation_transfer.py**
   - Serialize MLX tensors
   - Send over gRPC
   - Deserialize on receiving node

4. **Test on Mac Studios**
   - Load Llama-3.2-3B split across 2 nodes
   - Measure activation transfer latency
   - Validate correctness

## 💡 Why This Matters

This is the **only reason** to build a distributed inference system:

❌ **NOT for:** Load balancing (boring, already solved)
✅ **YES for:** Running models too large for one machine (innovative!)

**Use cases:**
- Llama-70B on 2x Mac Studio (64GB each)
- Mixtral-8x22B on 4x Mac Studio
- Future 200B+ models

## 🚀 Let's Build It!

Ready to implement the distributed layer inference feature?

Start with: `proto/distributed.proto`
