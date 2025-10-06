# Distributed Layer-Level Inference Design

**Date:** October 6, 2025  
**Priority:** CRITICAL - Core Feature  
**Status:** Design Phase

## Executive Summary

Enable true distributed inference where a single large model is split across multiple Mac Studios, with each node processing specific layers and passing activations over Thunderbolt networking. This allows running models larger than a single machine's memory and leverages the combined compute power of multiple Apple Silicon GPUs.

## Problem Statement

Current implementation only supports request-level distribution (different requests to different workers). We need **layer-level distribution** where:
- A single model is partitioned across multiple nodes
- Each node loads and processes specific layers
- Activations flow through the pipeline over Thunderbolt
- Both Mac Studios work together on a single inference request

## Architecture Overview

### Pipeline Parallelism Model

```
Request → [Node 1: Layers 0-15] → [Node 2: Layers 16-31] → Response
          ↓ Activations over      ↓
          Thunderbolt (10.201.x.x)
```

### Key Components

1. **Model Partitioner** - Splits model layers across nodes
2. **Pipeline Coordinator** - Orchestrates distributed execution
3. **Activation Transfer** - Efficient tensor passing between nodes
4. **Layer Worker** - Executes assigned layers
5. **Pipeline Scheduler** - Manages micro-batching and pipelining

## Detailed Design

### 1. Model Partitioning Strategy

```python
class ModelPartitioner:
    """Partition model layers across nodes."""
    
    def partition_model(self, model_path: str, num_nodes: int) -> List[LayerAssignment]:
        """
        Partition model layers across nodes.
        
        Strategies:
        - BALANCED: Equal layers per node
        - MEMORY_AWARE: Based on layer memory requirements
        - COMPUTE_AWARE: Based on layer compute requirements
        """
        
    def get_layer_assignment(self, node_id: int) -> LayerRange:
        """Get layer range for specific node."""
```

**Partitioning Strategies:**

1. **Balanced** - Equal number of layers per node
   - Simple, predictable
   - May not account for layer complexity

2. **Memory-Aware** - Based on layer memory footprint
   - Ensures no node runs out of memory
   - Requires profiling layer sizes

3. **Compute-Aware** - Based on layer compute requirements
   - Balances compute time across nodes
   - Optimal for throughput

### 2. Layer Worker Implementation

```python
class LayerWorker:
    """Worker that processes specific model layers."""
    
    def __init__(self, node_id: int, layer_range: LayerRange):
        self.node_id = node_id
        self.layer_range = layer_range
        self.layers = None
        self.next_node = None
        self.prev_node = None
    
    async def load_layers(self, model_path: str):
        """Load only assigned layers into memory."""
        
    async def forward(self, activations: mx.array) -> mx.array:
        """Process activations through assigned layers."""
        
    async def send_to_next(self, activations: mx.array):
        """Send activations to next node in pipeline."""
```

### 3. Activation Transfer Protocol

**Requirements:**
- Low latency (< 5ms over Thunderbolt)
- Efficient serialization (zero-copy when possible)
- Large tensor support (up to 1GB+)
- Compression for bandwidth optimization

**Implementation:**

```python
class ActivationTransfer:
    """Efficient activation transfer between nodes."""
    
    async def send_activation(
        self, 
        activation: mx.array, 
        target_node: str,
        compression: bool = True
    ):
        """Send activation tensor to next node."""
        # 1. Serialize MLX array
        # 2. Optional compression
        # 3. Stream over gRPC
        # 4. Zero-copy when possible
        
    async def receive_activation(self) -> mx.array:
        """Receive activation from previous node."""
```

**Optimization Techniques:**
- Use gRPC streaming for large tensors
- Implement tensor compression (quantization, sparsity)
- Zero-copy serialization where possible
- Thunderbolt-optimized networking

### 4. Pipeline Coordinator

```python
class PipelineCoordinator:
    """Coordinates distributed pipeline execution."""
    
    def __init__(self, nodes: List[NodeInfo]):
        self.nodes = nodes
        self.pipeline_stages = []
        
    async def setup_pipeline(self, model_path: str):
        """
        Setup distributed pipeline:
        1. Partition model across nodes
        2. Load layers on each node
        3. Establish connections
        4. Verify pipeline health
        """
        
    async def execute_inference(
        self, 
        input_tokens: List[int]
    ) -> List[int]:
        """
        Execute inference through pipeline:
        1. Embed tokens on first node
        2. Forward through pipeline
        3. Generate tokens iteratively
        4. Return complete sequence
        """
        
    async def execute_streaming(
        self, 
        input_tokens: List[int]
    ) -> AsyncIterator[int]:
        """Stream tokens as they're generated."""
```

### 5. Micro-Batching for Efficiency

```python
class PipelineBatcher:
    """Micro-batching for pipeline efficiency."""
    
    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self.pending_requests = []
        
    async def add_request(self, request: InferenceRequest):
        """Add request to batch."""
        
    async def execute_batch(self):
        """
        Execute batch through pipeline:
        - Multiple requests in flight
        - Overlapping computation and communication
        - Higher throughput
        """
```

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1)

**Tasks:**
1. ✅ Design architecture and protocols
2. Implement ModelPartitioner
3. Implement LayerWorker base class
4. Implement ActivationTransfer protocol
5. Add distributed inference proto definitions
6. Create basic pipeline coordinator

**Deliverables:**
- `worker/model_partitioner.py`
- `worker/layer_worker.py`
- `worker/activation_transfer.py`
- `coordinator/pipeline_coordinator.py`
- `proto/distributed.proto`

### Phase 2: Pipeline Execution (Week 1-2)

**Tasks:**
1. Implement pipeline setup and initialization
2. Implement forward pass through pipeline
3. Add token generation with pipeline
4. Implement streaming support
5. Add error handling and recovery
6. Optimize Thunderbolt networking

**Deliverables:**
- Working end-to-end pipeline
- Token generation through distributed model
- Streaming inference support

### Phase 3: Optimization (Week 2)

**Tasks:**
1. Implement micro-batching
2. Add activation compression
3. Optimize tensor serialization
4. Add pipeline parallelism (multiple requests in flight)
5. Implement memory-aware partitioning
6. Add compute-aware partitioning

**Deliverables:**
- Optimized throughput (target: 80% of single-node)
- Low latency (target: < 10ms overhead)
- Efficient memory usage

### Phase 4: Testing & Validation (Week 2)

**Tasks:**
1. Unit tests for all components
2. Integration tests for pipeline
3. Performance benchmarks
4. Stress testing
5. Failure recovery testing
6. Documentation

**Deliverables:**
- Comprehensive test suite
- Performance benchmarks
- User documentation
- Deployment guide

## Technical Specifications

### Network Requirements

**Thunderbolt Network:**
- IP Range: 10.201.x.x
- Bandwidth: 40 Gbps (Thunderbolt 3/4)
- Latency: < 1ms (local network)
- MTU: 9000 (jumbo frames for efficiency)

**gRPC Configuration:**
```python
options = [
    ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 100MB
    ('grpc.max_send_message_length', 100 * 1024 * 1024),     # 100MB
    ('grpc.http2.max_frame_size', 16 * 1024 * 1024),         # 16MB
    ('grpc.http2.bdp_probe', 1),  # Enable bandwidth detection
]
```

### Memory Requirements

**Per Node:**
- Model layers: 10-30GB (depending on partition)
- Activation buffers: 1-2GB
- KV cache: 2-4GB per request
- System overhead: 2GB

**Total System:**
- 2 nodes × 64GB = 128GB unified memory
- Can run models up to ~100GB

### Performance Targets

**Latency:**
- Single-node baseline: 50ms per token
- Distributed overhead: < 10ms per token
- Target: < 60ms per token (< 20% overhead)

**Throughput:**
- Single-node baseline: 20 tokens/sec
- Target distributed: 16+ tokens/sec (80%+ efficiency)

**Scalability:**
- Support 2-4 nodes in pipeline
- Linear scaling for larger models
- Graceful degradation on node failure

## API Design

### Configuration

```yaml
# distributed_config.yaml
distributed:
  enabled: true
  mode: pipeline  # pipeline, tensor_parallel (future)
  
  nodes:
    - id: node-1
      address: "10.201.1.100:50052"
      layers: [0, 15]  # Layers 0-15
      
    - id: node-2
      address: "10.201.1.101:50052"
      layers: [16, 31]  # Layers 16-31
  
  pipeline:
    micro_batch_size: 4
    max_in_flight: 8
    activation_compression: true
    
  network:
    use_thunderbolt: true
    max_message_size: 100MB
    timeout: 30s
```

### Python API

```python
# Initialize distributed inference
from worker.distributed import DistributedInferenceEngine

engine = DistributedInferenceEngine(
    config_path="distributed_config.yaml",
    node_id="node-1"
)

await engine.initialize()

# Run inference
response = await engine.generate(
    prompt="Hello, world!",
    max_tokens=100
)

# Streaming
async for token in engine.generate_stream(prompt="Hello"):
    print(token, end="", flush=True)
```

### REST API (No Changes)

The existing OpenAI-compatible API remains unchanged. The coordinator automatically routes to distributed pipeline when configured.

## Monitoring & Observability

### Metrics

**Pipeline Metrics:**
- `pipeline_latency_ms` - End-to-end latency
- `stage_latency_ms{stage}` - Per-stage latency
- `activation_transfer_bytes` - Data transferred
- `activation_transfer_time_ms` - Transfer time
- `pipeline_throughput_tokens_per_sec` - Overall throughput

**Node Metrics:**
- `node_layer_compute_time_ms` - Layer processing time
- `node_memory_usage_gb` - Memory per node
- `node_queue_depth` - Pending activations

### Logging

```python
logger.info(f"Pipeline initialized: {len(nodes)} nodes")
logger.debug(f"Node {node_id} processing layers {layer_range}")
logger.info(f"Activation transfer: {size_mb:.2f}MB in {time_ms:.2f}ms")
logger.warning(f"Pipeline stage {stage} slow: {latency_ms}ms")
```

### Tracing

Distributed tracing spans:
1. `pipeline.inference` - Full pipeline execution
2. `stage.{node_id}.forward` - Per-stage processing
3. `transfer.{from}->{to}` - Activation transfers
4. `generation.token` - Token generation

## Error Handling

### Failure Scenarios

1. **Node Failure**
   - Detect via heartbeat timeout
   - Attempt to restart pipeline
   - Fall back to single-node if available

2. **Network Partition**
   - Detect via transfer timeout
   - Retry with exponential backoff
   - Fail request after max retries

3. **Memory Exhaustion**
   - Monitor memory per node
   - Reject new requests when low
   - Trigger garbage collection

4. **Slow Node**
   - Monitor per-stage latency
   - Log warnings for investigation
   - Consider rebalancing partitions

### Recovery Strategies

```python
class PipelineRecovery:
    async def handle_node_failure(self, failed_node: str):
        """
        1. Mark node as unhealthy
        2. Drain in-flight requests
        3. Attempt node restart
        4. Reinitialize pipeline
        5. Resume processing
        """
        
    async def handle_network_partition(self):
        """
        1. Detect partition via timeout
        2. Retry with backoff
        3. Fail request if unrecoverable
        4. Log for investigation
        """
```

## Security Considerations

1. **Node Authentication**
   - Mutual TLS between nodes
   - Node identity verification
   - Prevent unauthorized nodes

2. **Data Encryption**
   - Encrypt activations in transit
   - Protect model weights
   - Secure configuration

3. **Access Control**
   - Restrict pipeline management
   - Audit pipeline operations
   - Rate limiting per client

## Future Enhancements

### Tensor Parallelism (Phase 2)

Split individual layers across nodes for even larger models:
```
Layer N: [Node 1: Columns 0-2048] + [Node 2: Columns 2048-4096]
```

### Dynamic Partitioning

Automatically adjust partitions based on:
- Real-time performance metrics
- Node availability
- Model characteristics

### Multi-Model Pipelines

Support multiple models in pipeline:
```
[Encoder on Node 1] → [Decoder on Node 2]
```

### Speculative Decoding

Use smaller model on one node to predict, verify on larger model:
```
[Small Model: Node 1] → [Large Model: Node 2 (verify)]
```

## Success Criteria

✅ **Functional:**
- Can load and run 70B+ models across 2 nodes
- Generates coherent text through pipeline
- Supports streaming inference
- Handles node failures gracefully

✅ **Performance:**
- < 20% latency overhead vs single-node
- > 80% throughput efficiency
- < 5ms activation transfer time

✅ **Reliability:**
- 99.9% pipeline success rate
- Automatic recovery from transient failures
- Comprehensive error handling

✅ **Usability:**
- Simple configuration
- Clear documentation
- Easy deployment
- Good observability

## References

- MLX Distributed Documentation
- gRPC Streaming Best Practices
- Pipeline Parallelism Papers
- Thunderbolt Networking Specs

---

**Next Steps:**
1. Review and approve design
2. Create implementation tasks
3. Begin Phase 1 development
4. Set up testing infrastructure
