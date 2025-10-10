# Design Document - Automatic Model Sharding

## Overview

This design implements automatic model sharding across multiple workers, enabling the system to run models larger than a single node's memory by distributing layers across nodes and pipelining inference through them.

## Architecture

### High-Level Flow

```
User Request → Coordinator → Worker 1 (Layers 0-60) → Worker 2 (Layers 61-120) → Response
                                  ↓ activations ↓
```

### Components

1. **ShardingCoordinator** (coordinator/sharding_coordinator.py)
   - Decides when to shard models
   - Assigns layers to workers
   - Tracks sharded model metadata

2. **ShardedModelLoader** (worker/sharded_model_loader.py)
   - Downloads only assigned layers
   - Loads partial model into memory
   - Manages layer-specific weights

3. **PipelineInferenceEngine** (worker/pipeline_inference_engine.py)
   - Executes inference on assigned layers
   - Transfers activations to next worker
   - Receives activations from previous worker

4. **ActivationTransferService** (worker/activation_transfer.py) - EXISTS
   - Handles activation serialization
   - Compresses activations
   - Manages gRPC streaming

## Data Models

### ShardedModelMetadata
```python
@dataclass
class ShardedModelMetadata:
    model_name: str
    total_layers: int
    shard_assignments: Dict[str, LayerAssignment]  # worker_id -> assignment
    created_at: float
    status: ShardStatus  # DOWNLOADING, READY, FAILED
```

### LayerAssignment
```python
@dataclass
class LayerAssignment:
    worker_id: str
    start_layer: int
    end_layer: int
    next_worker_id: Optional[str]
    next_worker_address: Optional[str]
    download_status: DownloadStatus
```

## Component Interfaces

### ShardingCoordinator

```python
class ShardingCoordinator:
    def should_shard_model(self, model_name: str, model_size_gb: float) -> bool:
        """Determine if model should be sharded"""
        
    def create_shard_plan(self, model_name: str, workers: List[WorkerInfo]) -> ShardedModelMetadata:
        """Create sharding plan for model"""
        
    def initiate_sharded_download(self, model_name: str, plan: ShardedModelMetadata) -> str:
        """Initiate download across workers"""
        
    def get_shard_status(self, model_name: str) -> ShardedModelMetadata:
        """Get current status of sharded model"""
```

### ShardedModelLoader

```python
class ShardedModelLoader:
    def download_layers(self, model_name: str, start_layer: int, end_layer: int) -> bool:
        """Download only specified layers"""
        
    def load_partial_model(self, model_name: str, assignment: LayerAssignment) -> Tuple[Model, Tokenizer]:
        """Load only assigned layers into memory"""
        
    def get_layer_count(self, model_name: str) -> int:
        """Determine total layer count for model"""
```

### PipelineInferenceEngine

```python
class PipelineInferenceEngine:
    def forward_partial(self, input_ids: mx.array, assignment: LayerAssignment) -> mx.array:
        """Run forward pass through assigned layers"""
        
    def send_activations(self, activations: mx.array, next_worker: str) -> bool:
        """Send activations to next worker"""
        
    def receive_activations(self) -> mx.array:
        """Receive activations from previous worker"""
        
    def generate_pipeline(self, prompt: str, assignment: LayerAssignment) -> Iterator[str]:
        """Generate tokens through pipeline"""
```

## Inference Flow

### Sharded Inference Sequence

1. **Request Arrives**
   - Coordinator receives inference request
   - Checks if model is sharded
   - Routes to first worker in pipeline

2. **First Worker (Layers 0-60)**
   - Tokenizes input
   - Runs embedding + first 60 layers
   - Compresses activations
   - Sends to Worker 2

3. **Second Worker (Layers 61-120)**
   - Receives compressed activations
   - Decompresses
   - Runs remaining 60 layers
   - Generates output token
   - Returns to coordinator

4. **Token Generation Loop**
   - Repeat for each token
   - Maintain KV cache on each worker
   - Stream tokens back to client

## Configuration

### Environment Variables

```bash
# Enable automatic sharding
ENABLE_AUTO_SHARDING=true

# Minimum model size for sharding (GB)
SHARD_THRESHOLD_GB=30

# Compression algorithm (zstd, lz4, none)
ACTIVATION_COMPRESSION=zstd

# Maximum activation transfer timeout (ms)
ACTIVATION_TIMEOUT_MS=5000
```

### Config File

```yaml
sharding:
  enabled: true
  threshold_gb: 30
  strategy: balanced  # balanced, memory_aware
  compression: zstd
  activation_timeout_ms: 5000
  max_pipeline_depth: 4
```

## Error Handling

### Download Failures
- If any worker fails during download, abort all downloads
- Clean up partial files on all workers
- Return error to user

### Inference Failures
- Timeout if activation transfer takes > 5s
- Retry activation transfer up to 3 times
- Fail request if pipeline breaks
- Log detailed error information

### Worker Failures
- Mark sharded model as unavailable if any worker goes offline
- Reject new requests immediately
- Allow in-flight requests to fail gracefully

## Performance Optimization

### Activation Compression
- Use ZSTD compression (level 3) for activations
- Reduces transfer size by ~60%
- Adds ~5ms compression overhead

### Pipelining
- Allow multiple tokens in flight
- Overlap computation and communication
- Target: 2-3 tokens in pipeline simultaneously

### Memory Management
- Pre-allocate activation buffers
- Reuse buffers across requests
- Monitor memory pressure

## Testing Strategy

### Unit Tests
- Test layer assignment logic
- Test partial model loading
- Test activation transfer
- Test compression/decompression

### Integration Tests
- Test end-to-end sharded download
- Test end-to-end sharded inference
- Test pipeline with 2 workers
- Test error scenarios

### Performance Tests
- Measure latency overhead
- Measure throughput
- Measure activation transfer time
- Compare sharded vs non-sharded

## Deployment

### Prerequisites
- 2+ workers available
- Network latency < 10ms between workers
- Sufficient disk space on each worker

### Rollout Plan
1. Deploy code to all workers
2. Enable sharding via config
3. Test with small model (3B)
4. Test with large model (120B)
5. Monitor metrics
6. Gradually enable for all models

## Monitoring

### Key Metrics
- `sharded_inference_latency_ms` - End-to-end latency
- `activation_transfer_time_ms` - Transfer time per layer
- `activation_transfer_size_bytes` - Transfer size
- `pipeline_stage_latency_ms` - Per-worker latency
- `sharded_model_memory_gb` - Memory per worker

### Alerts
- Alert if activation transfer > 100ms
- Alert if pipeline breaks
- Alert if worker goes offline with sharded model

## Migration Path

### Phase 1: Infrastructure (Week 1)
- Implement ShardingCoordinator
- Implement ShardedModelLoader
- Implement PipelineInferenceEngine
- Add configuration support

### Phase 2: Integration (Week 2)
- Integrate with model download flow
- Integrate with inference engine
- Add API endpoints
- Add monitoring

### Phase 3: Testing (Week 3)
- Unit tests
- Integration tests
- Performance tests
- Bug fixes

### Phase 4: Production (Week 4)
- Deploy to production
- Enable for 120B model
- Monitor and optimize
- Documentation

## Success Criteria

- ✅ Can download and shard 120B model across 2 nodes
- ✅ Each node uses ~30GB memory (vs 60GB for full model)
- ✅ Inference latency overhead < 30%
- ✅ Activation transfer < 20ms
- ✅ No memory leaks
- ✅ Graceful error handling
- ✅ Comprehensive monitoring

## Risks and Mitigations

### Risk: Network Latency
- **Mitigation**: Require < 10ms latency between workers
- **Mitigation**: Use compression to reduce transfer size
- **Mitigation**: Pipeline multiple tokens

### Risk: Worker Failures
- **Mitigation**: Detect failures quickly
- **Mitigation**: Fail requests gracefully
- **Mitigation**: Clear error messages

### Risk: Memory Leaks
- **Mitigation**: Careful buffer management
- **Mitigation**: Monitor memory usage
- **Mitigation**: Stress testing

### Risk: Complexity
- **Mitigation**: Thorough testing
- **Mitigation**: Good documentation
- **Mitigation**: Gradual rollout

## Future Enhancements

- Support for 3+ workers
- Dynamic rebalancing
- Automatic failover
- Model-specific optimization
- Tensor parallelism (in addition to pipeline parallelism)
