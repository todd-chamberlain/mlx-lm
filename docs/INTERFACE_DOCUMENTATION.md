# Interface Documentation

This document provides comprehensive documentation for the key interfaces in the sharding system.

## Table of Contents

1. [ModelLoader Interface](#modelloader-interface)
2. [InferenceEngine Routing Logic](#inferenceengine-routing-logic)
3. [ShardingCoordinator API](#shardingcoordinator-api)

---

## ModelLoader Interface

**Location:** `worker/model_loader.py`

The `ModelLoader` class handles downloading and loading models for distributed inference, supporting both full model loading and partial layer loading (sharding).

### Class: `ModelLoader`

#### Purpose
Loads models for distributed inference with support for layer-level sharding. When a `LayerAssignment` is provided, only the assigned layers are loaded, reducing memory usage per worker.

#### Constructor

```python
def __init__(self, cache_dir: Optional[str] = None)
```

**Parameters:**
- `cache_dir` (Optional[str]): Directory for caching downloaded files. Defaults to `HF_HOME` environment variable or `~/.cache/huggingface`

**Example:**
```python
loader = ModelLoader(cache_dir="/path/to/cache")
```

---

### Key Methods

#### `get_layer_count(model_name: str) -> int`

Get the total number of layers in a model by reading its config.json.

**Parameters:**
- `model_name` (str): HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")

**Returns:**
- `int`: Number of layers in the model

**Raises:**
- `ValueError`: If config cannot be read or layer count not found

**Example:**
```python
layer_count = loader.get_layer_count("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(f"Model has {layer_count} layers")
```

---

#### `download_shard(model_name: str, layer_assignment: LayerAssignment, progress_callback: Optional[callable] = None) -> bool`

Download the files needed for assigned layers.

**Parameters:**
- `model_name` (str): HuggingFace model name
- `layer_assignment` (LayerAssignment): Layer assignment specifying which layers to download
- `progress_callback` (Optional[callable]): Optional callback function `(filename, progress_pct)` for progress updates

**Returns:**
- `bool`: True if download successful, False otherwise

**Raises:**
- `Exception`: If critical files cannot be downloaded

**Behavior:**
- Downloads model configuration files (config.json, tokenizer files)
- Downloads weight files needed for the specified layer range
- Provides progress tracking via callback
- Logs structured information about download progress

**Example:**
```python
assignment = LayerAssignment(
    worker_id="worker-1",
    start_layer=0,
    end_layer=20,
    next_worker_id="worker-2",
    next_worker_address="localhost:50052"
)

def progress_callback(filename, progress_pct):
    print(f"Downloading {filename}: {progress_pct:.1f}%")

success = loader.download_shard(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    layer_assignment=assignment,
    progress_callback=progress_callback
)
```

---

#### `load_partial_model(model_name: str, layer_assignment: LayerAssignment) -> Tuple[Any, Any, Dict[str, Any]]`

Load a partial model with only assigned layers.

**Parameters:**
- `model_name` (str): HuggingFace model name or local path
- `layer_assignment` (LayerAssignment): Layer assignment specifying which layers this worker handles

**Returns:**
- `Tuple[Any, Any, Dict[str, Any]]`: Tuple of (model, tokenizer, shard_metadata) where:
  - `model`: The loaded MLX model (full model, to be filtered by LayerWorker)
  - `tokenizer`: The model's tokenizer
  - `shard_metadata`: Dictionary containing layer assignment and pipeline info

**Raises:**
- `RuntimeError`: If model loading fails

**Shard Metadata Structure:**
```python
{
    'start_layer': int,           # Starting layer index (inclusive)
    'end_layer': int,             # Ending layer index (exclusive)
    'worker_id': str,             # This worker's ID
    'next_worker_id': str,        # Next worker in pipeline (or None)
    'next_worker_address': str,   # Next worker's gRPC address (or None)
    'is_first_worker': bool,      # True if this is the first worker
    'is_last_worker': bool,       # True if this is the last worker
}
```

**Example:**
```python
model, tokenizer, shard_metadata = loader.load_partial_model(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    layer_assignment=assignment
)

print(f"Loaded layers {shard_metadata['start_layer']}-{shard_metadata['end_layer']}")
print(f"Is first worker: {shard_metadata['is_first_worker']}")
print(f"Next worker: {shard_metadata['next_worker_id']}")
```

---

#### `get_download_progress(model_name: str) -> Dict[str, Any]`

Get download progress for a model.

**Parameters:**
- `model_name` (str): HuggingFace model name

**Returns:**
- `Dict[str, Any]`: Dictionary with progress information:
  ```python
  {
      'status': str,              # 'not_started', 'in_progress', or 'complete'
      'progress_percent': float,  # 0.0 to 100.0
      'downloaded_files': int,    # Number of files downloaded
      'total_files': int          # Total files to download
  }
  ```

**Example:**
```python
progress = loader.get_download_progress("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
print(f"Download status: {progress['status']} ({progress['progress_percent']:.1f}%)")
```

---

### Supporting Classes

#### `LayerAssignment`

**Location:** `worker/model_loader.py`

Layer assignment for a worker in the distributed inference pipeline.

**Attributes:**
- `worker_id` (str): Unique identifier for this worker
- `start_layer` (int): Starting layer index (inclusive)
- `end_layer` (int): Ending layer index (exclusive)
- `next_worker_id` (Optional[str]): ID of the next worker in the pipeline (None if last)
- `next_worker_address` (Optional[str]): gRPC address of the next worker (None if last)

**Example:**
```python
assignment = LayerAssignment(
    worker_id="worker-1",
    start_layer=0,
    end_layer=20,
    next_worker_id="worker-2",
    next_worker_address="localhost:50052"
)
```

---

## InferenceEngine Routing Logic

**Location:** `worker/inference_engine.py`

The `InferenceEngine` class handles inference processing with routing logic that determines whether to use standard inference or distributed sharded inference.

### Routing Decision Point

The routing decision is made in the `process_inference()` and `stream_inference()` methods based on the model's shard metadata.

### Method: `process_inference(request: InferenceRequest) -> InferenceResponse`

Process a single inference request with automatic routing.

**Routing Logic:**

```python
# 1. Get model information
model_info = self.model_manager.get_model_info_dict(request.model)

# 2. Check if model is sharded
is_sharded = model_info and model_info.get('is_sharded', False)

# 3. Route based on sharding status
if is_sharded:
    # Route to distributed inference pipeline
    return await self._process_sharded_inference(request, model_info)
else:
    # Route to standard inference
    return await self._process_standard_inference(request)
```

**Decision Criteria:**
- **Sharded Path**: Model has `is_sharded=True` in its model_info
- **Standard Path**: Model has `is_sharded=False` or no shard metadata

**Logging:**
```python
logger.info(
    f"Routing inference request",
    extra={
        "request_id": request.request_id,
        "model": request.model,
        "worker_id": self.worker_id,
        "is_sharded": is_sharded,
        "routing_decision": "sharded_pipeline" if is_sharded else "standard_inference",
        "operation": "inference_routing"
    }
)
```

---

### Method: `_process_sharded_inference(request: InferenceRequest, model_info: Dict[str, Any]) -> InferenceResponse`

Process inference request for sharded model using distributed pipeline.

**Parameters:**
- `request` (InferenceRequest): Inference request
- `model_info` (Dict[str, Any]): Model information including shard metadata

**Returns:**
- `InferenceResponse`: Results from distributed pipeline

**Raises:**
- `RuntimeError`: If this is not the first worker in the pipeline or if inference fails

**Behavior:**

1. **Validate First Worker:**
   ```python
   if not shard_metadata.get('is_first_worker', False):
       raise RuntimeError("Sharded inference must be initiated on first worker")
   ```

2. **Build Pipeline Configuration:**
   ```python
   node_assignments = {
       self.worker_id: list(range(start_layer, end_layer))
   }
   node_addresses = {
       next_worker_id: next_worker_address
   }
   ```

3. **Create Distributed Forward Pass:**
   ```python
   forward_pass = DistributedForwardPass(
       pipeline_id=request.request_id,
       node_assignments=node_assignments,
       node_addresses=node_addresses,
       compression="zstd",
       timeout=30.0
   )
   ```

4. **Execute Pipeline:**
   ```python
   output_logits = await forward_pass.forward(
       input_ids=input_tensor,
       request_id=request.request_id
   )
   ```

5. **Generate Tokens:**
   - Sample from logits using greedy sampling
   - Continue generation until max_tokens or EOS

**Example Flow:**
```
Request → Check is_first_worker → Build pipeline config → 
Create DistributedForwardPass → Execute forward pass → 
Generate tokens → Return response
```

---

### Method: `stream_inference(request: InferenceRequest) -> Iterator[StreamChunk]`

Process streaming inference request with automatic routing.

**Routing Logic:**

Similar to `process_inference()`, but routes to:
- `_stream_sharded_inference()` for sharded models
- Standard streaming for non-sharded models

**Logging:**
```python
logger.info(
    f"Routing streaming inference request",
    extra={
        "request_id": request.request_id,
        "model": request.model,
        "worker_id": self.worker_id,
        "is_sharded": is_sharded,
        "routing_decision": "sharded_streaming_pipeline" if is_sharded else "standard_streaming",
        "operation": "streaming_inference_routing"
    }
)
```

---

### Routing Summary

| Condition | Route | Method |
|-----------|-------|--------|
| `is_sharded=True` | Distributed Pipeline | `_process_sharded_inference()` |
| `is_sharded=False` | Standard Inference | Standard path in `process_inference()` |
| `is_sharded=True` (streaming) | Distributed Streaming | `_stream_sharded_inference()` |
| `is_sharded=False` (streaming) | Standard Streaming | Standard path in `stream_inference()` |

**Key Metadata Fields:**
- `model_info['is_sharded']`: Boolean indicating if model is sharded
- `model_info['shard_metadata']`: Dictionary with layer assignments and pipeline info
- `shard_metadata['is_first_worker']`: Must be True to initiate sharded inference
- `shard_metadata['next_worker_id']`: Next worker in pipeline
- `shard_metadata['next_worker_address']`: gRPC address of next worker

---

## ShardingCoordinator API

**Location:** `coordinator/sharding_coordinator.py`

The `ShardingCoordinator` manages automatic model sharding decisions, shard plan creation, and shard metadata tracking for distributed model inference.

### Class: `ShardingCoordinator`

#### Purpose
Coordinates automatic model sharding across workers by:
- Determining if a model should be sharded
- Creating shard plans using ModelPartitioner
- Tracking shard metadata and download status
- Providing shard information for inference routing

#### Constructor

```python
def __init__(
    self,
    enable_auto_sharding: bool = False,
    shard_threshold_gb: float = 30.0,
    strategy: str = "balanced",
    min_workers_for_sharding: int = 2
)
```

**Parameters:**
- `enable_auto_sharding` (bool): Whether automatic sharding is enabled (default: False)
- `shard_threshold_gb` (float): Minimum model size (GB) to trigger sharding (default: 30.0)
- `strategy` (str): Partitioning strategy - "balanced" or "memory_aware" (default: "balanced")
- `min_workers_for_sharding` (int): Minimum workers required for sharding (default: 2)

**Example:**
```python
coordinator = ShardingCoordinator(
    enable_auto_sharding=True,
    shard_threshold_gb=20.0,
    strategy="balanced",
    min_workers_for_sharding=2
)
```

---

### Key Methods

#### `should_shard_model(model_name: str, model_size_gb: float, num_workers: int) -> bool`

Determine if a model should be sharded based on configuration and available resources.

**Parameters:**
- `model_name` (str): Name of the model
- `model_size_gb` (float): Estimated model size in GB
- `num_workers` (int): Number of available workers

**Returns:**
- `bool`: True if model should be sharded, False otherwise

**Decision Logic:**
1. Check if auto-sharding is enabled
2. Check if enough workers are available (`num_workers >= min_workers_for_sharding`)
3. Check if model size exceeds threshold (`model_size_gb >= shard_threshold_gb`)

**Example:**
```python
should_shard = coordinator.should_shard_model(
    model_name="meta-llama/Llama-2-70b-hf",
    model_size_gb=140.0,
    num_workers=2
)

if should_shard:
    print("Model will be sharded across workers")
else:
    print("Model will be loaded fully on each worker")
```

---

#### `create_shard_plan(model_name: str, num_layers: int, workers: List[Dict[str, any]]) -> ShardedModelMetadata`

Create a shard plan for a model using ModelPartitioner.

**Parameters:**
- `model_name` (str): Name of the model
- `num_layers` (int): Total number of layers in the model
- `workers` (List[Dict]): List of worker info dicts with:
  - `worker_id` (str): Unique worker identifier
  - `address` (str): gRPC address of worker
  - `available_memory` (float, optional): Available memory in GB (for memory_aware strategy)

**Returns:**
- `ShardedModelMetadata`: Metadata with layer assignments for each worker

**Behavior:**
1. Converts strategy string to PartitionStrategy enum
2. Prepares node info for ModelPartitioner
3. Creates pipeline configuration with layer assignments
4. Stores metadata in internal storage
5. Logs structured information about assignments

**Example:**
```python
workers = [
    {
        'worker_id': 'worker-1',
        'address': 'localhost:50051',
        'available_memory': 32.0
    },
    {
        'worker_id': 'worker-2',
        'address': 'localhost:50052',
        'available_memory': 32.0
    }
]

metadata = coordinator.create_shard_plan(
    model_name="meta-llama/Llama-2-70b-hf",
    num_layers=80,
    workers=workers
)

# Access assignments
for worker_id, assignment in metadata.shard_assignments.items():
    print(f"{worker_id}: layers {assignment.start_layer}-{assignment.end_layer}")
```

---

#### `get_shard_metadata(model_name: str) -> Optional[ShardedModelMetadata]`

Get shard metadata for a model.

**Parameters:**
- `model_name` (str): Name of the model

**Returns:**
- `Optional[ShardedModelMetadata]`: Metadata if model is sharded, None otherwise

**Example:**
```python
metadata = coordinator.get_shard_metadata("meta-llama/Llama-2-70b-hf")

if metadata:
    print(f"Model status: {metadata.status}")
    print(f"Total layers: {metadata.total_layers}")
    print(f"Workers: {len(metadata.shard_assignments)}")
else:
    print("Model is not sharded")
```

---

#### `update_shard_status(model_name: str, worker_id: str, download_status: DownloadStatus) -> None`

Update download status for a worker's shard.

**Parameters:**
- `model_name` (str): Name of the model
- `worker_id` (str): ID of the worker
- `download_status` (DownloadStatus): New download status (PENDING, IN_PROGRESS, COMPLETED, FAILED)

**Behavior:**
1. Updates the worker's download status in the assignment
2. Recalculates overall model status based on all worker statuses
3. Logs structured information about the update

**Status Calculation:**
- `READY`: All workers completed
- `FAILED`: At least one worker failed
- `PARTIAL`: Some workers completed, others haven't
- `DOWNLOADING`: Still in progress

**Example:**
```python
# Worker reports download started
coordinator.update_shard_status(
    model_name="meta-llama/Llama-2-70b-hf",
    worker_id="worker-1",
    download_status=DownloadStatus.IN_PROGRESS
)

# Worker reports download completed
coordinator.update_shard_status(
    model_name="meta-llama/Llama-2-70b-hf",
    worker_id="worker-1",
    download_status=DownloadStatus.COMPLETED
)
```

---

#### `is_model_sharded(model_name: str) -> bool`

Check if a model is sharded.

**Parameters:**
- `model_name` (str): Name of the model

**Returns:**
- `bool`: True if model is sharded, False otherwise

**Example:**
```python
if coordinator.is_model_sharded("meta-llama/Llama-2-70b-hf"):
    print("Model is sharded")
else:
    print("Model is not sharded")
```

---

#### `remove_shard_metadata(model_name: str) -> bool`

Remove shard metadata for a model (e.g., after deletion).

**Parameters:**
- `model_name` (str): Name of the model

**Returns:**
- `bool`: True if metadata was removed, False if not found

**Example:**
```python
success = coordinator.remove_shard_metadata("meta-llama/Llama-2-70b-hf")
if success:
    print("Shard metadata removed")
else:
    print("Model was not sharded")
```

---

#### `list_sharded_models() -> List[str]`

List all sharded models.

**Returns:**
- `List[str]`: List of model names that are sharded

**Example:**
```python
sharded_models = coordinator.list_sharded_models()
print(f"Sharded models: {', '.join(sharded_models)}")
```

---

#### `get_worker_assignment(model_name: str, worker_id: str) -> Optional[LayerAssignment]`

Get layer assignment for a specific worker.

**Parameters:**
- `model_name` (str): Name of the model
- `worker_id` (str): ID of the worker

**Returns:**
- `Optional[LayerAssignment]`: Assignment if found, None otherwise

**Example:**
```python
assignment = coordinator.get_worker_assignment(
    model_name="meta-llama/Llama-2-70b-hf",
    worker_id="worker-1"
)

if assignment:
    print(f"Worker handles layers {assignment.start_layer}-{assignment.end_layer}")
    print(f"Next worker: {assignment.next_node_id}")
else:
    print("Worker not assigned to this model")
```

---

### Supporting Classes

#### `ShardedModelMetadata`

Metadata for a sharded model.

**Attributes:**
- `model_name` (str): Name of the model
- `total_layers` (int): Total number of layers in the model
- `shard_assignments` (Dict[str, LayerAssignment]): Worker ID to assignment mapping
- `created_at` (float): Timestamp when metadata was created
- `status` (ShardStatus): Current status of the sharded model

**Methods:**
- `get_assignment(worker_id: str) -> Optional[LayerAssignment]`: Get assignment for a worker
- `update_download_status(worker_id: str, status: DownloadStatus)`: Update worker's download status

---

#### `ShardStatus` (Enum)

Status of a sharded model.

**Values:**
- `DOWNLOADING`: Model is being downloaded
- `READY`: All shards downloaded and ready
- `FAILED`: At least one shard failed to download
- `PARTIAL`: Some shards downloaded, others haven't

---

#### `DownloadStatus` (Enum)

Download status for individual worker.

**Values:**
- `PENDING`: Download not started
- `IN_PROGRESS`: Download in progress
- `COMPLETED`: Download completed successfully
- `FAILED`: Download failed

---

## Integration Example

Here's a complete example showing how these interfaces work together:

```python
# 1. Coordinator decides if model should be sharded
coordinator = ShardingCoordinator(
    enable_auto_sharding=True,
    shard_threshold_gb=20.0,
    strategy="balanced"
)

should_shard = coordinator.should_shard_model(
    model_name="meta-llama/Llama-2-70b-hf",
    model_size_gb=140.0,
    num_workers=2
)

if should_shard:
    # 2. Create shard plan
    workers = [
        {'worker_id': 'worker-1', 'address': 'localhost:50051'},
        {'worker_id': 'worker-2', 'address': 'localhost:50052'}
    ]
    
    metadata = coordinator.create_shard_plan(
        model_name="meta-llama/Llama-2-70b-hf",
        num_layers=80,
        workers=workers
    )
    
    # 3. Each worker downloads its shard
    for worker_id, assignment in metadata.shard_assignments.items():
        loader = ModelLoader()
        
        # Update status to in progress
        coordinator.update_shard_status(
            model_name="meta-llama/Llama-2-70b-hf",
            worker_id=worker_id,
            download_status=DownloadStatus.IN_PROGRESS
        )
        
        # Download shard
        success = loader.download_shard(
            model_name="meta-llama/Llama-2-70b-hf",
            layer_assignment=assignment
        )
        
        # Update status based on result
        status = DownloadStatus.COMPLETED if success else DownloadStatus.FAILED
        coordinator.update_shard_status(
            model_name="meta-llama/Llama-2-70b-hf",
            worker_id=worker_id,
            download_status=status
        )
        
        # Load partial model
        if success:
            model, tokenizer, shard_metadata = loader.load_partial_model(
                model_name="meta-llama/Llama-2-70b-hf",
                layer_assignment=assignment
            )
    
    # 4. Inference engine routes to distributed pipeline
    engine = InferenceEngine(model_manager, worker_id="worker-1")
    
    request = InferenceRequest(
        request_id="req-123",
        prompt="Hello, world!",
        model="meta-llama/Llama-2-70b-hf"
    )
    
    # Engine automatically detects sharded model and routes to distributed pipeline
    response = await engine.process_inference(request)
```

---

## See Also

- [Sharding Guide](SHARDING_GUIDE.md) - User guide for sharding features
- [Architecture Diagrams](ARCHITECTURE_DIAGRAMS.md) - Visual architecture documentation
- [API Reference](SHARDING_API_REFERENCE.md) - REST API documentation
- [Troubleshooting Guide](SHARDING_TROUBLESHOOTING.md) - Common issues and solutions
