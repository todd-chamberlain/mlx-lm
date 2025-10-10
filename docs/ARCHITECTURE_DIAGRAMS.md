# Architecture Diagrams

Visual representations of the MLX Distributed Inference System architecture, showing actual data flow and component interactions.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│                          CLIENT APPLICATIONS                            │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│  │   Web App    │  │  Mobile App  │  │  CLI Tool    │                │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                │
│         │                  │                  │                         │
│         └──────────────────┴──────────────────┘                         │
│                            │                                            │
│                            │ HTTP/REST (OpenAI Compatible)              │
│                            ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                    FastAPI Server (Port 8000)                    │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │  │     Auth     │  │   Streaming  │  │ Rate Limiting│          │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  └─────────────────────────────┬───────────────────────────────────┘  │
│                                │ gRPC                                  │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │              Coordinator Service (Port 50051)                    │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │  │Worker Manager│  │Request Router│  │Shard Coord   │          │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │  │
│  │  │Batch Scheduler│ │Health Monitor│  │Load Balancer │          │  │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │  │
│  └─────────┬──────────────┬──────────────┬────────────────────────┘  │
│            │ gRPC         │ gRPC         │ gRPC                       │
│            ▼              ▼              ▼                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Worker 1   │  │  Worker 2   │  │  Worker N   │                  │
│  │             │  │             │  │             │                  │
│  │ Full Model  │  │ Full Model  │  │ Full Model  │                  │
│  │ OR          │  │ OR          │  │ OR          │                  │
│  │ Layers 0-N  │  │ Layers N-M  │  │ Layers M-L  │                  │
│  │             │  │             │  │             │                  │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │                  │
│  │ │MLX Engine│ │  │ │MLX Engine│ │  │ │MLX Engine│ │                  │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │                  │
│  │ ┌─────────┐ │  │ ┌─────────┐ │  │ ┌─────────┐ │                  │
│  │ │Metal GPU│ │  │ │Metal GPU│ │  │ │Metal GPU│ │                  │
│  │ └─────────┘ │  │ └─────────┘ │  │ └─────────┘ │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
│    Mac Studio 1     Mac Studio 2     Mac Studio N                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Note:** Workers can run either full models (non-sharded) or partial models (sharded). Both modes are supported simultaneously.

---

## Non-Sharded Request Flow

Standard inference path for models that fit on a single worker:

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. POST /v1/chat/completions
     ▼
┌─────────────────┐
│   API Server    │
│  (Port 8000)    │
└────┬────────────┘
     │ 2. Authenticate & Validate
     │ 3. Forward to Coordinator (gRPC)
     ▼
┌─────────────────┐
│  Coordinator    │
│  (Port 50051)   │
└────┬────────────┘
     │ 4. Check: is_model_sharded(model) → FALSE
     │ 5. Select Worker (load balancing)
     │ 6. Route Request
     ▼
┌─────────────────┐
│   Worker 1      │  7. InferenceEngine.process_inference()
│  (Full Model)   │  8. Check: model_info.is_sharded → FALSE
└────┬────────────┘  9. Use standard inference path
     │ 10. Generate tokens with MLX
     │ 11. Return InferenceResponse
     ▼
┌─────────────────┐
│  Coordinator    │  12. Return Response
└────┬────────────┘
     │ 13. Forward to API
     ▼
┌─────────────────┐
│   API Server    │  14. Format Response (OpenAI compatible)
└────┬────────────┘  15. Stream Tokens (SSE) or return JSON
     │
     ▼
┌──────────┐
│  Client  │  16. Receive Response
└──────────┘
```

---

## Sharded Request Flow

Distributed inference path for large models split across workers:

```
┌──────────┐
│  Client  │
└────┬─────┘
     │ 1. POST /v1/chat/completions
     ▼
┌─────────────────┐
│   API Server    │
│  (Port 8000)    │
└────┬────────────┘
     │ 2. Authenticate & Validate
     │ 3. Forward to Coordinator (gRPC)
     ▼
┌─────────────────┐
│  Coordinator    │
│  (Port 50051)   │
└────┬────────────┘
     │ 4. Check: is_model_sharded(model) → TRUE
     │ 5. Get shard metadata
     │ 6. Find first worker (layer 0)
     │ 7. Route to first worker
     ▼
┌─────────────────┐
│   Worker 1      │  8. InferenceEngine.process_inference()
│  (Layers 0-60)  │  9. Check: model_info.is_sharded → TRUE
└────┬────────────┘  10. Route to _process_sharded_inference()
     │ 11. Build pipeline config from shard_metadata
     │ 12. Create DistributedForwardPass instance
     │ 13. Execute pipeline.forward()
     │     ├─ Process layers 0-60
     │     ├─ Generate activations
     │     └─ Serialize activations
     │ 14. gRPC: ForwardPass() → Worker 2
     ▼
┌─────────────────┐
│   Worker 2      │  15. DistributedInferenceService.ForwardPass()
│  (Layers 61-120)│  16. Deserialize activations
└────┬────────────┘  17. LayerWorker.forward(activations)
     │ 18. Process layers 61-120
     │ 19. Generate final output
     │ 20. Return activations to Worker 1
     ▼
┌─────────────────┐
│   Worker 1      │  21. Receive final activations
│  (Layers 0-60)  │  22. Generate tokens (greedy sampling)
└────┬────────────┘  23. Return InferenceResponse
     │
     ▼
┌─────────────────┐
│  Coordinator    │  24. Return Response
└────┬────────────┘
     │ 25. Forward to API
     ▼
┌─────────────────┐
│   API Server    │  26. Format Response
└────┬────────────┘  27. Stream Tokens (SSE) or return JSON
     │
     ▼
┌──────────┐
│  Client  │  28. Receive Response
└──────────┘
```



---

## Model Download Flow

### Non-Sharded Download

```
┌──────────┐
│  Client  │  1. DownloadModel(model_name)
└────┬─────┘
     ▼
┌─────────────────┐
│  Coordinator    │  2. Check: should_shard_model()
│                 │     ├─ Model size < threshold → FALSE
│                 │     ├─ Workers < min_workers → FALSE
│                 │     └─ Sharding disabled → FALSE
└────┬────────────┘
     │ 3. Broadcast download to all workers
     │
     ├──────────────┬──────────────┐
     ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│Worker 1 │   │Worker 2 │   │Worker N │
└────┬────┘   └────┬────┘   └────┬────┘
     │             │             │
     │ 4. Download full model from HuggingFace
     │ 5. Load all layers into memory
     │ 6. Report success to coordinator
     │             │             │
     └─────────────┴─────────────┘
                   │
                   ▼
┌─────────────────┐
│  Coordinator    │  7. Return success
└─────────────────┘
```

### Sharded Download

```
┌──────────┐
│  Client  │  1. DownloadModel(model_name)
└────┬─────┘
     ▼
┌─────────────────┐
│  Coordinator    │  2. Check: should_shard_model()
│                 │     ├─ Model size > threshold → TRUE
│                 │     ├─ Workers >= min_workers → TRUE
│                 │     └─ Sharding enabled → TRUE
└────┬────────────┘
     │ 3. Get layer count from model config
     │ 4. Create shard plan:
     │    ├─ Worker 1: layers 0-60
     │    ├─ Worker 2: layers 61-120
     │    └─ Store shard metadata
     │
     ├──────────────┬──────────────┐
     ▼              ▼              
┌─────────┐   ┌─────────┐   
│Worker 1 │   │Worker 2 │   
└────┬────┘   └────┬────┘   
     │             │             
     │ 5. Receive layer assignment (0-60)
     │ 6. Download model from HuggingFace
     │ 7. Load ONLY assigned layers (0-60)
     │ 8. Store shard metadata
     │ 9. Report success to coordinator
     │             │             
     │             │ 5. Receive layer assignment (61-120)
     │             │ 6. Download model from HuggingFace
     │             │ 7. Load ONLY assigned layers (61-120)
     │             │ 8. Store shard metadata
     │             │ 9. Report success to coordinator
     │             │
     └─────────────┴─────────────┘
                   │
                   ▼
┌─────────────────┐
│  Coordinator    │  10. Update shard status
│                 │  11. Check all workers complete
└─────────────────┘  12. Return success with shard plan
```

---

## Component Interactions

### Coordinator Components

```
┌─────────────────────────────────────────────────────────────┐
│                    CoordinatorService                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │WorkerPool    │  │RequestRouter │  │ShardingCoord │    │
│  │              │  │              │  │              │    │
│  │• register    │  │• route_infer │  │• should_shard│    │
│  │• get_healthy │  │• _route_shard│  │• create_plan │    │
│  │• heartbeat   │  │• get_stats   │  │• get_metadata│    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │BatchManager  │  │HealthMonitor │  │CacheCoord    │    │
│  │              │  │              │  │              │    │
│  │• add_request │  │• check_health│  │• coordinate  │    │
│  │• process_batch│ │• mark_unhealthy│ │• invalidate │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Worker Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Worker Service                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ModelManager  │  │InferenceEngine│ │ModelLoader   │    │
│  │              │  │              │  │              │    │
│  │• load_model  │  │• process_infer│ │• load_partial│    │
│  │• load_sharded│  │• _process_shard│ │• load_full  │    │
│  │• unload_model│  │• stream_infer │  │              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │LayerWorker   │  │DistributedFwd│  │GrpcService   │    │
│  │              │  │              │  │              │    │
│  │• load_layers │  │• forward     │  │• Infer       │    │
│  │• forward     │  │• _execute_stage│ │• ForwardPass │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Sharding Decision Flow

```
┌─────────────────┐
│DownloadModel RPC│
└────┬────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ ShardingCoordinator.should_shard_model()│
├─────────────────────────────────────────┤
│                                         │
│  1. Check: enable_auto_sharding?       │
│     └─ NO → return False               │
│                                         │
│  2. Check: num_workers >= min_workers? │
│     └─ NO → return False               │
│                                         │
│  3. Check: model_size_gb > threshold?  │
│     └─ NO → return False               │
│                                         │
│  4. All checks passed → return True    │
│                                         │
└────┬────────────────────────────────────┘
     │
     ├─ TRUE ──────────────────────────────┐
     │                                     │
     ▼                                     ▼
┌─────────────────┐              ┌─────────────────┐
│ create_shard_   │              │ Broadcast to    │
│ plan()          │              │ all workers     │
│                 │              │ (non-sharded)   │
│ • Get layer cnt │              └─────────────────┘
│ • Divide layers │
│ • Assign workers│
│ • Store metadata│
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ Send assignments│
│ to workers      │
└─────────────────┘
```

### Inference Routing Decision

```
┌─────────────────┐
│ InferenceRequest│
└────┬────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ RequestRouter.route_inference()         │
├─────────────────────────────────────────┤
│                                         │
│  1. Check: is_model_sharded(model)?    │
│     └─ Query ShardingCoordinator       │
│                                         │
└────┬────────────────────────────────────┘
     │
     ├─ TRUE ──────────────────────────────┐
     │                                     │
     ▼                                     ▼
┌─────────────────┐              ┌─────────────────┐
│ _route_sharded_ │              │ route_to_worker │
│ inference()     │              │ (load balance)  │
│                 │              └─────────────────┘
│ • Get metadata  │
│ • Find first    │
│   worker (L0)   │
│ • Route request │
└────┬────────────┘
     │
     ▼
┌─────────────────┐
│ Worker 1        │
│ (First in chain)│
└────┬────────────┘
     │
     ▼
┌─────────────────────────────────────────┐
│ InferenceEngine.process_inference()     │
├─────────────────────────────────────────┤
│                                         │
│  1. Get model_info                     │
│  2. Check: model_info.is_sharded?      │
│                                         │
└────┬────────────────────────────────────┘
     │
     ├─ TRUE ──────────────────────────────┐
     │                                     │
     ▼                                     ▼
┌─────────────────┐              ┌─────────────────┐
│ _process_       │              │ Standard        │
│ sharded_        │              │ inference path  │
│ inference()     │              └─────────────────┘
│                 │
│ • Build pipeline│
│ • Create        │
│   DistributedFwd│
│ • Execute stages│
│ • Generate      │
│   tokens        │
└─────────────────┘
```

---

## Data Structures

### Shard Metadata

```python
ShardedModelMetadata {
    model_name: str
    total_layers: int
    status: ShardStatus  # DOWNLOADING, PARTIAL, READY, FAILED
    shard_assignments: Dict[worker_id, ShardAssignment]
}

ShardAssignment {
    worker_id: str
    start_layer: int
    end_layer: int
    next_node_id: str
    next_node_address: str
    download_status: DownloadStatus  # PENDING, IN_PROGRESS, COMPLETED, FAILED
}
```

### Model Info

```python
ModelInfo {
    model_path: str
    model_name: str
    model: Any  # MLX model
    tokenizer: Any
    loaded_at: float
    memory_usage_gb: float
    max_tokens: int
    context_length: int
    shard_metadata: Optional[Dict]  # Present if sharded
    
    @property
    is_sharded: bool  # True if shard_metadata is not None
}
```

### Pipeline Configuration

```python
PipelineConfig {
    stages: List[PipelineStage]
}

PipelineStage {
    worker_id: str
    worker_address: str
    start_layer: int
    end_layer: int
    is_first: bool
    is_last: bool
}
```

---

## Component Status

### Implemented and Working ✅

1. **ShardingCoordinator** - Shard plan creation and metadata tracking
2. **ModelLoader** - Partial model loading with layer assignments
3. **InferenceEngine** - Sharding detection and routing
4. **DistributedForwardPass** - Multi-worker pipeline execution
5. **DistributedInferenceService** - Inter-worker gRPC communication
6. **RequestRouter** - Sharded model detection and routing
7. **LayerWorker** - Layer-specific forward pass execution

### Partially Implemented ⚠️

1. **Sampling** - Only greedy sampling for sharded models (no temperature/top_p)
2. **KV Cache** - Not implemented for distributed pipeline
3. **Failure Recovery** - No mid-inference worker failure recovery

### Not Implemented ❌

1. **Dynamic Rebalancing** - Pipeline topology is static
2. **Advanced Sampling** - Temperature, top_p, top_k for sharded inference
3. **Distributed KV Cache** - Cache sharing across pipeline stages
4. **Worker Migration** - Moving shards between workers

---

## Key Decision Points

### 1. Sharding Decision (Coordinator)
- **Location:** `coordinator/sharding_coordinator.py:should_shard_model()`
- **Inputs:** model_size_gb, num_workers, config
- **Output:** Boolean (shard or not)
- **Logged:** Yes, with structured context

### 2. Inference Routing (Coordinator)
- **Location:** `coordinator/request_router.py:route_inference()`
- **Inputs:** model_name, request
- **Output:** Target worker(s)
- **Logged:** Yes, with routing decision

### 3. Inference Path Selection (Worker)
- **Location:** `worker/inference_engine.py:process_inference()`
- **Inputs:** model_info.is_sharded
- **Output:** Standard or sharded inference path
- **Logged:** Yes, with routing decision

### 4. Pipeline Stage Execution (Worker)
- **Location:** `worker/distributed_forward.py:_execute_stage()`
- **Inputs:** activations, stage config
- **Output:** Next stage activations
- **Logged:** Yes, with stage timing

---

## Observability

### Structured Logging

All key operations include structured logging with:
- `request_id`: Unique request identifier
- `pipeline_id`: Pipeline execution identifier
- `worker_id`: Worker handling operation
- `operation`: Operation type (e.g., "sharding_decision", "inference_routing")
- Context-specific fields (model, layers, latency, etc.)

### Metrics

Key metrics tracked:
- `sharded_inference_count`: Total sharded requests
- `sharded_inference_latencies`: Latency distribution
- `shard_download_status`: Per-worker download status
- `activation_transfer_bytes`: Data transferred between workers
- `pipeline_stage_latency`: Per-stage execution time

### Health Checks

- **Coordinator:** `GET /health` - Overall system health
- **Worker:** `GET /health` - Worker-specific health
- **Shard Status:** `GetShardStatus` RPC - Per-model shard status

---

## References

- **Requirements:** `.kiro/specs/sharding-cleanup-integration/requirements.md`
- **Design:** `.kiro/specs/sharding-cleanup-integration/design.md`
- **Sharding Guide:** `docs/SHARDING_GUIDE.md`
- **User Guide:** `docs/USER_GUIDE.md`
