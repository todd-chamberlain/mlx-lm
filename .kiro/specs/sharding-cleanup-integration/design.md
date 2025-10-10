# Design Document - Sharding Feature Cleanup & Integration

## Overview

This design addresses the current state of confusion in the sharding implementation by conducting a systematic audit, cleanup, and proper integration of sharding functionality. The codebase currently has multiple partial implementations that create confusion about what actually works.

### Current State Analysis

**What Actually Exists:**
1. `SelectiveModelLoader` - Attempts memory-efficient layer loading (worker/selective_loader.py)
2. `ShardedModelLoader` - Partial implementation for downloading shards (worker/sharded_model_loader.py)
3. `ShardingCoordinator` - Coordinator-side sharding logic (coordinator/sharding_coordinator.py)
4. `LayerWorker` - Loads and processes specific layers (worker/layer_worker.py)
5. `DistributedForwardPass` - Pipeline execution across workers (worker/distributed_forward.py)
6. `ModelManager` - Has sharded loading methods but unclear integration (worker/model_manager.py)

**What's Actually Used:**
- `LayerWorker` is imported and used by `ModelManager.load_sharded_model()`
- `ShardedModelLoader` is imported by `ModelManager` but unclear if it works
- `SelectiveModelLoader` is only used in tests, not production code
- `DistributedForwardPass` exists but unclear if it's wired up to inference paths
- `ShardingCoordinator` is instantiated in `CoordinatorService` but integration is incomplete

**Key Problems:**
1. Multiple implementations of similar functionality (ShardedModelLoader vs SelectiveModelLoader)
2. Unclear which code paths are actually executed in production
3. Tests use mocks extensively, hiding real integration issues
4. Documentation claims features work that may not be fully integrated
5. No clear entry point for "how does sharding actually happen?"

## Architecture

### Phase 1: Audit & Documentation

Create a comprehensive audit document that maps:

```
Component Inventory:
├── Sharding Decision Logic
│   ├── ShardingCoordinator.should_shard_model() [EXISTS, UNCLEAR IF USED]
│   └── Config: enable_auto_sharding [EXISTS IN CODE]
│
├── Model Download & Loading
│   ├── ShardedModelLoader.download_shard() [EXISTS, UNCLEAR IF WORKS]
│   ├── SelectiveModelLoader.load_partial_model() [EXISTS, TEST ONLY]
│   ├── ModelManager.load_sharded_model() [EXISTS, CALLS LayerWorker]
│   └── LayerWorker.load_layers() [EXISTS, USED]
│
├── Inference Execution
│   ├── InferenceEngine.process_inference() [CHECK IF ROUTES TO SHARDED]
│   ├── DistributedForwardPass.forward() [EXISTS, UNCLEAR IF USED]
│   └── RequestRouter.route_inference() [CHECK SHARDING LOGIC]
│
└── Coordinator Integration
    ├── CoordinatorService.DownloadModel() [HAS SHARDING LOGIC]
    ├── CoordinatorService.DeleteModel() [HAS SHARDING LOGIC]
    └── ShardingCoordinator [INSTANTIATED, UNCLEAR USAGE]
```

### Phase 2: Consolidation Strategy

**Decision Matrix:**

| Component | Keep | Remove | Reason |
|-----------|------|--------|--------|
| ShardedModelLoader | TBD | TBD | Test if download_shard() actually works |
| SelectiveModelLoader | TBD | TBD | Test if load_partial_model() actually works |
| ShardingCoordinator | Keep | - | Core coordination logic, well-structured |
| LayerWorker | Keep | - | Actually used, core functionality |
| DistributedForwardPass | Keep | - | Core pipeline execution |
| ModelManager sharding methods | Keep | - | Integration point, needs fixing |

**Consolidation Plan:**

1. **Model Loading**: Choose ONE implementation
   - Test both ShardedModelLoader and SelectiveModelLoader with real models
   - Keep the one that actually works
   - If neither works, fix the better-designed one
   - Remove the other completely

2. **Inference Routing**: Establish clear path
   - Trace from API request → InferenceEngine → actual execution
   - Ensure sharded models route to DistributedForwardPass
   - Ensure non-sharded models route to standard inference
   - Remove any dead routing code

3. **Coordinator Integration**: Complete the wiring
   - Ensure ShardingCoordinator is actually called
   - Ensure shard metadata is actually stored and retrieved
   - Ensure download coordination actually works
   - Remove any unused coordinator methods

### Phase 3: Integration Architecture

**Clear Data Flow:**

```
1. Model Download (Sharded):
   API Request
   → CoordinatorService.DownloadModel()
   → ShardingCoordinator.should_shard_model() [DECISION]
   → ShardingCoordinator.create_shard_plan()
   → Send layer assignments to workers
   → Workers: ModelManager.load_sharded_model()
   → Workers: [ShardedModelLoader OR SelectiveModelLoader].download_shard()
   → Workers: LayerWorker.load_layers()
   → Store shard metadata in ModelInfo
   → Report back to coordinator
   → ShardingCoordinator.update_shard_status()

2. Inference (Sharded Model):
   API Request
   → CoordinatorService.RouteInference()
   → RequestRouter.route_inference()
   → Check if model is sharded (ModelInfo.is_sharded)
   → IF SHARDED:
       → Route to first worker in pipeline
       → Worker: InferenceEngine.process_inference()
       → Detect sharded model
       → Use DistributedForwardPass.forward()
       → Pipeline through workers
       → Return final output
   → IF NOT SHARDED:
       → Standard inference path
       → Return output

3. Model Deletion (Sharded):
   API Request
   → CoordinatorService.DeleteModel()
   → ShardingCoordinator.get_shard_metadata()
   → IF SHARDED:
       → Get list of workers with shards
       → Send delete to each worker
       → ShardingCoordinator.remove_shard_metadata()
   → IF NOT SHARDED:
       → Send delete to all workers
```

## Components and Interfaces

### 1. Unified Model Loader Interface

```python
class ModelLoader(ABC):
    """Abstract interface for model loading"""
    
    @abstractmethod
    def load_model(
        self,
        model_name: str,
        layer_assignment: Optional[LayerAssignment] = None
    ) -> Tuple[Any, Any, Optional[Dict]]:
        """
        Load a model (full or partial).
        
        Returns:
            (model, tokenizer, shard_metadata)
        """
        pass
```

**Implementation Decision:**
- Test both ShardedModelLoader and SelectiveModelLoader
- Keep the working one, remove the other
- If neither works, fix SelectiveModelLoader (better design)
- Rename to just `ModelLoader` (no "Sharded" or "Selective" prefix)

### 2. Clear Inference Routing

```python
class InferenceEngine:
    def process_inference(self, request):
        model_info = self.model_manager.get_model_info_dict(request.model)
        
        if model_info and model_info.get('is_sharded'):
            # Use distributed pipeline
            return self._process_sharded_inference(request, model_info)
        else:
            # Use standard inference
            return self._process_standard_inference(request)
    
    def _process_sharded_inference(self, request, model_info):
        # Get shard metadata
        shard_metadata = model_info['shard_metadata']
        
        # Create or get distributed forward pass
        pipeline = self._get_or_create_pipeline(shard_metadata)
        
        # Execute through pipeline
        return pipeline.forward(request.input_ids)
    
    def _process_standard_inference(self, request):
        # Existing standard inference logic
        pass
```

### 3. Coordinator Integration

```python
class CoordinatorService:
    async def DownloadModel(self, request, context):
        # Determine if sharding should be used
        should_shard = self.sharding_coordinator.should_shard_model(
            model_name=request.model_name,
            model_size_gb=request.model_size_gb,
            num_workers=len(healthy_workers)
        )
        
        if should_shard:
            # Create shard plan
            shard_metadata = self.sharding_coordinator.create_shard_plan(
                model_name=request.model_name,
                num_layers=request.num_layers,
                workers=worker_info_list
            )
            
            # Send layer assignments to workers
            for worker in healthy_workers:
                assignment = shard_metadata.get_assignment(worker.worker_id)
                # Send download request with assignment
                ...
        else:
            # Standard download to all workers
            ...
```

## Data Models

### Shard Metadata Structure

```python
@dataclass
class ShardMetadata:
    """Metadata stored in ModelInfo for sharded models"""
    start_layer: int
    end_layer: int
    worker_id: str
    next_worker_id: Optional[str]
    next_worker_address: Optional[str]
    is_first_worker: bool
    is_last_worker: bool
    total_layers: int
    download_status: str  # "completed", "in_progress", "failed"
```

### Model Info Enhancement

```python
@dataclass
class ModelInfo:
    # Existing fields...
    shard_metadata: Optional[ShardMetadata] = None
    
    @property
    def is_sharded(self) -> bool:
        return self.shard_metadata is not None
```

## Error Handling

### Clear Error Messages

```python
class ShardingError(MLXError):
    """Base class for sharding errors"""
    pass

class ShardDownloadError(ShardingError):
    """Shard download failed"""
    pass

class ShardInferenceError(ShardingError):
    """Sharded inference failed"""
    pass

class InsufficientWorkersError(ShardingError):
    """Not enough workers for sharding"""
    pass
```

### Graceful Fallbacks

1. **Download Failure**: If shard download fails, fall back to full model download
2. **Inference Failure**: If sharded inference fails, return clear error (don't fall back)
3. **Worker Failure**: If worker goes offline, mark model as unavailable

## Testing Strategy

### Real Integration Tests

```python
class TestShardingEndToEnd:
    """Real end-to-end tests with actual models"""
    
    def test_small_model_sharded_inference(self):
        """Test with a small model (e.g., TinyLlama) that can actually run"""
        # 1. Start coordinator and 2 workers
        # 2. Download model with sharding enabled
        # 3. Verify each worker has only assigned layers
        # 4. Run inference request
        # 5. Verify output is correct
        # 6. Measure memory usage per worker
        pass
    
    def test_sharding_disabled_still_works(self):
        """Ensure non-sharded path still works"""
        # 1. Start coordinator and workers
        # 2. Download model with sharding disabled
        # 3. Verify all workers have full model
        # 4. Run inference
        # 5. Verify output is correct
        pass
```

### Component Tests

```python
class TestModelLoader:
    """Test the chosen model loader implementation"""
    
    def test_load_partial_model_real(self):
        """Test loading partial model with real weights"""
        # Use a small real model
        # Verify only assigned layers are loaded
        # Measure memory usage
        pass
```

## Migration Path

### Phase 1: Audit (Week 1)
1. Create audit document mapping all sharding code
2. Test each component individually
3. Document what works vs what doesn't
4. Create removal list

### Phase 2: Cleanup (Week 1-2)
1. Remove confirmed dead code
2. Remove unused tests
3. Update documentation to remove false claims
4. Consolidate duplicate implementations

### Phase 3: Fix Integration (Week 2-3)
1. Wire up sharding decision logic
2. Fix model loading integration
3. Fix inference routing
4. Add proper error handling

### Phase 4: Test & Validate (Week 3-4)
1. Create real integration tests
2. Test with small models end-to-end
3. Measure memory savings
4. Validate performance
5. Update documentation with truth

## Success Criteria

1. **Code Clarity**: Any developer can trace the sharding path from API to execution
2. **Single Implementation**: One clear implementation of each feature
3. **Real Tests**: Integration tests use real models and services
4. **Accurate Docs**: Documentation matches actual functionality
5. **Production Ready**: Sharding works end-to-end with real models
6. **Backward Compatible**: Non-sharded inference still works perfectly
7. **Clear Errors**: Failures have clear, actionable error messages

## Non-Goals

- Performance optimization (focus on correctness first)
- Support for >2 workers (start with 2-worker sharding)
- Dynamic rebalancing (static shard assignments only)
- Automatic model size detection (require explicit configuration)
