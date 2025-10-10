# Implementation Plan - Automatic Model Sharding

## Overview

This implementation plan builds on existing distributed inference infrastructure (DistributedForwardPass, ActivationTransfer, LayerWorker, ModelPartitioner) to add automatic model sharding on download. The system already supports pipeline inference across workers; this adds automatic layer-based partitioning during model download.

## Tasks

- [-] 1. Sharding Configuration

  - [x] 1.1 Add sharding configuration to base config
    - Add ShardingConfig class to config/base.py with enable_auto_sharding, shard_threshold_gb, compression settings
    - Add sharding config to CoordinatorConfig in config/coordinator_config.py
    - Add environment variable support (ENABLE_AUTO_SHARDING, SHARD_THRESHOLD_GB)
    - _Requirements: 4.1, 4.2, 4.3, 4.4_

- [x] 2. Sharding Coordinator

  - [x] 2.1 Create ShardingCoordinator class in coordinator/sharding_coordinator.py

    - Implement should_shard_model(model_name, model_size_gb, num_workers) to determine if sharding is needed
    - Implement create_shard_plan(model_name, num_layers, workers) using existing ModelPartitioner logic
    - Implement get_shard_metadata(model_name) for metadata retrieval
    - Store shard metadata in coordinator state
    - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.3, 2.4, 4.1, 4.4_

  - [x] 2.2 Integrate ShardingCoordinator with CoordinatorService

    - Add ShardingCoordinator instance to CoordinatorService
    - Modify DownloadModel RPC to check if sharding should be enabled
    - Create shard plan and send layer assignments to workers
    - Track download status per worker
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 5.1_

  - [x] 2.3 Add shard metadata to ModelInfo
    - Extend worker/model_manager.py ModelInfo dataclass with shard_metadata field
    - Add is_sharded property to ModelInfo
    - Update list_loaded_models to include shard information
    - _Requirements: 5.1, 5.2, 10.1, 10.3_

- [x] 3. Sharded Model Download

  - [x] 3.1 Create ShardedModelLoader class in worker/sharded_model_loader.py

    - Implement download_shard(model_name, layer_assignment) for partial download
    - Use HuggingFace Hub API to download only specific layer weight files
    - Implement get_layer_count(model_name) by reading config.json
    - Add download progress tracking and logging
    - _Requirements: 1.3, 2.1, 2.2, 6.1, 6.2_

  - [x] 3.2 Integrate ShardedModelLoader with ModelManager

    - Add load_sharded_model(model_name, layer_assignment) method to ModelManager
    - Modify load_model to check if model is sharded and route appropriately
    - Use LayerWorker for loading only assigned layers
    - Store shard metadata in ModelInfo
    - _Requirements: 1.3, 5.1, 5.2, 10.1_

  - [x] 3.3 Add download coordination in worker gRPC service
    - Extend DownloadModel RPC to accept layer_assignment parameter
    - Call ShardedModelLoader when layer_assignment is provided
    - Return download status with shard information
    - Handle download failures and cleanup
    - _Requirements: 1.4, 1.5, 8.1, 8.2_

- [x] 4. Inference Routing for Sharded Models

  - [x] 4.1 Update InferenceEngine to detect sharded models

    - Add check in process_inference to detect if model is sharded
    - Route sharded models to DistributedForwardPass pipeline
    - Route non-sharded models to standard inference path
    - Maintain backward compatibility
    - _Requirements: 3.1, 10.1, 10.2, 10.3_

  - [x] 4.2 Update RequestRouter in coordinator for sharded inference

    - Detect sharded models in route_inference method
    - Route to first worker in pipeline for sharded models
    - Collect responses from pipeline
    - Add sharded inference metrics tracking
    - _Requirements: 3.1, 3.2, 9.1, 9.3_

  - [x] 4.3 Add streaming support for sharded inference
    - Extend stream_inference to support sharded models
    - Use DistributedGenerator for token streaming through pipeline
    - Handle backpressure across workers
    - Stream tokens to client as generated
    - _Requirements: 3.1, 3.2, 6.4_

- [x] 5. Model Deletion for Sharded Models

  - [x] 5.1 Update coordinator DeleteModel RPC

    - Check if model is sharded in coordinator metadata
    - Send delete requests to all workers holding shards
    - Clean up shard metadata from coordinator
    - Verify deletion on all workers
    - _Requirements: 5.3_

  - [x] 5.2 Update worker unload_model for sharded models
    - Handle partial model unloading in ModelManager
    - Clean up LayerWorker resources
    - Return shard-specific status
    - _Requirements: 5.3_

- [x] 6. API Endpoints and Admin Interface

  - [x] 6.1 Add sharding parameter to API download endpoint

    - Add optional enable_sharding parameter to POST /models/download
    - Update api/main.py to pass sharding flag to coordinator
    - Add validation for minimum worker count
    - Return shard plan in response
    - _Requirements: 4.3, 5.1_

  - [x] 6.2 Create shard status endpoint

    - Add GET /admin/models/{model_name}/shards endpoint
    - Return shard assignments per worker
    - Return download/load status per shard
    - Include inference metrics for sharded models
    - _Requirements: 5.2, 9.1, 9.2, 9.3_

  - [x] 6.3 Update health endpoint for sharded models
    - Extend GET /health to include sharded model count
    - Show shard status per model
    - Indicate if any shards are offline
    - Add shard-specific health checks
    - _Requirements: 5.4, 5.5, 8.5_

- [x] 7. Monitoring and Metrics

  - [x] 7.1 Add sharding-specific metrics to coordinator

    - Add sharded_models_total counter
    - Add sharded_inference_requests_total counter
    - Add sharded_inference_latency_seconds histogram
    - Add shard_download_status gauge per worker
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

  - [x] 7.2 Add sharding metrics to worker

    - Add shard_layer_range info metric
    - Add pipeline_stage_latency_seconds histogram
    - Extend existing activation_transfer metrics for sharding context
    - Track memory usage per shard
    - _Requirements: 7.1, 7.2, 7.3, 9.5_

  - [x] 7.3 Add structured logging for sharded operations
    - Log shard plan creation with worker assignments
    - Log download progress per worker with layer ranges
    - Log inference pipeline execution with stage timings
    - Log errors with shard context
    - _Requirements: 8.4, 9.4_

- [x] 8. Integration Testing

  - [x] 8.1 Create end-to-end sharding test

    - Write test_sharded_model_download.py to test download with 2 workers
    - Verify each worker loads only assigned layers
    - Verify shard metadata is stored correctly
    - Test inference through sharded model
    - _Requirements: 1.1, 1.3, 2.1, 3.1, 6.1, 6.2_

  - [x] 8.2 Create backward compatibility test

    - Write test_sharding_backward_compatibility.py
    - Test non-sharded model loading still works
    - Test mixed sharded/non-sharded workload
    - Verify no regressions in existing functionality
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5_

  - [x] 8.3 Create performance benchmark test
    - Write test_sharding_performance.py
    - Measure latency overhead for sharded vs non-sharded
    - Measure throughput with concurrent requests
    - Validate < 30% overhead requirement
    - Measure memory savings per worker
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 6.1, 6.2_

- [x] 9. Documentation

  - [x] 9.1 Create user guide for automatic sharding

    - Write docs/AUTOMATIC_SHARDING_GUIDE.md
    - Document how to enable sharding via config and API
    - Provide usage examples with curl/Python
    - Add troubleshooting section for common issues
    - Document memory savings and performance characteristics
    - _Requirements: 4.1, 4.2, 4.3_

  - [x] 9.2 Update API documentation

    - Update docs/USER_GUIDE.md with sharding endpoints
    - Document enable_sharding parameter for download endpoint
    - Document /admin/models/{model_name}/shards endpoint
    - Add example requests/responses
    - _Requirements: 4.3, 5.1, 5.2_

  - [x] 9.3 Update deployment documentation
    - Update DEPLOYMENT_GUIDE with sharding prerequisites
    - Document minimum worker requirements (2+ workers)
    - Document network latency requirements
    - Add monitoring setup for sharded models
    - _Requirements: 4.1, 9.1, 9.4_

## Progress Tracking

**Status:** Not Started
**Target Completion:** 3-4 weeks
**Current Phase:** Planning

**Phase 1 (Week 1):** Tasks 1-3 - Configuration, Coordinator, and Download
**Phase 2 (Week 2):** Tasks 4-5 - Inference Routing and Model Management
**Phase 3 (Week 3):** Tasks 6-7 - API Endpoints and Monitoring
**Phase 4 (Week 4):** Tasks 8-9 - Testing and Documentation

**Overall:** 0/27 tasks complete (0%)

## Notes

- Existing infrastructure (DistributedForwardPass, ActivationTransfer, LayerWorker, ModelPartitioner) provides foundation
- Focus is on automatic sharding during download and routing logic
- Pipeline inference already implemented, just needs integration with sharding metadata
- Activation compression (ZSTD) already implemented in ActivationTransfer
- KV cache management already implemented in DistributedGenerator
