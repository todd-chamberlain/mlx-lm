# Implementation Plan - Sharding Feature Cleanup & Integration

## Overview

This plan systematically audits, cleans up, and properly integrates the sharding functionality. The focus is on establishing what actually works, removing what doesn't, and creating a single clear implementation path.

## Tasks

- [x] 1. Comprehensive Code Audit

  - [x] 1.1 Create audit document mapping all sharding-related code

    - List all files with "shard" in name or containing sharding logic
    - Map import relationships (what imports what)
    - Identify entry points (where is code actually called from)
    - Document test coverage (real tests vs mocked tests)
    - _Requirements: 1.1, 1.2, 1.3_

  - [x] 1.2 Test each model loading implementation

    - Test ShardedModelLoader.download_shard() with a small real model
    - Test SelectiveModelLoader.load_partial_model() with a small real model
    - Measure actual memory usage for each
    - Document which one actually works
    - _Requirements: 1.4, 3.1, 3.2_

  - [x] 1.3 Trace inference execution paths

    - Add debug logging to InferenceEngine.process_inference()
    - Run inference with sharded model, capture logs
    - Run inference with non-sharded model, capture logs
    - Document actual code paths taken
    - Identify if DistributedForwardPass is actually used
    - _Requirements: 1.1, 3.1, 3.3_

  - [x] 1.4 Verify coordinator integration
    - Test CoordinatorService.DownloadModel() with sharding enabled
    - Verify ShardingCoordinator.should_shard_model() is actually called
    - Verify shard metadata is actually stored
    - Test CoordinatorService.DeleteModel() with sharded model
    - Document what works vs what's broken
    - _Requirements: 1.1, 3.1, 5.1_

- [x] 2. Remove Dead Code

  - [x] 2.1 Remove unused model loader implementation

    - Based on audit, remove the non-working loader (ShardedModelLoader OR SelectiveModelLoader)
    - Remove associated tests
    - Update imports in ModelManager
    - _Requirements: 2.1, 2.2, 4.1, 4.4_

  - [x] 2.2 Remove orphaned test files

    - Identify test files that test removed code
    - Remove test files completely
    - Update test documentation
    - _Requirements: 2.1, 2.2_

  - [x] 2.3 Remove misleading documentation

    - Identify docs that claim features work that don't
    - Remove or mark as "planned" not "implemented"
    - Remove completion summary files for incomplete features
    - _Requirements: 2.3, 6.1, 6.2, 6.4_

  - [x] 2.4 Clean up duplicate implementations
    - Remove any duplicate helper functions
    - Consolidate similar logic into single functions
    - Remove commented-out code
    - _Requirements: 2.1, 4.1, 4.3_

- [x] 3. Consolidate Model Loading

  - [x] 3.1 Rename and clean chosen model loader

    - Rename to just `ModelLoader` (remove "Sharded" or "Selective" prefix)
    - Clean up the implementation (remove TODOs, fix issues found in audit)
    - Add clear docstrings explaining what it does
    - _Requirements: 4.1, 4.2, 4.4, 9.1, 9.2_

  - [x] 3.2 Update ModelManager integration

    - Simplify ModelManager.load_sharded_model() to use consolidated loader
    - Ensure shard_metadata is properly populated in ModelInfo
    - Add validation that layer assignment is valid
    - Add clear error messages for failures
    - _Requirements: 4.2, 5.1, 5.4, 9.3_

  - [x] 3.3 Add fallback logic
    - If sharded loading fails, log clear error
    - Optionally fall back to full model loading with warning
    - Track fallback metrics
    - _Requirements: 5.5, 8.1, 8.2_

- [x] 4. Fix Inference Routing

  - [x] 4.1 Add sharding detection to InferenceEngine

    - In process_inference(), check ModelInfo.is_sharded
    - Route to \_process_sharded_inference() if sharded
    - Route to \_process_standard_inference() if not sharded
    - Add structured logging for routing decisions
    - _Requirements: 5.1, 5.2, 5.3, 9.3, 9.4_

  - [x] 4.2 Implement \_process_sharded_inference()

    - Extract shard_metadata from ModelInfo
    - Create or get DistributedForwardPass instance
    - Execute pipeline.forward() with input
    - Handle errors with clear messages
    - _Requirements: 5.2, 5.3, 5.4, 9.3_

  - [x] 4.3 Verify DistributedForwardPass integration

    - Ensure DistributedForwardPass.forward() actually works
    - Test with 2 workers and small model
    - Verify activations flow correctly between workers
    - Fix any issues found
    - _Requirements: 3.1, 3.2, 5.2, 5.3_

  - [x] 4.4 Update RequestRouter for sharded models
    - In route_inference(), detect sharded models
    - Route to first worker in pipeline for sharded models
    - Add metrics for sharded vs non-sharded routing
    - _Requirements: 5.1, 5.2, 9.1_

- [x] 5. Complete Coordinator Integration

  - [x] 5.1 Fix DownloadModel RPC sharding logic

    - Ensure should_shard_model() is actually called
    - Ensure create_shard_plan() is actually called when sharding
    - Ensure layer assignments are sent to workers
    - Add structured logging for shard plan creation
    - _Requirements: 5.1, 5.2, 9.1_

  - [x] 5.2 Fix shard status tracking

    - Ensure update_shard_status() is called when workers report status
    - Verify shard metadata is actually stored in coordinator
    - Add endpoint to query shard status
    - _Requirements: 5.1, 5.2, 9.1_

  - [x] 5.3 Fix DeleteModel for sharded models

    - Ensure get_shard_metadata() is called
    - Ensure delete requests go to correct workers
    - Ensure remove_shard_metadata() is called after deletion
    - Add verification that all shards are deleted
    - _Requirements: 5.1, 5.2, 5.3_

  - [x] 5.4 Add configuration validation
    - Validate enable_auto_sharding config on startup
    - Validate shard_threshold_gb is reasonable
    - Validate min_workers_for_sharding >= 2
    - Log configuration on startup
    - _Requirements: 5.1, 8.1, 9.1_

- [x] 6. Establish Real Testing

  - [x] 6.1 Create end-to-end sharding test with real model

    - Use TinyLlama or similar small model
    - Start coordinator and 2 workers
    - Download model with sharding enabled
    - Verify each worker has only assigned layers (check memory)
    - Run inference request
    - Verify output is correct
    - _Requirements: 7.1, 7.2, 7.3, 10.1, 10.2_

  - [x] 6.2 Create backward compatibility test

    - Use same small model
    - Download with sharding disabled
    - Verify all workers have full model
    - Run inference
    - Verify output matches sharded output
    - _Requirements: 7.1, 7.2, 8.1, 10.1_

  - [x] 6.3 Create failure scenario tests

    - Test worker failure during download
    - Test worker failure during inference
    - Test insufficient workers for sharding
    - Verify error messages are clear
    - Get rid of any test files or script that aren't needed anymore
    - _Requirements: 7.1, 7.2, 5.4, 5.5_

  - [x] 6.4 Remove or fix mocked tests
    - Identify tests that are 100% mocked
    - Either add real integration or remove test
    - Keep unit tests for pure logic functions
    - Remove tests for removed code
    - _Requirements: 2.2, 7.1, 7.2_

- [x] 7. Update Documentation

  - [x] 7.1 Create accurate sharding guide

    - Document what actually works
    - Provide working examples with real commands
    - Document limitations clearly
    - Remove claims of features that don't work
    - Consolidate all these .md files & get rid of what is not useful
    - _Requirements: 6.1, 6.2, 6.3, 6.4_

  - [x] 7.2 Update architecture documentation

    - Document actual data flow (not planned flow)
    - Create clear diagrams showing component interactions
    - Document which components are used vs unused
    - _Requirements: 6.1, 6.2, 9.1, 9.2_

  - [x] 7.3 Update API documentation

    - Document sharding-related API parameters
    - Provide working curl examples
    - Document error responses
    - Remove documentation for removed features
    - _Requirements: 6.1, 6.2, 6.3_

  - [x] 7.4 Create troubleshooting guide
    - Document common issues and solutions
    - Provide debugging steps
    - Document how to verify sharding is working
    - Include memory measurement commands
    - _Requirements: 6.1, 6.4, 9.4_

- [x] 8. Simplify Architecture

  - [x] 8.1 Create clear component diagram

    - Show coordinator, workers, and data flow
    - Highlight sharding decision points
    - Show both sharded and non-sharded paths
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 8.2 Document interfaces clearly

    - Document ModelLoader interface
    - Document InferenceEngine routing logic
    - Document ShardingCoordinator API
    - _Requirements: 9.1, 9.2, 9.3_

  - [x] 8.3 Add code comments at decision points
    - Comment where sharding decisions are made
    - Comment where routing happens
    - Comment where shard metadata is used
    - _Requirements: 9.3, 9.4_

- [x] 9. Validation & Cleanup

  - [x] 9.1 Run all tests and verify they pass

    - Run unit tests
    - Run integration tests
    - Run end-to-end tests
    - Fix any failures
    - _Requirements: 10.1, 10.2_

  - [x] 9.2 Measure memory savings with real model

    - Load model sharded across 2 workers
    - Measure memory per worker
    - Load same model non-sharded
    - Measure memory per worker
    - Document actual savings
    - _Requirements: 10.3, 10.4_

  - [x] 9.3 Measure performance with real model

    - Run inference with sharded model
    - Run inference with non-sharded model
    - Compare latency
    - Document actual overhead
    - _Requirements: 10.3, 10.4_

  - [x] 9.4 Final code cleanup pass

    - Remove any remaining TODOs
    - Remove any remaining commented code
    - Fix any remaining linting issues
    - Ensure consistent naming
    - _Requirements: 2.1, 9.1, 9.2_

  - [x] 9.5 Update project README
    - Add clear section on sharding
    - Link to detailed sharding guide
    - Provide quick start example
    - Document requirements (2+ workers)
    - _Requirements: 6.1, 6.2, 6.3_

## Progress Tracking

**Status:** Not Started
**Target Completion:** 4 weeks
**Current Phase:** Planning

**Phase 1 (Week 1):** Tasks 1-2 - Audit and Remove Dead Code
**Phase 2 (Week 2):** Tasks 3-5 - Consolidate and Fix Integration
**Phase 3 (Week 3):** Tasks 6-7 - Real Testing and Documentation
**Phase 4 (Week 4):** Tasks 8-9 - Simplify and Validate

**Overall:** 0/39 tasks complete (0%)

## Notes

- Focus on correctness over performance
- Start with 2-worker sharding only
- Use small models (TinyLlama, Qwen-0.5B) for testing
- Backward compatibility is critical - don't break non-sharded inference
- Real integration tests are mandatory - no more mock-only tests
- Documentation must match reality - remove aspirational claims
