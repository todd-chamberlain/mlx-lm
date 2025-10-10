# Sharding API Reference

**gRPC API reference for model sharding functionality**

## Overview

The sharding functionality is exposed through gRPC endpoints on the Coordinator service. This document describes the actual implemented API, not planned features.

---

## Table of Contents

1. [Service Definition](#service-definition)
2. [Download Model](#download-model)
3. [Get Shard Status](#get-shard-status)
4. [Delete Model](#delete-model)
5. [Error Responses](#error-responses)
6. [Python Examples](#python-examples)

---

## Service Definition

**Service:** `CoordinatorService`
**Port:** 50051 (default)
**Protocol:** gRPC

```protobuf
service CoordinatorService {
  rpc DownloadModel(ModelDownloadRequest) returns (ModelDownloadResponse);
  rpc GetShardStatus(ShardStatusRequest) returns (ShardStatusResponse);
  rpc DeleteModel(ModelDeleteRequest) returns (ModelDeleteResponse);
  // ... other RPCs
}
```

---

## Download Model

Downloads a model to workers. The coordinator automatically decides whether to shard based on configuration and model size.

### Request

```protobuf
message ModelDownloadRequest {
  string model_name = 1;        // HuggingFace model identifier
  bool force_redownload = 2;    // Force re-download if already exists
}
```

**Fields:**
- `model_name` (required): HuggingFace model identifier (e.g., "meta-llama/Llama-2-70b-chat-hf")
- `force_redownload` (optional): If true, re-download even if model exists (default: false)

**Note:** The `enable_sharding` field shown in some documentation does not exist in the actual proto definition. Sharding is automatically determined by the coordinator based on:
- Model size exceeds `shard_threshold_gb` configuration
- Number of healthy workers >= `min_workers_for_sharding`
- `enable_auto_sharding` is true in configuration

### Response

```protobuf
message ModelDownloadResponse {
  bool success = 1;
  string message = 2;
  int32 workers_notified = 3;
  bool is_sharded = 4;
  ShardPlan shard_plan = 5;  // Only present if is_sharded=true
}

message ShardPlan {
  string model_name = 1;
  int32 total_layers = 2;
  repeated ShardAssignment assignments = 3;
}

message ShardAssignment {
  string worker_id = 1;
  int32 start_layer = 2;
  int32 end_layer = 3;
  string next_worker_id = 4;
}
```

**Fields:**
- `success`: Whether download was initiated successfully
- `message`: Human-readable status message
- `workers_notified`: Number of workers that received download request
- `is_sharded`: Whether model was sharded across workers
- `shard_plan`: Shard plan details (only if `is_sharded=true`)

### Example

**Python:**

```python
import grpc
from proto.inference_pb2 import ModelDownloadRequest
from proto.inference_pb2_grpc import CoordinatorServiceStub

# Connect to coordinator
channel = grpc.insecure_channel('localhost:50051')
stub = CoordinatorServiceStub(channel)

# Download model
request = ModelDownloadRequest(
    model_name="meta-llama/Llama-2-70b-chat-hf",
    force_redownload=False
)

response = stub.DownloadModel(request)

print(f"Success: {response.success}")
print(f"Message: {response.message}")
print(f"Workers notified: {response.workers_notified}")
print(f"Is sharded: {response.is_sharded}")

if response.is_sharded:
    print(f"\nShard Plan:")
    print(f"  Model: {response.shard_plan.model_name}")
    print(f"  Total layers: {response.shard_plan.total_layers}")
    print(f"  Assignments:")
    for assignment in response.shard_plan.assignments:
        print(f"    {assignment.worker_id}: layers {assignment.start_layer}-{assignment.end_layer}")
        if assignment.next_worker_id:
            print(f"      → next: {assignment.next_worker_id}")
```

**Output:**

```
Success: True
Message: Sharded download initiated across 2 workers
Workers notified: 2
Is sharded: True

Shard Plan:
  Model: meta-llama/Llama-2-70b-chat-hf
  Total layers: 80
  Assignments:
    worker-1: layers 0-40
      → next: worker-2
    worker-2: layers 40-80
```

---

## Get Shard Status

Queries the status of a sharded model, including per-worker download status and inference metrics.

### Request

```protobuf
message ShardStatusRequest {
  string model_name = 1;  // Model to query
}
```

**Fields:**
- `model_name` (required): Model identifier to query

### Response

```protobuf
message ShardStatusResponse {
  string model_name = 1;
  bool is_sharded = 2;
  int32 total_layers = 3;
  string overall_status = 4;  // "DOWNLOADING", "PARTIAL", "READY", "FAILED"
  repeated ShardAssignmentStatus shard_assignments = 5;
}

message ShardAssignmentStatus {
  string worker_id = 1;
  int32 start_layer = 2;
  int32 end_layer = 3;
  string download_status = 4;  // "PENDING", "IN_PROGRESS", "COMPLETED", "FAILED"
  float progress_percent = 5;
  string error = 6;  // Error message if failed
}
```

**Fields:**
- `model_name`: Model identifier
- `is_sharded`: Whether model is sharded
- `total_layers`: Total number of layers in model
- `overall_status`: Overall shard status
  - `DOWNLOADING`: Download in progress
  - `PARTIAL`: Some shards complete, some not
  - `READY`: All shards complete and ready
  - `FAILED`: One or more shards failed
- `shard_assignments`: Per-worker shard status

### Example

**Python:**

```python
from proto.inference_pb2 import ShardStatusRequest

request = ShardStatusRequest(
    model_name="meta-llama/Llama-2-70b-chat-hf"
)

response = stub.GetShardStatus(request)

print(f"Model: {response.model_name}")
print(f"Is sharded: {response.is_sharded}")
print(f"Total layers: {response.total_layers}")
print(f"Overall status: {response.overall_status}")

print(f"\nShard Assignments:")
for shard in response.shard_assignments:
    print(f"  {shard.worker_id}:")
    print(f"    Layers: {shard.start_layer}-{shard.end_layer}")
    print(f"    Status: {shard.download_status}")
    print(f"    Progress: {shard.progress_percent:.1f}%")
    if shard.error:
        print(f"    Error: {shard.error}")
```

**Output:**

```
Model: meta-llama/Llama-2-70b-chat-hf
Is sharded: True
Total layers: 80
Overall status: READY

Shard Assignments:
  worker-1:
    Layers: 0-40
    Status: COMPLETED
    Progress: 100.0%
  worker-2:
    Layers: 40-80
    Status: COMPLETED
    Progress: 100.0%
```

---

## Delete Model

Deletes a model from all workers. For sharded models, ensures all shards are deleted before removing metadata.

### Request

```protobuf
message ModelDeleteRequest {
  string model_name = 1;  // Model to delete
}
```

**Fields:**
- `model_name` (required): Model identifier to delete

### Response

```protobuf
message ModelDeleteResponse {
  bool success = 1;
  string message = 2;
  int32 workers_notified = 3;
}
```

**Fields:**
- `success`: Whether deletion was successful
- `message`: Human-readable status message
- `workers_notified`: Number of workers that received delete request

### Example

**Python:**

```python
from proto.inference_pb2 import ModelDeleteRequest

request = ModelDeleteRequest(
    model_name="meta-llama/Llama-2-70b-chat-hf"
)

response = stub.DeleteModel(request)

print(f"Success: {response.success}")
print(f"Message: {response.message}")
print(f"Workers notified: {response.workers_notified}")
```

**Output:**

```
Success: True
Message: Model deleted successfully from all workers
Workers notified: 2
```

**Note:** For sharded models, the coordinator:
1. Identifies which workers have shards
2. Sends delete requests only to those workers
3. Verifies all shards are deleted
4. Removes shard metadata only if all deletions succeed

---

## Error Responses

gRPC errors are returned with standard status codes:

### INVALID_ARGUMENT (Code 3)

**Cause:** Invalid request parameters

**Example:**
```python
try:
    request = ModelDownloadRequest(model_name="")
    response = stub.DownloadModel(request)
except grpc.RpcError as e:
    print(f"Error: {e.code()}")  # INVALID_ARGUMENT
    print(f"Details: {e.details()}")  # "model_name is required"
```

### UNAVAILABLE (Code 14)

**Cause:** No workers available or insufficient workers for sharding

**Example:**
```python
try:
    request = ModelDownloadRequest(model_name="large-model")
    response = stub.DownloadModel(request)
except grpc.RpcError as e:
    print(f"Error: {e.code()}")  # UNAVAILABLE
    print(f"Details: {e.details()}")  # "No healthy workers available"
```

### NOT_FOUND (Code 5)

**Cause:** Model not found

**Example:**
```python
try:
    request = ShardStatusRequest(model_name="nonexistent-model")
    response = stub.GetShardStatus(request)
except grpc.RpcError as e:
    print(f"Error: {e.code()}")  # NOT_FOUND
    print(f"Details: {e.details()}")  # "Model not found"
```

### INTERNAL (Code 13)

**Cause:** Internal server error

**Example:**
```python
try:
    request = ModelDownloadRequest(model_name="model")
    response = stub.DownloadModel(request)
except grpc.RpcError as e:
    print(f"Error: {e.code()}")  # INTERNAL
    print(f"Details: {e.details()}")  # "Failed to create shard plan: ..."
```

---

## Python Examples

### Complete Download and Monitor Flow

```python
import grpc
import time
from proto.inference_pb2 import (
    ModelDownloadRequest,
    ShardStatusRequest,
    InferenceRequest
)
from proto.inference_pb2_grpc import CoordinatorServiceStub

def download_and_monitor(model_name: str):
    """Download model and monitor until ready"""
    
    # Connect to coordinator
    channel = grpc.insecure_channel('localhost:50051')
    stub = CoordinatorServiceStub(channel)
    
    # Step 1: Initiate download
    print(f"Downloading {model_name}...")
    download_req = ModelDownloadRequest(
        model_name=model_name,
        force_redownload=False
    )
    
    try:
        download_resp = stub.DownloadModel(download_req)
    except grpc.RpcError as e:
        print(f"Download failed: {e.details()}")
        return False
    
    if not download_resp.success:
        print(f"Download failed: {download_resp.message}")
        return False
    
    print(f"Download initiated: {download_resp.message}")
    print(f"Sharded: {download_resp.is_sharded}")
    print(f"Workers: {download_resp.workers_notified}")
    
    # Step 2: Monitor progress
    print("\nMonitoring download progress...")
    status_req = ShardStatusRequest(model_name=model_name)
    
    while True:
        try:
            status_resp = stub.GetShardStatus(status_req)
        except grpc.RpcError as e:
            print(f"Status check failed: {e.details()}")
            time.sleep(5)
            continue
        
        print(f"\nOverall status: {status_resp.overall_status}")
        
        if status_resp.is_sharded:
            for shard in status_resp.shard_assignments:
                print(f"  {shard.worker_id}: {shard.download_status} ({shard.progress_percent:.1f}%)")
                if shard.error:
                    print(f"    Error: {shard.error}")
        
        if status_resp.overall_status == "READY":
            print("\n✓ Download complete!")
            return True
        elif status_resp.overall_status == "FAILED":
            print("\n✗ Download failed!")
            return False
        
        time.sleep(5)

# Usage
if __name__ == "__main__":
    success = download_and_monitor("meta-llama/Llama-2-70b-chat-hf")
    
    if success:
        print("\nModel ready for inference!")
```

### Check Multiple Models

```python
def check_all_sharded_models():
    """Check status of all sharded models"""
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = CoordinatorServiceStub(channel)
    
    # Get list of models (you'd need to track this separately)
    models = [
        "meta-llama/Llama-2-70b-chat-hf",
        "meta-llama/Llama-2-13b-chat-hf"
    ]
    
    for model_name in models:
        try:
            request = ShardStatusRequest(model_name=model_name)
            response = stub.GetShardStatus(request)
            
            if response.is_sharded:
                print(f"\n{model_name}:")
                print(f"  Status: {response.overall_status}")
                print(f"  Layers: {response.total_layers}")
                print(f"  Shards: {len(response.shard_assignments)}")
                
                healthy = sum(1 for s in response.shard_assignments 
                            if s.download_status == "COMPLETED")
                print(f"  Healthy: {healthy}/{len(response.shard_assignments)}")
        
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.NOT_FOUND:
                print(f"\n{model_name}: Not found")
            else:
                print(f"\n{model_name}: Error - {e.details()}")

# Usage
check_all_sharded_models()
```

### Error Handling Best Practices

```python
def safe_download(model_name: str, max_retries: int = 3):
    """Download with retry logic and error handling"""
    
    channel = grpc.insecure_channel('localhost:50051')
    stub = CoordinatorServiceStub(channel)
    
    for attempt in range(max_retries):
        try:
            request = ModelDownloadRequest(
                model_name=model_name,
                force_redownload=(attempt > 0)  # Force on retry
            )
            
            response = stub.DownloadModel(request, timeout=30)
            
            if response.success:
                return response
            else:
                print(f"Attempt {attempt + 1} failed: {response.message}")
        
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNAVAILABLE:
                print(f"Attempt {attempt + 1}: No workers available")
            elif e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
                print(f"Attempt {attempt + 1}: Timeout")
            else:
                print(f"Attempt {attempt + 1}: {e.details()}")
        
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    raise Exception(f"Failed to download {model_name} after {max_retries} attempts")

# Usage
try:
    response = safe_download("meta-llama/Llama-2-70b-chat-hf")
    print(f"Download successful: {response.message}")
except Exception as e:
    print(f"Download failed: {e}")
```

---

## Configuration

Sharding behavior is controlled by coordinator configuration:

```yaml
# config/coordinator.yaml
sharding:
  enabled: true                    # Enable automatic sharding
  threshold_gb: 30                 # Shard models > 30GB
  strategy: balanced               # 'balanced' or 'memory_aware'
  min_workers_for_sharding: 2      # Minimum workers required
```

**Environment Variables:**

```bash
export ENABLE_AUTO_SHARDING=true
export SHARD_THRESHOLD_GB=30
export MIN_WORKERS_FOR_SHARDING=2
```

---

## Limitations

1. **No REST API**: Sharding functionality is only available via gRPC, not REST
2. **Automatic Sharding Only**: Cannot manually specify shard plan
3. **No Progress Callbacks**: Must poll GetShardStatus for progress
4. **No Partial Deletion**: Cannot delete individual shards, only entire model

---

## Additional Resources

- **Sharding Guide**: `docs/SHARDING_GUIDE.md`
- **Architecture**: `docs/ARCHITECTURE_DIAGRAMS.md`
- **Proto Definitions**: `proto/inference.proto`
- **User Guide**: `docs/USER_GUIDE.md`

---

**Questions or issues?** Check the troubleshooting section in the Sharding Guide or review coordinator logs for detailed error messages.
