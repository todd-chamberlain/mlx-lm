# DeleteModel RPC Usage Guide

## Overview

The DeleteModel RPC allows you to delete models from workers, with automatic handling of both sharded and non-sharded models.

## gRPC API

### Coordinator DeleteModel

Delete a model from all workers (or just workers holding shards for sharded models).

**Request:**
```protobuf
message ModelDeleteRequest {
    string model_name = 1;
}
```

**Response:**
```protobuf
message ModelDeleteResponse {
    bool success = 1;
    string message = 2;
    int32 workers_deleted = 3;
    repeated WorkerDeleteStatus worker_status = 4;
}

message WorkerDeleteStatus {
    string worker_id = 1;
    bool success = 2;
    string error = 3;
}
```

### Example Usage (Python)

```python
import grpc
from proto.inference_pb2 import ModelDeleteRequest
from proto.inference_pb2_grpc import CoordinatorServiceStub

# Connect to coordinator
channel = grpc.insecure_channel('localhost:50051')
stub = CoordinatorServiceStub(channel)

# Delete a model
request = ModelDeleteRequest(model_name="meta-llama/Llama-2-70b-hf")
response = stub.DeleteModel(request)

if response.success:
    print(f"Model deleted from {response.workers_deleted} workers")
    print(f"Message: {response.message}")
else:
    print(f"Deletion failed: {response.message}")
    
# Check per-worker status
for worker_status in response.worker_status:
    if worker_status.success:
        print(f"✓ Worker {worker_status.worker_id}: deleted")
    else:
        print(f"✗ Worker {worker_status.worker_id}: {worker_status.error}")

channel.close()
```

## Behavior

### Non-Sharded Models
- Deletes model from **all workers** in the cluster
- Each worker unloads the complete model from memory
- Coordinator tracks deletion status per worker

### Sharded Models
- Identifies workers holding shards via coordinator metadata
- Deletes **only from workers holding shards**
- Each worker unloads its partial model (shard)
- Coordinator cleans up shard metadata
- More efficient than deleting from all workers

### Edge Cases

#### Model Not Loaded
If a model is not loaded on a worker, the worker returns success with a message indicating the model wasn't loaded. This is not considered an error.

#### Partial Failure
If deletion fails on some workers but succeeds on others:
- `success` will be `false`
- `workers_deleted` will show the count of successful deletions
- `worker_status` will detail which workers succeeded/failed
- Coordinator metadata is still cleaned up

#### Model Not Found
If the model doesn't exist on any worker:
- All workers return success (nothing to delete)
- Overall response is success
- Message indicates model was not found

## REST API Integration

The DeleteModel RPC can be exposed via REST API:

```python
# In api/main.py
@app.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete a model from all workers"""
    request = ModelDeleteRequest(model_name=model_name)
    response = await coordinator_stub.DeleteModel(request)
    
    return {
        "success": response.success,
        "message": response.message,
        "workers_deleted": response.workers_deleted,
        "worker_status": [
            {
                "worker_id": ws.worker_id,
                "success": ws.success,
                "error": ws.error
            }
            for ws in response.worker_status
        ]
    }
```

### Example REST API Call

```bash
# Delete a model
curl -X DELETE http://localhost:8000/models/meta-llama/Llama-2-70b-hf

# Response
{
  "success": true,
  "message": "Model deleted from 2/2 workers (sharded model)",
  "workers_deleted": 2,
  "worker_status": [
    {
      "worker_id": "worker-1",
      "success": true,
      "error": ""
    },
    {
      "worker_id": "worker-2",
      "success": true,
      "error": ""
    }
  ]
}
```

## Monitoring

### Logs

Coordinator logs:
```
INFO: Model deletion requested: meta-llama/Llama-2-70b-hf
INFO: Deleting sharded model meta-llama/Llama-2-70b-hf from 2 workers: ['worker-1', 'worker-2']
INFO: Cleaned up shard metadata for meta-llama/Llama-2-70b-hf
```

Worker logs:
```
INFO: Model deletion requested: meta-llama/Llama-2-70b-hf
INFO: Deleting sharded model meta-llama/Llama-2-70b-hf (layers 0-60)
INFO: Cleaned up LayerWorker resources for meta-llama/Llama-2-70b-hf
INFO: Successfully unloaded model: meta-llama/Llama-2-70b-hf
```

### Metrics

Consider adding these metrics for deletion operations:
- `model_deletions_total` - Counter of deletion requests
- `model_deletion_duration_seconds` - Histogram of deletion latency
- `model_deletion_failures_total` - Counter of failed deletions

## Best Practices

### 1. Check Model Status First
Before deleting, verify the model exists:
```python
status_response = stub.GetDownloadStatus(ModelStatusRequest(model_name=model_name))
if status_response.status == "not_found":
    print("Model not found, nothing to delete")
```

### 2. Handle Partial Failures
Always check per-worker status for production systems:
```python
if not response.success:
    failed_workers = [
        ws.worker_id for ws in response.worker_status 
        if not ws.success
    ]
    print(f"Deletion failed on workers: {failed_workers}")
    # Implement retry logic or alerting
```

### 3. Graceful Degradation
For sharded models, if deletion fails on some workers:
- The model becomes unavailable for inference
- Coordinator metadata is cleaned up
- Manual cleanup may be needed on failed workers

### 4. Concurrent Requests
Avoid deleting a model while inference requests are in flight:
- Check active requests before deletion
- Implement graceful draining if needed
- Consider maintenance windows for deletions

## Troubleshooting

### Issue: Deletion Hangs
**Cause:** Worker is unresponsive
**Solution:** Check worker health, restart if needed

### Issue: Partial Deletion
**Cause:** Network issues or worker failures
**Solution:** Retry deletion, check worker logs

### Issue: Memory Not Freed
**Cause:** Python garbage collection delay
**Solution:** Memory will be freed eventually; force GC if needed

### Issue: Shard Metadata Remains
**Cause:** Coordinator cleanup failed
**Solution:** Metadata is cleaned up even on partial failure; check coordinator logs

## Security Considerations

### Authorization
Add authorization checks before allowing deletion:
```python
if not user.has_permission("delete_models"):
    raise PermissionError("User not authorized to delete models")
```

### Audit Logging
Log all deletion requests for audit trails:
```python
audit_log.info(
    "Model deletion",
    user=user.id,
    model=model_name,
    success=response.success,
    workers_affected=response.workers_deleted
)
```

### Rate Limiting
Implement rate limiting to prevent abuse:
```python
@limiter.limit("10/minute")
async def delete_model(model_name: str):
    ...
```

## Related Documentation

- [Automatic Sharding Guide](../docs/AUTOMATIC_SHARDING_GUIDE.md)
- [Model Management API](../docs/USER_GUIDE.md#model-management)
- [Worker Management](../docs/DEPLOYMENT_GUIDE.md#worker-management)
