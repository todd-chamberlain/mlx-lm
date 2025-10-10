"""
Tests for sharded model deletion functionality.

Tests the DeleteModel RPC implementation for both sharded and non-sharded models.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from coordinator.service import CoordinatorService
from coordinator.worker_pool import WorkerPool
from coordinator.sharding_coordinator import ShardingCoordinator, ShardedModelMetadata, ShardStatus
from coordinator.models import WorkerInfo, WorkerStatus
from worker.model_manager import ModelManager, ModelInfo
from worker.grpc_service import InferenceServiceImpl
from worker.inference_engine import InferenceEngine
from worker.model_partitioner import LayerAssignment


@pytest.fixture
def mock_worker_pool():
    """Create a mock worker pool"""
    pool = Mock(spec=WorkerPool)
    
    # Create mock workers using Mock instead of actual WorkerInfo
    worker1 = Mock()
    worker1.worker_id = "worker-1"
    worker1.address = "localhost:8081"
    worker1.status = WorkerStatus.HEALTHY
    
    worker2 = Mock()
    worker2.worker_id = "worker-2"
    worker2.address = "localhost:8082"
    worker2.status = WorkerStatus.HEALTHY
    
    pool.get_all_workers.return_value = [worker1, worker2]
    return pool


@pytest.fixture
def sharding_coordinator():
    """Create a sharding coordinator"""
    return ShardingCoordinator(
        enable_auto_sharding=True,
        shard_threshold_gb=30.0,
        strategy="balanced"
    )


@pytest.fixture
def coordinator_service(mock_worker_pool, sharding_coordinator):
    """Create a coordinator service"""
    return CoordinatorService(mock_worker_pool, sharding_coordinator)


@pytest.fixture
def model_manager():
    """Create a model manager"""
    return ModelManager(model_cache_size=2)


@pytest.fixture
def inference_engine(model_manager):
    """Create an inference engine"""
    engine = Mock(spec=InferenceEngine)
    engine.worker_id = "test-worker"
    engine.get_metrics.return_value = {
        'avg_latency_ms': 100.0,
        'tokens_per_second': 50.0,
        'total_requests': 10,
        'failed_requests': 0,
        'active_requests': 0,
        'success_rate': 1.0
    }
    return engine


@pytest.fixture
def worker_service(model_manager, inference_engine):
    """Create a worker service"""
    return InferenceServiceImpl(model_manager, inference_engine)


class TestCoordinatorDeleteModel:
    """Test coordinator DeleteModel RPC"""
    
    @pytest.mark.asyncio
    async def test_delete_non_sharded_model(self, coordinator_service, mock_worker_pool):
        """Test deleting a non-sharded model"""
        from proto.inference_pb2 import ModelDeleteRequest, ModelDeleteResponse
        
        # Create request
        request = ModelDeleteRequest(model_name="test-model")
        context = Mock()
        
        # Mock worker responses
        with patch('grpc.aio.insecure_channel') as mock_channel:
            mock_stub = AsyncMock()
            mock_stub.DeleteModel.return_value = ModelDeleteResponse(
                success=True,
                message="Model deleted",
                workers_deleted=1
            )
            
            mock_channel_instance = AsyncMock()
            mock_channel_instance.__aenter__.return_value = mock_channel_instance
            mock_channel_instance.__aexit__.return_value = None
            mock_channel.return_value = mock_channel_instance
            
            from proto.inference_pb2_grpc import InferenceServiceStub
            with patch.object(InferenceServiceStub, '__new__', return_value=mock_stub):
                # Call DeleteModel
                response = await coordinator_service.DeleteModel(request, context)
        
        # Verify response
        assert response.success is True
        assert response.workers_deleted == 2  # Both workers
        assert "non-sharded" in response.message.lower() or "2/2" in response.message
    
    @pytest.mark.asyncio
    async def test_delete_sharded_model(self, coordinator_service, sharding_coordinator):
        """Test deleting a sharded model"""
        from proto.inference_pb2 import ModelDeleteRequest
        import time
        
        # Create shard metadata
        assignment1 = LayerAssignment(
            node_id="worker-1",
            start_layer=0,
            end_layer=60,
            next_node_id="worker-2",
            next_node_address="localhost:8082"
        )
        
        assignment2 = LayerAssignment(
            node_id="worker-2",
            start_layer=60,
            end_layer=120,
            next_node_id=None,
            next_node_address=None
        )
        
        metadata = ShardedModelMetadata(
            model_name="test-sharded-model",
            total_layers=120,
            shard_assignments={
                "worker-1": assignment1,
                "worker-2": assignment2
            },
            created_at=time.time(),
            status=ShardStatus.READY
        )
        
        # Store metadata
        sharding_coordinator.shard_metadata["test-sharded-model"] = metadata
        
        # Create request
        request = ModelDeleteRequest(model_name="test-sharded-model")
        context = Mock()
        
        # Mock worker responses
        with patch('grpc.aio.insecure_channel') as mock_channel:
            mock_stub = AsyncMock()
            mock_stub.DeleteModel.return_value = Mock(
                success=True,
                message="Sharded model deleted"
            )
            
            mock_channel_instance = AsyncMock()
            mock_channel_instance.close = AsyncMock()
            mock_channel.return_value = mock_channel_instance
            
            from proto.inference_pb2_grpc import InferenceServiceStub
            with patch.object(InferenceServiceStub, '__new__', return_value=mock_stub):
                # Call DeleteModel
                response = await coordinator_service.DeleteModel(request, context)
        
        # Verify response
        assert response.success is True
        assert response.workers_deleted == 2
        assert "sharded" in response.message.lower()
        
        # Verify metadata was cleaned up
        assert sharding_coordinator.get_shard_metadata("test-sharded-model") is None


class TestWorkerDeleteModel:
    """Test worker DeleteModel RPC"""
    
    @pytest.mark.asyncio
    async def test_delete_loaded_model(self, worker_service, model_manager):
        """Test deleting a loaded model"""
        from proto.inference_pb2 import ModelDeleteRequest
        
        # Add a mock model to model_manager
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        model_info = ModelInfo(
            model_path="/path/to/model",
            model_name="test-model",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=0.0,
            memory_usage_gb=4.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        
        model_manager.loaded_models["test-model"] = model_info
        
        # Create request
        request = ModelDeleteRequest(model_name="test-model")
        context = Mock()
        
        # Call DeleteModel
        response = await worker_service.DeleteModel(request, context)
        
        # Verify response
        assert response.success is True
        assert "successfully deleted" in response.message.lower()
        
        # Verify model was unloaded
        assert "test-model" not in model_manager.loaded_models
    
    @pytest.mark.asyncio
    async def test_delete_sharded_model(self, worker_service, model_manager):
        """Test deleting a sharded model"""
        from proto.inference_pb2 import ModelDeleteRequest
        
        # Add a mock sharded model
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        shard_metadata = {
            'start_layer': 0,
            'end_layer': 60,
            'worker_id': 'worker-1',
            'next_worker_id': 'worker-2',
            'next_worker_address': 'localhost:8082',
            'is_first_worker': True,
            'is_last_worker': False
        }
        
        model_info = ModelInfo(
            model_path="/path/to/model",
            model_name="test-sharded-model",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=0.0,
            memory_usage_gb=2.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=shard_metadata
        )
        
        model_manager.loaded_models["test-sharded-model"] = model_info
        
        # Create request
        request = ModelDeleteRequest(model_name="test-sharded-model")
        context = Mock()
        
        # Call DeleteModel
        response = await worker_service.DeleteModel(request, context)
        
        # Verify response
        assert response.success is True
        assert "sharded" in response.message.lower()
        
        # Verify model was unloaded
        assert "test-sharded-model" not in model_manager.loaded_models
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_model(self, worker_service, model_manager):
        """Test deleting a model that doesn't exist"""
        from proto.inference_pb2 import ModelDeleteRequest
        
        # Create request for non-existent model
        request = ModelDeleteRequest(model_name="nonexistent-model")
        context = Mock()
        
        # Call DeleteModel
        response = await worker_service.DeleteModel(request, context)
        
        # Verify response - should succeed with message that model wasn't loaded
        assert response.success is True
        assert "not loaded" in response.message.lower()


class TestModelManagerUnload:
    """Test ModelManager unload_model for sharded models"""
    
    def test_unload_regular_model(self, model_manager):
        """Test unloading a regular model"""
        # Add a mock model
        mock_model = Mock()
        mock_tokenizer = Mock()
        
        model_info = ModelInfo(
            model_path="/path/to/model",
            model_name="test-model",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=0.0,
            memory_usage_gb=4.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        
        model_manager.loaded_models["test-model"] = model_info
        
        # Unload model
        success = model_manager.unload_model("test-model")
        
        # Verify
        assert success is True
        assert "test-model" not in model_manager.loaded_models
    
    def test_unload_sharded_model(self, model_manager):
        """Test unloading a sharded model"""
        # Add a mock sharded model
        mock_model = Mock()
        mock_model.cleanup = Mock()  # Add cleanup method
        mock_tokenizer = Mock()
        
        shard_metadata = {
            'start_layer': 0,
            'end_layer': 60,
            'worker_id': 'worker-1',
            'next_worker_id': 'worker-2',
            'next_worker_address': 'localhost:8082',
            'is_first_worker': True,
            'is_last_worker': False
        }
        
        model_info = ModelInfo(
            model_path="/path/to/model",
            model_name="test-sharded-model",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=0.0,
            memory_usage_gb=2.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=shard_metadata
        )
        
        model_manager.loaded_models["test-sharded-model"] = model_info
        
        # Unload model
        success = model_manager.unload_model("test-sharded-model")
        
        # Verify
        assert success is True
        assert "test-sharded-model" not in model_manager.loaded_models
        # Verify cleanup was called
        mock_model.cleanup.assert_called_once()
    
    def test_unload_nonexistent_model(self, model_manager):
        """Test unloading a model that doesn't exist"""
        success = model_manager.unload_model("nonexistent-model")
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
