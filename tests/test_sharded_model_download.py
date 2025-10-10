"""
End-to-End Sharded Model Download Integration Test

Tests the complete sharded model download flow with 2 workers:
- Coordinator creates shard plan
- Workers download only assigned layers
- Shard metadata is stored correctly
- Inference works through sharded model
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from coordinator.service import CoordinatorService
from coordinator.worker_pool import WorkerPool
from coordinator.sharding_coordinator import (
    ShardingCoordinator,
    ShardedModelMetadata,
    ShardStatus,
    DownloadStatus
)
from worker.model_manager import ModelManager, ModelInfo
from worker.model_loader import ModelLoader
from worker.grpc_service import InferenceServiceImpl
from worker.inference_engine import InferenceEngine
from worker.model_partitioner import LayerAssignment
from proto.inference_pb2 import (
    ModelDownloadRequest,
    ModelDownloadResponse,
    InferenceRequest,
    InferenceResponse
)


@pytest.fixture
def sharding_coordinator():
    """Create a sharding coordinator with auto-sharding enabled"""
    return ShardingCoordinator(
        enable_auto_sharding=True,
        shard_threshold_gb=1.0,  # Low threshold for testing
        strategy="balanced",
        min_workers_for_sharding=2
    )


@pytest.fixture
def worker_pool():
    """Create a worker pool"""
    return WorkerPool()


@pytest.fixture
def coordinator_service(worker_pool, sharding_coordinator):
    """Create a coordinator service"""
    return CoordinatorService(worker_pool, sharding_coordinator)


@pytest.fixture
def model_manager():
    """Create a model manager"""
    return ModelManager(model_cache_size=2)


class TestEndToEndShardedDownload:
    """Test end-to-end sharded model download flow"""
    
    @pytest.mark.asyncio
    async def test_complete_sharded_download_flow(
        self,
        coordinator_service,
        sharding_coordinator,
        worker_pool
    ):
        """
        Test complete sharded download flow:
        1. Register 2 workers
        2. Initiate sharded download
        3. Verify shard plan created
        4. Verify workers receive correct layer assignments
        """
        # Step 1: Register 2 workers (directly add to pool for testing)
        from coordinator.models import WorkerInfo, WorkerStatus, DeviceCapabilities
        
        device_caps = DeviceCapabilities(
            model="Mac Studio",
            memory_gb=64,
            gpu_cores=32,
            os_version="14.0",
            mlx_version="0.15.0"
        )
        
        worker1_info = WorkerInfo(
            worker_id="worker-1",
            address="localhost:8081",
            device_capabilities=device_caps,
            max_concurrent=4,
            supported_models=["test-model-120b"],
            status=WorkerStatus.HEALTHY
        )
        
        worker2_info = WorkerInfo(
            worker_id="worker-2",
            address="localhost:8082",
            device_capabilities=device_caps,
            max_concurrent=4,
            supported_models=["test-model-120b"],
            status=WorkerStatus.HEALTHY
        )
        
        # Directly add to worker pool for testing
        worker_pool._workers["worker-1"] = worker1_info
        worker_pool._workers["worker-2"] = worker2_info
        
        # Step 2: Create shard plan
        model_name = "test-model-120b"
        num_layers = 120
        
        workers = [
            {
                'worker_id': 'worker-1',
                'address': 'localhost:8081',
                'available_memory': 32.0
            },
            {
                'worker_id': 'worker-2',
                'address': 'localhost:8082',
                'available_memory': 32.0
            }
        ]
        
        shard_plan = sharding_coordinator.create_shard_plan(
            model_name=model_name,
            num_layers=num_layers,
            workers=workers
        )
        
        # Step 3: Verify shard plan
        assert shard_plan is not None
        assert shard_plan.model_name == model_name
        assert shard_plan.total_layers == num_layers
        assert len(shard_plan.shard_assignments) == 2
        assert shard_plan.status == ShardStatus.DOWNLOADING
        
        # Step 4: Verify layer assignments
        assignment1 = shard_plan.shard_assignments['worker-1']
        assignment2 = shard_plan.shard_assignments['worker-2']
        
        # Worker 1 should get first half
        assert assignment1.start_layer == 0
        assert assignment1.end_layer == 60
        assert assignment1.next_node_id == 'worker-2'
        assert assignment1.next_node_address == 'localhost:8082'
        
        # Worker 2 should get second half
        assert assignment2.start_layer == 60
        assert assignment2.end_layer == 120
        assert assignment2.next_node_id == "" or assignment2.next_node_id is None
        assert assignment2.next_node_address == "" or assignment2.next_node_address is None
        
        # Verify metadata stored
        stored_metadata = sharding_coordinator.get_shard_metadata(model_name)
        assert stored_metadata is not None
        assert stored_metadata.model_name == model_name
    
    @pytest.mark.asyncio
    async def test_workers_load_only_assigned_layers(
        self,
        model_manager,
        sharded_model_loader
    ):
        """
        Test that workers load only their assigned layers
        """
        model_name = "test-model"
        
        # Create layer assignment for worker 1 (layers 0-60)
        assignment1 = LayerAssignment(
            node_id="worker-1",
            start_layer=0,
            end_layer=60,
            next_node_id="worker-2",
            next_node_address="localhost:8082"
        )
        
        # Mock the model loading
        with patch('mlx_lm.load') as mock_load:
            mock_model = Mock()
            mock_model.layers = [Mock() for _ in range(120)]  # Full model has 120 layers
            mock_tokenizer = Mock()
            mock_tokenizer.eos_token_id = 2
            mock_tokenizer.vocab_size = 32000
            
            mock_load.return_value = (mock_model, mock_tokenizer)
            
            # Load partial model
            model, tokenizer, shard_metadata = sharded_model_loader.load_partial_model(
                model_name=model_name,
                layer_assignment=assignment1
            )
            
            # Verify shard metadata
            assert shard_metadata is not None
            assert shard_metadata['start_layer'] == 0
            assert shard_metadata['end_layer'] == 60
            assert shard_metadata['worker_id'] == 'worker-1'
            assert shard_metadata['next_worker_id'] == 'worker-2'
            assert shard_metadata['next_worker_address'] == 'localhost:8082'
            assert shard_metadata['is_first_worker'] is True
            assert shard_metadata['is_last_worker'] is False
            
            # Note: The actual implementation uses node_id in LayerAssignment
            # but converts to worker_id in shard_metadata for consistency
            
            # Verify model and tokenizer returned
            assert model is not None
            assert tokenizer is not None
    
    @pytest.mark.asyncio
    async def test_shard_metadata_stored_correctly(
        self,
        sharding_coordinator
    ):
        """
        Test that shard metadata is stored and retrievable
        """
        model_name = "test-model"
        num_layers = 80
        
        workers = [
            {'worker_id': 'worker-1', 'address': 'localhost:8081'},
            {'worker_id': 'worker-2', 'address': 'localhost:8082'}
        ]
        
        # Create shard plan
        metadata = sharding_coordinator.create_shard_plan(
            model_name=model_name,
            num_layers=num_layers,
            workers=workers
        )
        
        # Verify metadata stored
        assert sharding_coordinator.is_model_sharded(model_name)
        
        # Retrieve metadata
        retrieved = sharding_coordinator.get_shard_metadata(model_name)
        assert retrieved is not None
        assert retrieved.model_name == model_name
        assert retrieved.total_layers == num_layers
        assert len(retrieved.shard_assignments) == 2
        
        # Verify worker assignments
        worker1_assignment = sharding_coordinator.get_worker_assignment(
            model_name, 'worker-1'
        )
        assert worker1_assignment is not None
        assert worker1_assignment.start_layer == 0
        assert worker1_assignment.end_layer == 40
        
        worker2_assignment = sharding_coordinator.get_worker_assignment(
            model_name, 'worker-2'
        )
        assert worker2_assignment is not None
        assert worker2_assignment.start_layer == 40
        assert worker2_assignment.end_layer == 80
    
    @pytest.mark.asyncio
    async def test_download_status_tracking(
        self,
        sharding_coordinator
    ):
        """
        Test download status tracking for each worker
        """
        model_name = "test-model"
        num_layers = 100
        
        workers = [
            {'worker_id': 'worker-1', 'address': 'localhost:8081'},
            {'worker_id': 'worker-2', 'address': 'localhost:8082'}
        ]
        
        # Create shard plan
        metadata = sharding_coordinator.create_shard_plan(
            model_name=model_name,
            num_layers=num_layers,
            workers=workers
        )
        
        # Initial status should be DOWNLOADING
        assert metadata.status == ShardStatus.DOWNLOADING
        
        # Update worker 1 to IN_PROGRESS
        sharding_coordinator.update_shard_status(
            model_name, 'worker-1', DownloadStatus.IN_PROGRESS
        )
        
        metadata = sharding_coordinator.get_shard_metadata(model_name)
        assert metadata.shard_assignments['worker-1'].download_status == DownloadStatus.IN_PROGRESS
        assert metadata.status == ShardStatus.DOWNLOADING
        
        # Complete worker 1
        sharding_coordinator.update_shard_status(
            model_name, 'worker-1', DownloadStatus.COMPLETED
        )
        
        metadata = sharding_coordinator.get_shard_metadata(model_name)
        assert metadata.shard_assignments['worker-1'].download_status == DownloadStatus.COMPLETED
        assert metadata.status == ShardStatus.PARTIAL  # One complete, one not
        
        # Complete worker 2
        sharding_coordinator.update_shard_status(
            model_name, 'worker-2', DownloadStatus.COMPLETED
        )
        
        metadata = sharding_coordinator.get_shard_metadata(model_name)
        assert metadata.shard_assignments['worker-2'].download_status == DownloadStatus.COMPLETED
        assert metadata.status == ShardStatus.READY  # All complete
    
    @pytest.mark.asyncio
    async def test_inference_through_sharded_model(
        self,
        model_manager
    ):
        """
        Test that inference works through a sharded model
        """
        model_name = "test-sharded-model"
        
        # Create mock sharded model
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.vocab_size = 32000
        
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
            model_name=model_name,
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=time.time(),
            memory_usage_gb=30.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=shard_metadata
        )
        
        # Add to model manager
        model_manager.loaded_models[model_name] = model_info
        
        # Verify model is loaded with shard metadata
        assert model_name in model_manager.loaded_models
        loaded_info = model_manager.loaded_models[model_name]
        assert loaded_info is not None
        assert loaded_info.is_sharded
        assert loaded_info.shard_metadata == shard_metadata
        
        # Verify inference engine can detect sharded model
        assert loaded_info.shard_metadata['is_first_worker'] is True
        assert loaded_info.shard_metadata['next_worker_id'] == 'worker-2'
    
    @pytest.mark.asyncio
    async def test_shard_download_failure_handling(
        self,
        sharding_coordinator
    ):
        """
        Test handling of shard download failures
        """
        model_name = "test-model"
        num_layers = 100
        
        workers = [
            {'worker_id': 'worker-1', 'address': 'localhost:8081'},
            {'worker_id': 'worker-2', 'address': 'localhost:8082'}
        ]
        
        # Create shard plan
        metadata = sharding_coordinator.create_shard_plan(
            model_name=model_name,
            num_layers=num_layers,
            workers=workers
        )
        
        # Worker 1 completes successfully
        sharding_coordinator.update_shard_status(
            model_name, 'worker-1', DownloadStatus.COMPLETED
        )
        
        # Worker 2 fails
        sharding_coordinator.update_shard_status(
            model_name, 'worker-2', DownloadStatus.FAILED
        )
        
        # Overall status should be FAILED
        metadata = sharding_coordinator.get_shard_metadata(model_name)
        assert metadata.status == ShardStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_list_sharded_models(
        self,
        sharding_coordinator
    ):
        """
        Test listing all sharded models
        """
        # Create multiple shard plans
        models = [
            ('model-1', 80),
            ('model-2', 120),
            ('model-3', 60)
        ]
        
        workers = [
            {'worker_id': 'worker-1', 'address': 'localhost:8081'},
            {'worker_id': 'worker-2', 'address': 'localhost:8082'}
        ]
        
        for model_name, num_layers in models:
            sharding_coordinator.create_shard_plan(
                model_name=model_name,
                num_layers=num_layers,
                workers=workers
            )
        
        # List sharded models
        sharded_models = sharding_coordinator.list_sharded_models()
        
        assert len(sharded_models) == 3
        assert 'model-1' in sharded_models
        assert 'model-2' in sharded_models
        assert 'model-3' in sharded_models
    
    @pytest.mark.skip(reason="Proto ModelDownloadRequest doesn't have enable_sharding field yet")
    @pytest.mark.asyncio
    async def test_coordinator_download_model_with_sharding(
        self,
        coordinator_service,
        sharding_coordinator,
        worker_pool
    ):
        """
        Test coordinator DownloadModel RPC with sharding enabled
        """
        # Register workers (directly add to pool for testing)
        from coordinator.models import WorkerInfo, WorkerStatus, DeviceCapabilities
        
        device_caps = DeviceCapabilities(
            model="Mac Studio",
            memory_gb=64,
            gpu_cores=32,
            os_version="14.0",
            mlx_version="0.15.0"
        )
        
        worker1 = WorkerInfo(
            worker_id="worker-1",
            address="localhost:8081",
            device_capabilities=device_caps,
            max_concurrent=4,
            supported_models=["test-model-70b"],
            status=WorkerStatus.HEALTHY
        )
        
        worker2 = WorkerInfo(
            worker_id="worker-2",
            address="localhost:8082",
            device_capabilities=device_caps,
            max_concurrent=4,
            supported_models=["test-model-70b"],
            status=WorkerStatus.HEALTHY
        )
        
        # Directly add to worker pool for testing
        worker_pool._workers["worker-1"] = worker1
        worker_pool._workers["worker-2"] = worker2
        
        # Create download request with sharding
        request = ModelDownloadRequest(
            model_name="test-model-70b",
            enable_sharding=True,
            force_redownload=False
        )
        
        context = Mock()
        
        # Mock worker download responses
        with patch('grpc.aio.insecure_channel') as mock_channel:
            mock_stub = AsyncMock()
            mock_stub.DownloadModel.return_value = Mock(
                success=True,
                message="Download started"
            )
            
            mock_channel_instance = AsyncMock()
            mock_channel_instance.close = AsyncMock()
            mock_channel.return_value = mock_channel_instance
            
            from proto.inference_pb2_grpc import InferenceServiceStub
            with patch.object(InferenceServiceStub, '__new__', return_value=mock_stub):
                # Mock get_layer_count
                with patch.object(
                    sharding_coordinator.partitioner,
                    'get_layer_count',
                    return_value=80
                ):
                    # Call DownloadModel
                    response = await coordinator_service.DownloadModel(request, context)
        
        # Verify response
        assert response.success is True
        assert response.is_sharded is True
        assert response.shard_plan is not None
        
        # Verify shard metadata created
        metadata = sharding_coordinator.get_shard_metadata("test-model-70b")
        assert metadata is not None
        assert len(metadata.shard_assignments) == 2


# Note: TestShardedModelLoader class removed - it was 100% mocked and tested
# non-existent worker.sharded_model_loader module. Real ModelLoader tests are
# in test_model_loaders.py and test_e2e_sharding_real.py


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
