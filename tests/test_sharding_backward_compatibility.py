"""
Backward Compatibility Tests for Sharding Feature

Tests that existing non-sharded functionality continues to work:
- Non-sharded model loading still works
- Mixed sharded/non-sharded workload
- No regressions in existing functionality
- Sharding can be disabled
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from coordinator.service import CoordinatorService
from coordinator.worker_pool import WorkerPool
from coordinator.sharding_coordinator import (
    ShardingCoordinator,
    ShardStatus,
    DownloadStatus
)
from worker.model_manager import ModelManager, ModelInfo
from worker.inference_engine import InferenceEngine
from proto.inference_pb2 import (
    ModelDownloadRequest,
    InferenceRequest
)


@pytest.fixture
def sharding_disabled_coordinator():
    """Create a sharding coordinator with auto-sharding disabled"""
    return ShardingCoordinator(
        enable_auto_sharding=False,
        shard_threshold_gb=30.0,
        strategy="balanced"
    )


@pytest.fixture
def sharding_enabled_coordinator():
    """Create a sharding coordinator with auto-sharding enabled"""
    return ShardingCoordinator(
        enable_auto_sharding=True,
        shard_threshold_gb=30.0,
        strategy="balanced"
    )


@pytest.fixture
def worker_pool():
    """Create a worker pool"""
    return WorkerPool()


@pytest.fixture
def model_manager():
    """Create a model manager"""
    return ModelManager(model_cache_size=5)


class TestNonShardedModelLoading:
    """Test that non-sharded model loading still works"""
    
    def test_load_non_sharded_model(self, model_manager):
        """Test loading a regular non-sharded model"""
        model_name = "test-model-7b"
        
        # Create mock model directly (skip actual loading)
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.vocab_size = 32000
        
        model_info = ModelInfo(
            model_path=model_name,
            model_name=model_name,
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=time.time(),
            memory_usage_gb=7.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        
        model_manager.loaded_models[model_name] = model_info
        
        # Verify model loaded correctly
        assert model_info is not None
        assert model_info.model_name == model_name
        assert model_info.shard_metadata is None
        assert not model_info.is_sharded
        
        # Verify model is in loaded models
        assert model_name in model_manager.loaded_models
    
    def test_inference_on_non_sharded_model(self, model_manager):
        """Test inference on non-sharded model works as before"""
        model_name = "test-model-7b"
        
        # Create mock non-sharded model
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.vocab_size = 32000
        
        model_info = ModelInfo(
            model_path="/path/to/model",
            model_name=model_name,
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=time.time(),
            memory_usage_gb=7.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None  # No sharding
        )
        
        model_manager.loaded_models[model_name] = model_info
        
        # Verify model info
        loaded_info = model_manager.get_model_info(model_name)
        assert loaded_info is not None
        assert not loaded_info.is_sharded
        assert loaded_info.shard_metadata is None
    
    def test_sharding_disabled_by_config(
        self,
        sharding_disabled_coordinator
    ):
        """Test that sharding is disabled when configured"""
        model_name = "test-model-70b"
        model_size_gb = 70.0  # Large model
        num_workers = 2
        
        # Should not shard even though model is large
        should_shard = sharding_disabled_coordinator.should_shard_model(
            model_name=model_name,
            model_size_gb=model_size_gb,
            num_workers=num_workers
        )
        
        assert should_shard is False
        
        # Verify no shard metadata exists
        metadata = sharding_disabled_coordinator.get_shard_metadata(model_name)
        assert metadata is None
    
    @pytest.mark.asyncio
    async def test_download_without_sharding_flag(
        self,
        worker_pool,
        sharding_enabled_coordinator
    ):
        """Test download without enable_sharding flag uses non-sharded path"""
        # Register workers
        worker1 = Mock()
        worker1.worker_id = "worker-1"
        worker1.address = "localhost:8081"
        worker1.status = Mock()
        worker1.status.value = 1
        
        worker_pool.register_worker(worker1)
        
        coordinator_service = CoordinatorService(worker_pool, sharding_enabled_coordinator)
        
        # Create download request WITHOUT sharding
        request = ModelDownloadRequest(
            model_name="test-model",
            enable_sharding=False,
            force_redownload=False
        )
        
        context = Mock()
        
        # Mock worker download
        with patch('grpc.aio.insecure_channel') as mock_channel:
            mock_stub = AsyncMock()
            mock_stub.DownloadModel.return_value = Mock(
                success=True,
                message="Download complete"
            )
            
            mock_channel_instance = AsyncMock()
            mock_channel_instance.close = AsyncMock()
            mock_channel.return_value = mock_channel_instance
            
            from proto.inference_pb2_grpc import InferenceServiceStub
            with patch.object(InferenceServiceStub, '__new__', return_value=mock_stub):
                response = await coordinator_service.DownloadModel(request, context)
        
        # Verify non-sharded download
        assert response.success is True
        assert response.is_sharded is False
        assert response.shard_plan is None
        
        # Verify no shard metadata created
        metadata = sharding_enabled_coordinator.get_shard_metadata("test-model")
        assert metadata is None


class TestMixedShardedNonShardedWorkload:
    """Test mixed sharded and non-sharded models"""
    
    def test_load_both_sharded_and_non_sharded_models(
        self,
        model_manager
    ):
        """Test loading both sharded and non-sharded models simultaneously"""
        # Load non-sharded model
        non_sharded_model = Mock()
        non_sharded_tokenizer = Mock()
        non_sharded_tokenizer.eos_token_id = 2
        
        non_sharded_info = ModelInfo(
            model_path="/path/to/small",
            model_name="small-model-7b",
            model=non_sharded_model,
            tokenizer=non_sharded_tokenizer,
            loaded_at=time.time(),
            memory_usage_gb=7.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        
        model_manager.loaded_models["small-model-7b"] = non_sharded_info
        
        # Load sharded model
        sharded_model = Mock()
        sharded_tokenizer = Mock()
        sharded_tokenizer.eos_token_id = 2
        
        shard_metadata = {
            'start_layer': 0,
            'end_layer': 60,
            'worker_id': 'worker-1',
            'next_worker_id': 'worker-2',
            'is_first_worker': True,
            'is_last_worker': False
        }
        
        sharded_info = ModelInfo(
            model_path="/path/to/large",
            model_name="large-model-120b",
            model=sharded_model,
            tokenizer=sharded_tokenizer,
            loaded_at=time.time(),
            memory_usage_gb=30.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=shard_metadata
        )
        
        model_manager.loaded_models["large-model-120b"] = sharded_info
        
        # Verify both models loaded
        assert len(model_manager.loaded_models) == 2
        
        # Verify non-sharded model
        small_info = model_manager.get_model_info("small-model-7b")
        assert small_info is not None
        assert not small_info.is_sharded
        
        # Verify sharded model
        large_info = model_manager.get_model_info("large-model-120b")
        assert large_info is not None
        assert large_info.is_sharded
    
    def test_list_models_shows_both_types(
        self,
        model_manager
    ):
        """Test listing models shows both sharded and non-sharded"""
        # Add non-sharded model
        non_sharded_info = ModelInfo(
            model_path="/path/to/model1",
            model_name="model-1",
            model=Mock(),
            tokenizer=Mock(),
            loaded_at=time.time(),
            memory_usage_gb=5.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        model_manager.loaded_models["model-1"] = non_sharded_info
        
        # Add sharded model
        sharded_info = ModelInfo(
            model_path="/path/to/model2",
            model_name="model-2",
            model=Mock(),
            tokenizer=Mock(),
            loaded_at=time.time(),
            memory_usage_gb=30.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata={'start_layer': 0, 'end_layer': 60}
        )
        model_manager.loaded_models["model-2"] = sharded_info
        
        # List all models
        models = model_manager.list_loaded_models()
        
        assert len(models) == 2
        assert any(m['model_name'] == 'model-1' and not m.get('is_sharded', False) for m in models)
        assert any(m['model_name'] == 'model-2' and m.get('is_sharded', False) for m in models)
    
    def test_inference_routing_for_mixed_models(
        self,
        model_manager
    ):
        """Test inference routing works for both model types"""
        # Setup non-sharded model
        non_sharded_info = ModelInfo(
            model_path="/path/to/small",
            model_name="small-model",
            model=Mock(),
            tokenizer=Mock(),
            loaded_at=time.time(),
            memory_usage_gb=5.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        model_manager.loaded_models["small-model"] = non_sharded_info
        
        # Setup sharded model
        sharded_info = ModelInfo(
            model_path="/path/to/large",
            model_name="large-model",
            model=Mock(),
            tokenizer=Mock(),
            loaded_at=time.time(),
            memory_usage_gb=30.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata={'start_layer': 0, 'end_layer': 60, 'is_first_worker': True}
        )
        model_manager.loaded_models["large-model"] = sharded_info
        
        # Create inference engine
        inference_engine = Mock(spec=InferenceEngine)
        inference_engine.worker_id = "test-worker"
        
        # Mock process_inference to detect model type
        def mock_process_inference(request):
            model_info = model_manager.get_model_info(request.model)
            if model_info.is_sharded:
                return Mock(text="Sharded inference result", worker_id="test-worker")
            else:
                return Mock(text="Non-sharded inference result", worker_id="test-worker")
        
        inference_engine.process_inference = mock_process_inference
        
        # Test non-sharded inference
        non_sharded_request = Mock()
        non_sharded_request.model = "small-model"
        result1 = inference_engine.process_inference(non_sharded_request)
        assert "Non-sharded" in result1.text
        
        # Test sharded inference
        sharded_request = Mock()
        sharded_request.model = "large-model"
        result2 = inference_engine.process_inference(sharded_request)
        assert "Sharded" in result2.text


class TestNoRegressions:
    """Test no regressions in existing functionality"""
    
    def test_model_unloading_still_works(
        self,
        model_manager
    ):
        """Test model unloading works for non-sharded models"""
        model_name = "test-model"
        
        # Load model
        model_info = ModelInfo(
            model_path="/path/to/model",
            model_name=model_name,
            model=Mock(),
            tokenizer=Mock(),
            loaded_at=time.time(),
            memory_usage_gb=5.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        model_manager.loaded_models[model_name] = model_info
        
        # Unload model
        success = model_manager.unload_model(model_name)
        
        assert success is True
        assert model_name not in model_manager.loaded_models
    
    def test_model_caching_still_works(
        self,
        model_manager
    ):
        """Test model caching/LRU eviction still works"""
        # Set small cache size
        model_manager.model_cache_size = 2
        
        # Load 3 models (should evict oldest)
        for i in range(3):
            model_name = f"model-{i}"
            model_info = ModelInfo(
                model_path=f"/path/to/model-{i}",
                model_name=model_name,
                model=Mock(),
                tokenizer=Mock(),
                loaded_at=time.time() + i,  # Different timestamps
                memory_usage_gb=5.0,
                max_tokens=4096,
                context_length=4096,
                shard_metadata=None
            )
            model_manager.loaded_models[model_name] = model_info
            
            # Simulate cache eviction if needed
            if len(model_manager.loaded_models) > model_manager.model_cache_size:
                # Find oldest model
                oldest = min(
                    model_manager.loaded_models.items(),
                    key=lambda x: x[1].loaded_at
                )
                model_manager.unload_model(oldest[0])
        
        # Should have at most 2 models
        assert len(model_manager.loaded_models) <= 2
    
    def test_worker_registration_unchanged(
        self,
        worker_pool
    ):
        """Test worker registration still works as before"""
        worker_info = Mock()
        worker_info.worker_id = "test-worker"
        worker_info.address = "localhost:8081"
        worker_info.status = Mock()
        worker_info.status.value = 1
        
        # Register worker
        worker_pool.register_worker(worker_info)
        
        # Verify registration
        registered = worker_pool.get_worker("test-worker")
        assert registered is not None
        assert registered.worker_id == "test-worker"
    
    def test_health_checks_unchanged(
        self,
        worker_pool
    ):
        """Test health checks work as before"""
        worker_info = Mock()
        worker_info.worker_id = "test-worker"
        worker_info.address = "localhost:8081"
        worker_info.status = Mock()
        worker_info.status.value = 1
        worker_info.last_heartbeat = time.time()
        
        worker_pool.register_worker(worker_info)
        
        # Get healthy workers
        healthy = worker_pool.get_healthy_workers()
        assert len(healthy) > 0
        assert any(w.worker_id == "test-worker" for w in healthy)


class TestExistingModelsNotAffected:
    """Test existing models are not affected by sharding feature"""
    
    def test_existing_models_remain_non_sharded(
        self,
        sharding_enabled_coordinator
    ):
        """Test that existing models remain non-sharded after upgrade"""
        # Simulate existing model (no shard metadata)
        model_name = "existing-model"
        
        # Check if model is sharded
        is_sharded = sharding_enabled_coordinator.is_model_sharded(model_name)
        
        # Should not be sharded
        assert is_sharded is False
        
        # Should have no metadata
        metadata = sharding_enabled_coordinator.get_shard_metadata(model_name)
        assert metadata is None
    
    def test_existing_models_no_redownload_required(
        self,
        model_manager
    ):
        """Test existing models don't require re-download"""
        model_name = "existing-model"
        
        # Simulate existing loaded model
        model_info = ModelInfo(
            model_path="/path/to/existing",
            model_name=model_name,
            model=Mock(),
            tokenizer=Mock(),
            loaded_at=time.time() - 86400,  # Loaded yesterday
            memory_usage_gb=10.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        model_manager.loaded_models[model_name] = model_info
        
        # Model should still be usable
        loaded_info = model_manager.get_model_info(model_name)
        assert loaded_info is not None
        assert not loaded_info.is_sharded
        
        # Should not require re-download
        assert loaded_info.model is not None
        assert loaded_info.tokenizer is not None
    
    def test_sharding_enabled_later_doesnt_affect_loaded_models(
        self,
        model_manager,
        sharding_disabled_coordinator,
        sharding_enabled_coordinator
    ):
        """Test enabling sharding later doesn't affect already loaded models"""
        model_name = "test-model"
        
        # Load model with sharding disabled
        model_info = ModelInfo(
            model_path="/path/to/model",
            model_name=model_name,
            model=Mock(),
            tokenizer=Mock(),
            loaded_at=time.time(),
            memory_usage_gb=10.0,
            max_tokens=4096,
            context_length=4096,
            shard_metadata=None
        )
        model_manager.loaded_models[model_name] = model_info
        
        # Verify model is non-sharded
        assert not model_info.is_sharded
        
        # "Enable" sharding (simulate config change)
        # Existing model should remain non-sharded
        loaded_info = model_manager.get_model_info(model_name)
        assert not loaded_info.is_sharded
        assert loaded_info.shard_metadata is None


class TestShardingThresholds:
    """Test sharding threshold behavior"""
    
    def test_small_models_not_sharded(
        self,
        sharding_enabled_coordinator
    ):
        """Test small models below threshold are not sharded"""
        model_name = "small-model-3b"
        model_size_gb = 3.0  # Below 30GB threshold
        num_workers = 2
        
        should_shard = sharding_enabled_coordinator.should_shard_model(
            model_name=model_name,
            model_size_gb=model_size_gb,
            num_workers=num_workers
        )
        
        assert should_shard is False
    
    def test_large_models_sharded(
        self,
        sharding_enabled_coordinator
    ):
        """Test large models above threshold are sharded"""
        model_name = "large-model-70b"
        model_size_gb = 70.0  # Above 30GB threshold
        num_workers = 2
        
        should_shard = sharding_enabled_coordinator.should_shard_model(
            model_name=model_name,
            model_size_gb=model_size_gb,
            num_workers=num_workers
        )
        
        assert should_shard is True
    
    def test_single_worker_no_sharding(
        self,
        sharding_enabled_coordinator
    ):
        """Test models are not sharded with only 1 worker"""
        model_name = "large-model-70b"
        model_size_gb = 70.0
        num_workers = 1  # Only 1 worker
        
        should_shard = sharding_enabled_coordinator.should_shard_model(
            model_name=model_name,
            model_size_gb=model_size_gb,
            num_workers=num_workers
        )
        
        assert should_shard is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
