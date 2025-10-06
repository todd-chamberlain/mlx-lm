"""
Unit tests for MLX worker components.

Tests model loading and validation, inference processing with mock MLX calls,
streaming functionality and error handling, and metrics collection and health reporting
as specified in task 4.6.
"""

import asyncio
import pytest
import time
import json
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from typing import Dict, Any, List, Iterator

# Create mock modules for MLX dependencies
mock_mlx_core = Mock()
mock_mlx_core.__version__ = "0.15.0"

mock_mlx_lm = Mock()
mock_transformers = Mock()

# Create mock psutil with proper structure
mock_psutil = Mock()
mock_virtual_memory = Mock()
mock_virtual_memory.return_value = Mock(
    total=68719476736,  # 64GB
    available=17179869184,  # 16GB
    used=51539607552,  # 48GB
    percent=75.0
)
mock_psutil.virtual_memory = mock_virtual_memory
mock_psutil.Process = Mock()

# Patch sys.modules before importing worker components
sys.modules['mlx'] = Mock()
sys.modules['mlx.core'] = mock_mlx_core
sys.modules['mlx_lm'] = mock_mlx_lm
sys.modules['transformers'] = mock_transformers
sys.modules['psutil'] = mock_psutil

# Import worker components after mocking
from worker.model_manager import ModelManager, DeviceInfo, ModelInfo
from worker.inference_engine import (
    InferenceEngine, InferenceRequest, InferenceResponse, 
    StreamChunk, InferenceMetrics
)
from worker.grpc_service import InferenceServiceImpl, WorkerGrpcServer
from worker.health_monitor import HealthMonitor, HealthMetrics, SystemHealth, MetalMemoryTracker


class TestModelManager:
    """Test ModelManager functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        with patch('worker.model_manager.subprocess.check_output') as mock_subprocess, \
             patch('worker.model_manager.mx') as mock_mx:
            # Mock system calls for device detection
            mock_subprocess.side_effect = [
                "Mac14,6",  # hw.model
                "68719476736",  # hw.memsize (64GB in bytes)
                "14.0",  # sw_vers
            ]
            # Mock MLX version
            mock_mx.__version__ = "0.15.0"
            
            self.model_manager = ModelManager(model_cache_size=2)
    
    def test_device_info_detection(self):
        """Test device information detection"""
        device_info = self.model_manager.device_info
        
        assert isinstance(device_info, DeviceInfo)
        assert device_info.model is not None
        assert device_info.memory_gb > 0
        assert device_info.gpu_cores > 0
        assert device_info.os_version is not None
        assert device_info.mlx_version is not None
    
    @patch('subprocess.check_output')
    def test_device_info_detection_with_mocks(self, mock_subprocess):
        """Test device info detection with mocked system calls"""
        # Mock system calls
        mock_subprocess.side_effect = [
            "Mac14,6",  # hw.model
            "68719476736",  # hw.memsize (64GB in bytes)
            "14.0",  # sw_vers
        ]
        
        with patch('mlx.core.__version__', '0.15.0'):
            manager = ModelManager()
            device_info = manager.device_info
            
            assert device_info.model == "Mac14,6"
            assert device_info.memory_gb == 64
            assert device_info.os_version == "14.0"
            assert device_info.mlx_version == "0.15.0"
    
    def test_gpu_cores_estimation(self):
        """Test GPU cores estimation for different chip types"""
        test_cases = [
            ("M1", 8),         # M1
            ("M1 Pro", 16),    # M1 Pro
            ("M1 Max", 32),    # M1 Max
            ("M2", 10),        # M2
            ("M2 Pro", 19),    # M2 Pro
            ("M2 Max", 38),    # M2 Max
            ("M2 Ultra", 76),  # M2 Ultra
            ("M3", 10),        # M3
            ("M3 Pro", 18),    # M3 Pro
            ("M3 Max", 40),    # M3 Max
            ("Unknown", 8),    # Default
        ]
        
        for model, expected_cores in test_cases:
            cores = self.model_manager._estimate_gpu_cores(model)
            assert cores == expected_cores
    
    def test_validate_model_path_success(self):
        """Test successful model path validation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            
            # Create required files
            (model_path / "config.json").write_text('{"model_type": "test"}')
            (model_path / "weights.npz").write_text("fake weights")
            
            assert self.model_manager.validate_model_path(str(model_path)) is True
    
    def test_validate_model_path_missing_directory(self):
        """Test model path validation with missing directory"""
        assert self.model_manager.validate_model_path("/nonexistent/path") is False
    
    def test_validate_model_path_missing_config(self):
        """Test model path validation with missing config.json"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            (model_path / "weights.npz").write_text("fake weights")
            
            assert self.model_manager.validate_model_path(str(model_path)) is False
    
    def test_validate_model_path_missing_weights(self):
        """Test model path validation with missing weight files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            (model_path / "config.json").write_text('{"model_type": "test"}')
            
            assert self.model_manager.validate_model_path(str(model_path)) is False
    
    def test_get_model_info_success(self):
        """Test successful model info retrieval"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            config = {
                "model_type": "llama",
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_hidden_layers": 32,
                "max_position_embeddings": 2048,
                "architectures": ["LlamaForCausalLM"]
            }
            (model_path / "config.json").write_text(json.dumps(config))
            
            info = self.model_manager.get_model_info(str(model_path))
            
            assert info is not None
            assert info["model_type"] == "llama"
            assert info["vocab_size"] == 32000
            assert info["hidden_size"] == 4096
            assert info["num_layers"] == 32
            assert info["max_position_embeddings"] == 2048
    
    def test_get_model_info_missing_config(self):
        """Test model info retrieval with missing config"""
        with tempfile.TemporaryDirectory() as temp_dir:
            info = self.model_manager.get_model_info(str(temp_dir))
            assert info is None
    
    def test_estimate_model_memory(self):
        """Test model memory estimation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            config = {
                "vocab_size": 32000,
                "hidden_size": 4096,
                "num_hidden_layers": 32
            }
            (model_path / "config.json").write_text(json.dumps(config))
            
            memory_gb = self.model_manager.estimate_model_memory(str(model_path))
            
            assert memory_gb >= 1.0  # Should be at least 1GB
            assert memory_gb < 100.0  # Should be reasonable
    
    @patch('mlx_lm.load')
    def test_load_model_success(self, mock_mlx_load):
        """Test successful model loading"""
        # Mock MLX load
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_mlx_load.return_value = (mock_model, mock_tokenizer)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            
            # Create required files
            config = {"model_type": "test", "max_position_embeddings": 2048}
            (model_path / "config.json").write_text(json.dumps(config))
            (model_path / "weights.npz").write_text("fake weights")
            
            # Mock memory checks
            with patch.object(self.model_manager, '_get_available_memory', return_value=16.0):
                with patch.object(self.model_manager, '_measure_model_memory', return_value=2.0):
                    success = self.model_manager.load_model(str(model_path), "test-model")
            
            assert success is True
            assert "test-model" in self.model_manager.loaded_models
            
            # Verify model info
            model_info = self.model_manager.loaded_models["test-model"]
            assert model_info.model_name == "test-model"
            assert model_info.model_path == str(model_path)
            assert model_info.model == mock_model
            assert model_info.tokenizer == mock_tokenizer
    
    @patch('mlx_lm.load')
    def test_load_model_insufficient_memory(self, mock_mlx_load):
        """Test model loading with insufficient memory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            
            # Create required files
            config = {"model_type": "test"}
            (model_path / "config.json").write_text(json.dumps(config))
            (model_path / "weights.npz").write_text("fake weights")
            
            # Mock insufficient memory
            with patch.object(self.model_manager, '_get_available_memory', return_value=1.0):
                with patch.object(self.model_manager, 'estimate_model_memory', return_value=8.0):
                    with patch.object(self.model_manager, '_free_memory_for_model', return_value=False):
                        success = self.model_manager.load_model(str(model_path), "test-model")
            
            assert success is False
            assert "test-model" not in self.model_manager.loaded_models
    
    @patch('mlx_lm.load')
    def test_load_model_already_loaded(self, mock_mlx_load):
        """Test loading a model that's already loaded"""
        # Pre-load a model
        mock_model = Mock()
        mock_tokenizer = Mock()
        model_info = ModelInfo(
            model_path="/fake/path",
            model_name="test-model",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=time.time(),
            memory_usage_gb=2.0
        )
        self.model_manager.loaded_models["test-model"] = model_info
        
        # Try to load again
        success = self.model_manager.load_model("/fake/path", "test-model")
        
        assert success is True
        assert mock_mlx_load.call_count == 0  # Should not call MLX load
    
    def test_get_model_success(self):
        """Test successful model retrieval"""
        # Pre-load a model
        mock_model = Mock()
        mock_tokenizer = Mock()
        model_info = ModelInfo(
            model_path="/fake/path",
            model_name="test-model",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=time.time(),
            memory_usage_gb=2.0
        )
        self.model_manager.loaded_models["test-model"] = model_info
        
        result = self.model_manager.get_model("test-model")
        
        assert result is not None
        model, tokenizer = result
        assert model == mock_model
        assert tokenizer == mock_tokenizer
    
    def test_get_model_not_found(self):
        """Test model retrieval for non-existent model"""
        result = self.model_manager.get_model("nonexistent-model")
        assert result is None
    
    def test_unload_model_success(self):
        """Test successful model unloading"""
        # Pre-load a model
        mock_model = Mock()
        mock_tokenizer = Mock()
        model_info = ModelInfo(
            model_path="/fake/path",
            model_name="test-model",
            model=mock_model,
            tokenizer=mock_tokenizer,
            loaded_at=time.time(),
            memory_usage_gb=2.0
        )
        self.model_manager.loaded_models["test-model"] = model_info
        
        success = self.model_manager.unload_model("test-model")
        
        assert success is True
        assert "test-model" not in self.model_manager.loaded_models
    
    def test_unload_model_not_found(self):
        """Test unloading non-existent model"""
        success = self.model_manager.unload_model("nonexistent-model")
        assert success is False
    
    def test_list_loaded_models(self):
        """Test listing loaded models"""
        # Pre-load models
        for i in range(2):
            model_info = ModelInfo(
                model_path=f"/fake/path/{i}",
                model_name=f"test-model-{i}",
                model=Mock(),
                tokenizer=Mock(),
                loaded_at=time.time(),
                memory_usage_gb=2.0
            )
            self.model_manager.loaded_models[f"test-model-{i}"] = model_info
        
        models = self.model_manager.list_loaded_models()
        
        assert len(models) == 2
        assert "test-model-0" in models
        assert "test-model-1" in models
        
        for model_name, model_info in models.items():
            assert "model_name" in model_info
            assert "model_path" in model_info
            assert "loaded_at" in model_info
    
    @patch('psutil.virtual_memory')
    def test_get_memory_usage(self, mock_memory):
        """Test memory usage retrieval"""
        # Mock memory info
        mock_memory.return_value = Mock(
            total=68719476736,  # 64GB
            available=17179869184,  # 16GB
            used=51539607552,  # 48GB
            percent=75.0
        )
        
        usage = self.model_manager.get_memory_usage()
        
        assert usage["total_gb"] == 64.0
        assert usage["available_gb"] == 16.0
        assert usage["used_gb"] == 48.0
        assert usage["percent"] == 75.0


class TestInferenceEngine:
    """Test InferenceEngine functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_model_manager = Mock()
        self.worker_id = "test-worker-1"
        self.inference_engine = InferenceEngine(self.mock_model_manager, self.worker_id)
    
    def test_initialization(self):
        """Test inference engine initialization"""
        assert self.inference_engine.model_manager == self.mock_model_manager
        assert self.inference_engine.worker_id == self.worker_id
        assert isinstance(self.inference_engine.metrics, InferenceMetrics)
        assert len(self.inference_engine.active_requests) == 0
    
    def test_validate_request_success(self):
        """Test successful request validation"""
        request = InferenceRequest(
            request_id="test-123",
            prompt="Hello, world!",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            model="test-model"
        )
        
        # Mock model availability
        self.mock_model_manager.get_model.return_value = (Mock(), Mock())
        
        error = self.inference_engine._validate_request(request)
        assert error is None
    
    def test_validate_request_empty_prompt(self):
        """Test request validation with empty prompt"""
        request = InferenceRequest(
            request_id="test-123",
            prompt="",
            max_tokens=100,
            model="test-model"
        )
        
        error = self.inference_engine._validate_request(request)
        assert error == "Prompt cannot be empty"
    
    def test_validate_request_invalid_max_tokens(self):
        """Test request validation with invalid max_tokens"""
        request = InferenceRequest(
            request_id="test-123",
            prompt="Hello",
            max_tokens=0,
            model="test-model"
        )
        
        error = self.inference_engine._validate_request(request)
        assert "max_tokens must be between 1 and 4096" in error
    
    def test_validate_request_invalid_temperature(self):
        """Test request validation with invalid temperature"""
        request = InferenceRequest(
            request_id="test-123",
            prompt="Hello",
            max_tokens=100,
            temperature=3.0,
            model="test-model"
        )
        
        error = self.inference_engine._validate_request(request)
        assert "temperature must be between 0.0 and 2.0" in error
    
    def test_validate_request_model_not_loaded(self):
        """Test request validation with unloaded model"""
        request = InferenceRequest(
            request_id="test-123",
            prompt="Hello",
            max_tokens=100,
            model="nonexistent-model"
        )
        
        # Mock model not available
        self.mock_model_manager.get_model.return_value = None
        
        error = self.inference_engine._validate_request(request)
        assert "Model nonexistent-model is not loaded" in error
    
    def test_count_tokens(self):
        """Test token counting"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]
        
        count = self.inference_engine._count_tokens(mock_tokenizer, "Hello world")
        assert count == 5
    
    def test_count_tokens_fallback(self):
        """Test token counting fallback when tokenizer fails"""
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = Exception("Tokenizer error")
        
        count = self.inference_engine._count_tokens(mock_tokenizer, "Hello world")
        # Should fall back to character count / 4
        assert count == len("Hello world") // 4
    
    def test_prepare_generation_params(self):
        """Test generation parameter preparation"""
        request = InferenceRequest(
            request_id="test-123",
            prompt="Hello",
            max_tokens=100,
            temperature=0.8,
            top_p=0.95,
            seed=42,
            model="test-model"
        )
        
        params = self.inference_engine._prepare_generation_params(request)
        
        assert params["max_tokens"] == 100
        assert params["temp"] == 0.8
        assert params["top_p"] == 0.95
        assert params["seed"] == 42
    
    @pytest.mark.asyncio
    @patch('mlx_lm.generate')
    async def test_process_inference_success(self, mock_generate):
        """Test successful inference processing"""
        # Mock MLX generate
        mock_generate.return_value = "Generated response text"
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [
            [1, 2, 3],  # prompt tokens
            [4, 5, 6, 7, 8]  # response tokens
        ]
        self.mock_model_manager.get_model.return_value = (mock_model, mock_tokenizer)
        
        request = InferenceRequest(
            request_id="test-123",
            prompt="Hello",
            max_tokens=100,
            model="test-model"
        )
        
        response = await self.inference_engine.process_inference(request)
        
        assert isinstance(response, InferenceResponse)
        assert response.request_id == "test-123"
        assert response.text == "Generated response text"
        assert response.prompt_tokens == 3
        assert response.completion_tokens == 5
        assert response.worker_id == self.worker_id
        assert response.latency_ms > 0
        assert response.tokens_per_second > 0
    
    @pytest.mark.asyncio
    async def test_process_inference_validation_error(self):
        """Test inference processing with validation error"""
        request = InferenceRequest(
            request_id="test-123",
            prompt="",  # Empty prompt
            max_tokens=100,
            model="test-model"
        )
        
        with pytest.raises(ValueError, match="Prompt cannot be empty"):
            await self.inference_engine.process_inference(request)
    
    @pytest.mark.asyncio
    async def test_process_inference_model_not_available(self):
        """Test inference processing with unavailable model"""
        self.mock_model_manager.get_model.return_value = None
        
        request = InferenceRequest(
            request_id="test-123",
            prompt="Hello",
            max_tokens=100,
            model="nonexistent-model"
        )
        
        with pytest.raises(RuntimeError, match="Model nonexistent-model not available"):
            await self.inference_engine.process_inference(request)
    
    @pytest.mark.asyncio
    @patch('mlx_lm.stream_generate')
    async def test_stream_inference_success(self, mock_stream_generate):
        """Test successful streaming inference"""
        # Mock streaming generator
        mock_stream_generate.return_value = iter(["Hello", " world", "!"])
        
        # Mock model and tokenizer
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_tokenizer.encode.side_effect = [
            [1, 2, 3],  # prompt tokens
            [4], [5], [6]  # individual token encodings
        ]
        self.mock_model_manager.get_model.return_value = (mock_model, mock_tokenizer)
        
        request = InferenceRequest(
            request_id="test-123",
            prompt="Hello",
            max_tokens=100,
            model="test-model"
        )
        
        chunks = []
        async for chunk in self.inference_engine.stream_inference(request):
            chunks.append(chunk)
        
        # Should have 3 content chunks + 1 final chunk
        assert len(chunks) == 4
        
        # Check content chunks
        assert chunks[0].text == "Hello"
        assert chunks[0].is_final is False
        assert chunks[1].text == " world"
        assert chunks[1].is_final is False
        assert chunks[2].text == "!"
        assert chunks[2].is_final is False
        
        # Check final chunk
        assert chunks[3].text == ""
        assert chunks[3].is_final is True
        
        # All chunks should have correct metadata
        for chunk in chunks:
            assert chunk.request_id == "test-123"
            assert chunk.worker_id == self.worker_id
    
    def test_get_metrics(self):
        """Test metrics retrieval"""
        # Add some fake active requests
        self.inference_engine.active_requests["req1"] = time.time() - 5
        self.inference_engine.active_requests["req2"] = time.time() - 10
        
        metrics = self.inference_engine.get_metrics()
        
        assert "total_requests" in metrics
        assert "failed_requests" in metrics
        assert "active_requests" in metrics
        assert "avg_latency_ms" in metrics
        assert "tokens_per_second" in metrics
        assert "longest_active_request_seconds" in metrics
        
        assert metrics["active_requests"] == 2
        assert metrics["longest_active_request_seconds"] >= 10
    
    def test_get_active_request_count(self):
        """Test active request count"""
        assert self.inference_engine.get_active_request_count() == 0
        
        # Add active requests
        self.inference_engine.active_requests["req1"] = time.time()
        self.inference_engine.active_requests["req2"] = time.time()
        
        assert self.inference_engine.get_active_request_count() == 2
    
    def test_cancel_request(self):
        """Test request cancellation"""
        # Add active request
        self.inference_engine.active_requests["req1"] = time.time()
        
        # Cancel existing request
        result = self.inference_engine.cancel_request("req1")
        assert result is True
        
        # Cancel non-existent request
        result = self.inference_engine.cancel_request("nonexistent")
        assert result is False
    
    def test_cleanup_stale_requests(self):
        """Test cleanup of stale requests"""
        current_time = time.time()
        
        # Add requests with different ages
        self.inference_engine.active_requests["fresh"] = current_time - 10  # 10 seconds old
        self.inference_engine.active_requests["stale1"] = current_time - 400  # 400 seconds old
        self.inference_engine.active_requests["stale2"] = current_time - 500  # 500 seconds old
        
        # Cleanup requests older than 300 seconds
        cleaned_count = self.inference_engine.cleanup_stale_requests(max_age_seconds=300)
        
        assert cleaned_count == 2
        assert "fresh" in self.inference_engine.active_requests
        assert "stale1" not in self.inference_engine.active_requests
        assert "stale2" not in self.inference_engine.active_requests


class TestInferenceMetrics:
    """Test InferenceMetrics functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.metrics = InferenceMetrics()
    
    def test_initial_state(self):
        """Test initial metrics state"""
        assert self.metrics.total_requests == 0
        assert self.metrics.failed_requests == 0
        assert self.metrics.active_requests == 0
        assert self.metrics.total_tokens == 0
        assert self.metrics.total_latency == 0.0
    
    def test_request_tracking(self):
        """Test request start/end tracking"""
        # Start requests
        self.metrics.start_request()
        self.metrics.start_request()
        
        assert self.metrics.total_requests == 2
        assert self.metrics.active_requests == 2
        
        # End successful request
        self.metrics.end_request(success=True, tokens=50, latency=100.0)
        
        assert self.metrics.active_requests == 1
        assert self.metrics.failed_requests == 0
        assert self.metrics.total_tokens == 50
        assert self.metrics.total_latency == 100.0
        
        # End failed request
        self.metrics.end_request(success=False, tokens=0, latency=200.0)
        
        assert self.metrics.active_requests == 0
        assert self.metrics.failed_requests == 1
        assert self.metrics.total_tokens == 50  # No tokens added for failed request
        assert self.metrics.total_latency == 100.0  # No latency added for failed request
    
    def test_get_stats(self):
        """Test statistics calculation"""
        # Add some requests
        self.metrics.start_request()
        self.metrics.start_request()
        self.metrics.start_request()
        
        self.metrics.end_request(success=True, tokens=100, latency=150.0)
        self.metrics.end_request(success=True, tokens=200, latency=250.0)
        self.metrics.end_request(success=False, tokens=0, latency=0.0)
        
        stats = self.metrics.get_stats()
        
        assert stats["total_requests"] == 3
        assert stats["failed_requests"] == 1
        assert stats["active_requests"] == 0
        assert stats["avg_latency_ms"] == 200.0  # (150 + 250) / 2
        assert stats["tokens_per_second"] == 750.0  # 300 tokens / 0.4 seconds
        assert stats["success_rate"] == 2/3  # 2 successful out of 3 total


class TestHealthMonitor:
    """Test HealthMonitor functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.mock_model_manager = Mock()
        self.mock_inference_engine = Mock()
        self.health_monitor = HealthMonitor(
            self.mock_model_manager,
            self.mock_inference_engine,
            history_size=10,
            check_interval=1.0
        )
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_collect_system_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test system metrics collection"""
        # Mock system calls
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(
            percent=60.0,
            used=32 * 1024**3,  # 32GB
            available=16 * 1024**3  # 16GB
        )
        mock_disk.return_value = Mock(
            used=500 * 1024**3,  # 500GB
            total=1000 * 1024**3  # 1TB
        )
        
        metrics = self.health_monitor._collect_system_metrics()
        
        assert metrics["cpu_percent"] == 45.5
        assert metrics["memory_percent"] == 60.0
        assert metrics["memory_used_gb"] == 32.0
        assert metrics["memory_available_gb"] == 16.0
        assert metrics["disk_usage_percent"] == 50.0
    
    def test_collect_inference_metrics(self):
        """Test inference metrics collection"""
        # Mock inference engine metrics
        self.mock_inference_engine.get_metrics.return_value = {
            'active_requests': 5,
            'avg_latency_ms': 150.0,
            'tokens_per_second': 25.0,
            'total_requests': 100,
            'failed_requests': 5
        }
        
        metrics = self.health_monitor._collect_inference_metrics()
        
        assert metrics["active_requests"] == 5
        assert metrics["avg_latency_ms"] == 150.0
        assert metrics["tokens_per_second"] == 25.0
        assert metrics["total_requests"] == 100
        assert metrics["failed_requests"] == 5
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_collect_metrics(self, mock_disk, mock_memory, mock_cpu):
        """Test comprehensive metrics collection"""
        # Mock system calls
        mock_cpu.return_value = 45.5
        mock_memory.return_value = Mock(
            percent=60.0,
            used=32 * 1024**3,
            available=16 * 1024**3
        )
        mock_disk.return_value = Mock(
            used=500 * 1024**3,
            total=1000 * 1024**3
        )
        
        # Mock inference metrics
        self.mock_inference_engine.get_metrics.return_value = {
            'active_requests': 3,
            'avg_latency_ms': 120.0,
            'tokens_per_second': 30.0,
            'total_requests': 50,
            'failed_requests': 2
        }
        
        metrics = self.health_monitor.collect_metrics()
        
        assert isinstance(metrics, HealthMetrics)
        assert metrics.cpu_percent == 45.5
        assert metrics.memory_percent == 60.0
        assert metrics.active_requests == 3
        assert metrics.avg_latency_ms == 120.0
        assert metrics.uptime_seconds > 0
        
        # Check that metrics are added to history
        assert len(self.health_monitor.metrics_history) == 1
    
    def test_calculate_health_score(self):
        """Test health score calculation"""
        # Create metrics with various conditions
        good_metrics = HealthMetrics(
            timestamp=time.time(),
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=32.0,
            memory_available_gb=16.0,
            disk_usage_percent=70.0,
            active_requests=2,
            avg_latency_ms=100.0,
            tokens_per_second=30.0,
            total_requests=100,
            failed_requests=2,
            uptime_seconds=3600.0
        )
        
        score = self.health_monitor._calculate_health_score(good_metrics)
        assert score == 1.0  # Should be perfect score
        
        # Test with high CPU
        bad_cpu_metrics = good_metrics
        bad_cpu_metrics.cpu_percent = 98.0
        
        score = self.health_monitor._calculate_health_score(bad_cpu_metrics)
        assert score < 1.0  # Should be penalized
    
    def test_identify_issues(self):
        """Test issue identification"""
        # Create metrics with issues
        problematic_metrics = HealthMetrics(
            timestamp=time.time(),
            cpu_percent=98.0,  # Critical CPU
            memory_percent=92.0,  # Critical memory
            memory_used_gb=45.0,
            memory_available_gb=3.0,
            disk_usage_percent=96.0,  # Critical disk
            active_requests=25,  # High load
            avg_latency_ms=8000.0,  # Critical latency
            tokens_per_second=5.0,
            total_requests=100,
            failed_requests=20,  # High error rate
            uptime_seconds=3600.0
        )
        
        issues = self.health_monitor._identify_issues(problematic_metrics)
        
        assert len(issues) > 0
        issue_text = " ".join(issues)
        assert "Critical CPU usage" in issue_text
        assert "Critical memory usage" in issue_text
        assert "Critical disk usage" in issue_text
        assert "Critical latency" in issue_text
        assert "Critical error rate" in issue_text
        assert "High request load" in issue_text
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    @patch('psutil.disk_usage')
    def test_get_health_status(self, mock_disk, mock_memory, mock_cpu):
        """Test health status determination"""
        # Mock healthy system
        mock_cpu.return_value = 30.0
        mock_memory.return_value = Mock(
            percent=50.0,
            used=24 * 1024**3,
            available=24 * 1024**3
        )
        mock_disk.return_value = Mock(
            used=400 * 1024**3,
            total=1000 * 1024**3
        )
        
        self.mock_inference_engine.get_metrics.return_value = {
            'active_requests': 2,
            'avg_latency_ms': 100.0,
            'tokens_per_second': 30.0,
            'total_requests': 100,
            'failed_requests': 1
        }
        
        health = self.health_monitor.get_health_status()
        
        assert isinstance(health, SystemHealth)
        assert health.status == "healthy"
        assert health.score >= 0.8
        assert len(health.issues) == 0
        assert "normally" in health.message.lower()
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self):
        """Test starting and stopping monitoring"""
        # Start monitoring
        await self.health_monitor.start_monitoring()
        assert self.health_monitor.monitor_task is not None
        assert not self.health_monitor.monitor_task.done()
        
        # Stop monitoring
        await self.health_monitor.stop_monitoring()
        assert self.health_monitor.monitor_task is None or self.health_monitor.monitor_task.done()


class TestMetalMemoryTracker:
    """Test MetalMemoryTracker functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tracker = MetalMemoryTracker()
    
    def test_initial_state(self):
        """Test initial tracker state"""
        assert self.tracker.peak_usage == 0.0
        assert self.tracker.current_usage == 0.0
        assert self.tracker.allocation_count == 0
    
    @patch('psutil.Process')
    def test_update_usage(self, mock_process):
        """Test memory usage update"""
        # Mock process memory info
        mock_process.return_value.memory_info.return_value.rss = 4 * 1024**3  # 4GB
        
        self.tracker.update_usage()
        
        assert self.tracker.current_usage > 0
        assert self.tracker.peak_usage >= self.tracker.current_usage
        assert self.tracker.allocation_count == 1
    
    def test_get_stats(self):
        """Test statistics retrieval"""
        # Manually set some values
        self.tracker.current_usage = 2.5
        self.tracker.peak_usage = 3.0
        self.tracker.allocation_count = 10
        
        stats = self.tracker.get_stats()
        
        assert stats["current_gb"] == 2.5
        assert stats["peak_gb"] == 3.0
        assert stats["allocation_count"] == 10
    
    def test_reset_peak(self):
        """Test peak usage reset"""
        self.tracker.current_usage = 2.0
        self.tracker.peak_usage = 5.0
        
        self.tracker.reset_peak()
        
        assert self.tracker.peak_usage == 2.0  # Should match current


if __name__ == "__main__":
    pytest.main([__file__, "-v"])