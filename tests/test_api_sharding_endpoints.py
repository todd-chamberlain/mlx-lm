"""
Unit tests for API sharding endpoints.

Tests the new sharding-related endpoints added in task 6:
- POST /admin/download_model with enable_sharding parameter
- GET /admin/models/{model_name}/shards
- GET /health with sharded model information
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient

from api.main import app, get_coordinator_client
from api.models import (
    ModelDownloadRequest,
    ModelDownloadResponse,
    ShardStatusResponse,
    HealthResponse
)
from api.coordinator_client import CoordinatorClient


class TestShardingEndpoints:
    """Test sharding-related API endpoints"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
        
        # Mock coordinator client
        self.mock_coordinator = AsyncMock(spec=CoordinatorClient)
        
        # Override dependency
        app.dependency_overrides[get_coordinator_client] = lambda: self.mock_coordinator
    
    def teardown_method(self):
        """Clean up after tests"""
        app.dependency_overrides.clear()
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_download_model_without_sharding(self):
        """Test model download without sharding"""
        # Mock coordinator response
        self.mock_coordinator.download_model.return_value = {
            'success': True,
            'message': 'Download initiated',
            'download_id': 'dl-123',
            'workers_notified': 2,
            'is_sharded': False
        }
        
        request_data = {
            "model_name": "test-model",
            "enable_sharding": False,
            "force_redownload": False
        }
        
        response = self.client.post("/admin/download_model", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["download_id"] == "dl-123"
        assert data["is_sharded"] is False
        assert data["shard_plan"] is None
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_download_model_with_sharding(self):
        """Test model download with sharding enabled"""
        # Mock health check
        mock_health = Mock()
        mock_health.healthy_workers = 2
        mock_health.total_workers = 2
        mock_health.active_requests = 0
        self.mock_coordinator.get_health.return_value = mock_health
        
        # Mock coordinator response with shard plan
        self.mock_coordinator.download_model.return_value = {
            'success': True,
            'message': 'Sharded download initiated',
            'download_id': 'dl-456',
            'workers_notified': 2,
            'is_sharded': True,
            'shard_plan': {
                'model_name': 'test-model',
                'total_layers': 120,
                'assignments': [
                    {
                        'worker_id': 'worker-1',
                        'start_layer': 0,
                        'end_layer': 60,
                        'next_worker_id': 'worker-2'
                    },
                    {
                        'worker_id': 'worker-2',
                        'start_layer': 60,
                        'end_layer': 120,
                        'next_worker_id': None
                    }
                ]
            }
        }
        
        request_data = {
            "model_name": "test-model",
            "enable_sharding": True,
            "force_redownload": False
        }
        
        response = self.client.post("/admin/download_model", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["is_sharded"] is True
        assert data["shard_plan"] is not None
        assert data["shard_plan"]["total_layers"] == 120
        assert len(data["shard_plan"]["assignments"]) == 2
        assert data["shard_plan"]["assignments"][0]["worker_id"] == "worker-1"
        assert data["shard_plan"]["assignments"][0]["start_layer"] == 0
        assert data["shard_plan"]["assignments"][0]["end_layer"] == 60
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_download_model_sharding_insufficient_workers(self):
        """Test model download with sharding but insufficient workers"""
        # Mock health check with only 1 worker
        mock_health = Mock()
        mock_health.healthy_workers = 1
        mock_health.total_workers = 1
        mock_health.active_requests = 0
        self.mock_coordinator.get_health.return_value = mock_health
        
        request_data = {
            "model_name": "test-model",
            "enable_sharding": True,
            "force_redownload": False
        }
        
        response = self.client.post("/admin/download_model", json=request_data)
        
        assert response.status_code == 400
        data = response.json()
        assert "requires at least 2 workers" in data["detail"]
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_get_shard_status_sharded_model(self):
        """Test getting shard status for a sharded model"""
        # Mock coordinator response
        self.mock_coordinator.get_shard_status.return_value = {
            'model_name': 'test-model',
            'is_sharded': True,
            'total_layers': 120,
            'shard_assignments': [
                {
                    'worker_id': 'worker-1',
                    'start_layer': 0,
                    'end_layer': 60,
                    'status': 'READY',
                    'progress_percent': 100.0,
                    'error': None
                },
                {
                    'worker_id': 'worker-2',
                    'start_layer': 60,
                    'end_layer': 120,
                    'status': 'READY',
                    'progress_percent': 100.0,
                    'error': None
                }
            ],
            'overall_status': 'READY',
            'inference_count': 42,
            'avg_latency_ms': 125.5
        }
        
        response = self.client.get("/admin/models/test-model/shards")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test-model"
        assert data["is_sharded"] is True
        assert data["total_layers"] == 120
        assert len(data["shard_assignments"]) == 2
        assert data["overall_status"] == "READY"
        assert data["inference_count"] == 42
        assert data["avg_latency_ms"] == 125.5
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_get_shard_status_non_sharded_model(self):
        """Test getting shard status for a non-sharded model"""
        # Mock coordinator response
        self.mock_coordinator.get_shard_status.return_value = {
            'model_name': 'test-model',
            'is_sharded': False,
            'total_layers': 0,
            'shard_assignments': [],
            'overall_status': 'NOT_SHARDED',
            'inference_count': None,
            'avg_latency_ms': None
        }
        
        response = self.client.get("/admin/models/test-model/shards")
        
        assert response.status_code == 200
        data = response.json()
        assert data["model_name"] == "test-model"
        assert data["is_sharded"] is False
        assert len(data["shard_assignments"]) == 0
    
    @patch.dict('os.environ', {'ENABLE_AUTH': 'false'})
    def test_get_shard_status_downloading(self):
        """Test getting shard status while model is downloading"""
        # Mock coordinator response
        self.mock_coordinator.get_shard_status.return_value = {
            'model_name': 'test-model',
            'is_sharded': True,
            'total_layers': 120,
            'shard_assignments': [
                {
                    'worker_id': 'worker-1',
                    'start_layer': 0,
                    'end_layer': 60,
                    'status': 'DOWNLOADING',
                    'progress_percent': 75.0,
                    'error': None
                },
                {
                    'worker_id': 'worker-2',
                    'start_layer': 60,
                    'end_layer': 120,
                    'status': 'DOWNLOADING',
                    'progress_percent': 50.0,
                    'error': None
                }
            ],
            'overall_status': 'DOWNLOADING',
            'inference_count': None,
            'avg_latency_ms': None
        }
        
        response = self.client.get("/admin/models/test-model/shards")
        
        assert response.status_code == 200
        data = response.json()
        assert data["overall_status"] == "DOWNLOADING"
        assert data["shard_assignments"][0]["progress_percent"] == 75.0
        assert data["shard_assignments"][1]["progress_percent"] == 50.0
    
    def test_health_endpoint_without_sharded_models(self):
        """Test health endpoint without requesting sharded model info"""
        # Mock health response
        mock_health = Mock()
        mock_health.healthy_workers = 2
        mock_health.total_workers = 2
        mock_health.active_requests = 5
        self.mock_coordinator.get_health.return_value = mock_health
        
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["total_workers"] == 2
        assert data["healthy_workers"] == 2
        assert data["sharded_model_count"] is None
        assert data["sharded_models"] is None
    
    def test_health_endpoint_with_sharded_models(self):
        """Test health endpoint with sharded model information"""
        # Mock health response with sharded models
        mock_health = Mock()
        mock_health.healthy_workers = 2
        mock_health.total_workers = 2
        mock_health.active_requests = 5
        
        # Add sharded models info
        mock_sharded_model = Mock()
        mock_sharded_model.model_name = "test-model"
        mock_sharded_model.total_shards = 2
        mock_sharded_model.healthy_shards = 2
        mock_sharded_model.status = "ready"
        
        mock_health.sharded_models = [mock_sharded_model]
        
        self.mock_coordinator.get_health.return_value = mock_health
        
        response = self.client.get("/health?include_sharded_models=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["sharded_model_count"] == 1
        assert data["sharded_models"] is not None
        assert len(data["sharded_models"]) == 1
        assert data["sharded_models"][0]["model_name"] == "test-model"
        assert data["sharded_models"][0]["total_shards"] == 2
        assert data["sharded_models"][0]["healthy_shards"] == 2
    
    def test_health_endpoint_with_degraded_sharded_model(self):
        """Test health endpoint with degraded sharded model"""
        # Mock health response with degraded sharded model
        mock_health = Mock()
        mock_health.healthy_workers = 2
        mock_health.total_workers = 2
        mock_health.active_requests = 5
        
        # Add degraded sharded model
        mock_sharded_model = Mock()
        mock_sharded_model.model_name = "test-model"
        mock_sharded_model.total_shards = 2
        mock_sharded_model.healthy_shards = 1
        mock_sharded_model.status = "degraded"
        
        mock_health.sharded_models = [mock_sharded_model]
        
        self.mock_coordinator.get_health.return_value = mock_health
        
        response = self.client.get("/health?include_sharded_models=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"  # Overall status should be degraded
        assert data["sharded_models"][0]["status"] == "degraded"
        assert data["sharded_models"][0]["healthy_shards"] == 1
    
    def test_health_endpoint_with_offline_sharded_model(self):
        """Test health endpoint with offline sharded model"""
        # Mock health response with offline sharded model
        mock_health = Mock()
        mock_health.healthy_workers = 2
        mock_health.total_workers = 2
        mock_health.active_requests = 5
        
        # Add offline sharded model
        mock_sharded_model = Mock()
        mock_sharded_model.model_name = "test-model"
        mock_sharded_model.total_shards = 2
        mock_sharded_model.healthy_shards = 0
        mock_sharded_model.status = "offline"
        
        mock_health.sharded_models = [mock_sharded_model]
        
        self.mock_coordinator.get_health.return_value = mock_health
        
        response = self.client.get("/health?include_sharded_models=true")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "degraded"  # Overall status should be degraded
        assert data["sharded_models"][0]["status"] == "offline"
        assert data["sharded_models"][0]["healthy_shards"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
