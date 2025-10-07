"""
Tests for Pipeline Initialization

Tests the complete pipeline setup workflow.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

from coordinator.pipeline_manager import PipelineManager, PipelineState
from coordinator.pipeline_coordinator import PipelineCoordinator, NodeInfo
from worker.model_partitioner import PartitionStrategy, LayerAssignment


class TestPipelineSetup:
    """Test pipeline setup workflow"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.coordinator = Mock(spec=PipelineCoordinator)
        self.manager = PipelineManager(self.coordinator)
        
        # Create mock nodes
        self.mock_nodes = {
            "node1": NodeInfo(
                node_id="node1",
                address="localhost:8081",
                status="healthy",
                memory_gb=16.0
            ),
            "node2": NodeInfo(
                node_id="node2",
                address="localhost:8082",
                status="healthy",
                memory_gb=16.0
            )
        }
    
    @pytest.mark.asyncio
    async def test_setup_pipeline_no_nodes(self):
        """Test setup fails when no nodes are registered"""
        self.coordinator.get_registered_nodes.return_value = {}
        
        result = await self.manager.setup_pipeline(
            pipeline_id="test-pipeline",
            model_path="test-model"
        )
        
        assert result.success == False
        assert "No nodes registered" in result.error
    
    @pytest.mark.asyncio
    @patch('coordinator.pipeline_manager.ModelPartitioner')
    async def test_setup_pipeline_success(self, mock_partitioner_class):
        """Test successful pipeline setup"""
        # Setup mocks
        self.coordinator.get_registered_nodes.return_value = self.mock_nodes
        self.coordinator.setup_pipeline = AsyncMock(return_value=True)
        
        # Mock partitioner
        mock_partitioner = Mock()
        mock_assignments = [
            LayerAssignment(
                node_id="node1",
                start_layer=0,
                end_layer=16,
                num_layers=16,
                estimated_memory_gb=8.0
            ),
            LayerAssignment(
                node_id="node2",
                start_layer=16,
                end_layer=32,
                num_layers=16,
                estimated_memory_gb=8.0
            )
        ]
        mock_partitioner.partition.return_value = mock_assignments
        mock_partitioner_class.return_value = mock_partitioner
        
        # Setup pipeline
        result = await self.manager.setup_pipeline(
            pipeline_id="test-pipeline",
            model_path="test-model",
            verify_nodes=False
        )
        
        # Verify result
        assert result.success == True
        assert result.pipeline_id == "test-pipeline"
        assert self.manager.get_pipeline_state("test-pipeline") == PipelineState.READY


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
