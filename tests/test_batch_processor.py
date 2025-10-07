"""
Tests for Batch Processing Pipeline

Tests batch assembly, padding, disassembly, error handling,
and end-to-end batch processing.
"""

import pytest
import time
import asyncio
from unittest.mock import Mock, AsyncMock, patch
import mlx.core as mx

from coordinator.batch_processor import (
    BatchAssembler,
    BatchDisassembler,
    BatchProcessor,
    PaddedBatch,
    BatchResult
)
from coordinator.batch_manager import Batch, BatchRequest, BatchPriority
from coordinator.models import InferenceResult


class TestBatchAssembler:
    """Test batch assembler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.assembler = BatchAssembler(
            padding_token="<pad>",
            max_padding_ratio=0.5
        )
    
    def test_assembler_initialization(self):
        """Test assembler initialization."""
        assert self.assembler.padding_token == "<pad>"
        assert self.assembler.max_padding_ratio == 0.5
    
    def test_assemble_uniform_batch(self):
        """Test assembling batch with uniform prompt lengths."""
        requests = [
            BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Hello world",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            for i in range(3)
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        padded_batch = self.assembler.assemble(batch)
        
        assert padded_batch is not None
        assert padded_batch.size == 3
        assert len(padded_batch.padded_prompts) == 3
        assert padded_batch.attention_mask.shape == (3, 2)  # 3 requests, 2 tokens each
    
    def test_assemble_variable_length_batch(self):
        """Test assembling batch with variable prompt lengths."""
        requests = [
            BatchRequest(
                request_id="req_1",
                model_name="test-model",
                prompt="Short",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ),
            BatchRequest(
                request_id="req_2",
                model_name="test-model",
                prompt="This is a much longer prompt",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ),
            BatchRequest(
                request_id="req_3",
                model_name="test-model",
                prompt="Medium length prompt",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        padded_batch = self.assembler.assemble(batch)
        
        assert padded_batch is not None
        assert padded_batch.size == 3
        
        # All prompts should be padded to same length
        max_length = max(req.prompt_length for req in requests)
        assert padded_batch.max_length == max_length
        
        # Attention mask should have correct shape
        assert padded_batch.attention_mask.shape == (3, max_length)
    
    def test_assemble_empty_batch(self):
        """Test assembling empty batch."""
        batch = Batch(batch_id="batch_1", requests=[])
        padded_batch = self.assembler.assemble(batch)
        
        assert padded_batch is None
    
    def test_attention_mask_correctness(self):
        """Test attention mask is correctly generated."""
        requests = [
            BatchRequest(
                request_id="req_1",
                model_name="test-model",
                prompt="Short",  # 1 token
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ),
            BatchRequest(
                request_id="req_2",
                model_name="test-model",
                prompt="Much longer prompt",  # 3 tokens
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        padded_batch = self.assembler.assemble(batch)
        
        assert padded_batch is not None
        
        # First request: 1 real token, 2 padding
        mask_0 = padded_batch.attention_mask[0].tolist()
        assert mask_0 == [1.0, 0.0, 0.0]
        
        # Second request: 3 real tokens, 0 padding
        mask_1 = padded_batch.attention_mask[1].tolist()
        assert mask_1 == [1.0, 1.0, 1.0]
    
    def test_validate_batch_success(self):
        """Test successful batch validation."""
        requests = [
            BatchRequest(
                request_id="req_1",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        is_valid, error_msg = self.assembler.validate_batch(batch)
        
        assert is_valid is True
        assert error_msg is None
    
    def test_validate_empty_batch(self):
        """Test validation of empty batch."""
        batch = Batch(batch_id="batch_1", requests=[])
        is_valid, error_msg = self.assembler.validate_batch(batch)
        
        assert is_valid is False
        assert "Empty batch" in error_msg
    
    def test_validate_multiple_models(self):
        """Test validation fails with multiple models."""
        requests = [
            BatchRequest(
                request_id="req_1",
                model_name="model-a",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ),
            BatchRequest(
                request_id="req_2",
                model_name="model-b",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        is_valid, error_msg = self.assembler.validate_batch(batch)
        
        assert is_valid is False
        assert "Multiple models" in error_msg


class TestBatchDisassembler:
    """Test batch disassembler functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.disassembler = BatchDisassembler()
    
    def test_disassembler_initialization(self):
        """Test disassembler initialization."""
        assert self.disassembler is not None
    
    def test_disassemble_successful_batch(self):
        """Test disassembling successful batch."""
        requests = [
            BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            for i in range(3)
        ]
        
        padded_batch = PaddedBatch(
            batch_id="batch_1",
            original_requests=requests,
            padded_prompts=["Test"] * 3,
            attention_mask=mx.ones((3, 1)),
            max_length=1
        )
        
        batch_outputs = [
            "Output 1",
            "Output 2",
            "Output 3"
        ]
        
        batch_result = self.disassembler.disassemble(
            padded_batch,
            batch_outputs,
            processing_time_ms=100.0
        )
        
        assert batch_result.batch_id == "batch_1"
        assert batch_result.success_count == 3
        assert batch_result.failure_count == 0
        assert len(batch_result.request_results) == 3
        
        # Check individual results
        for i in range(3):
            result = batch_result.request_results[f"req_{i}"]
            assert result.success is True
            assert result.generated_text == f"Output {i+1}"
    
    def test_disassemble_with_padding_removal(self):
        """Test padding tokens are removed from outputs."""
        requests = [
            BatchRequest(
                request_id="req_1",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        ]
        
        padded_batch = PaddedBatch(
            batch_id="batch_1",
            original_requests=requests,
            padded_prompts=["Test"],
            attention_mask=mx.ones((1, 1)),
            max_length=1,
            padding_token="<pad>"
        )
        
        batch_outputs = ["Output with <pad> <pad> padding"]
        
        batch_result = self.disassembler.disassemble(
            padded_batch,
            batch_outputs,
            processing_time_ms=100.0
        )
        
        result = batch_result.request_results["req_1"]
        assert "<pad>" not in result.generated_text
        # Padding tokens are replaced, leaving extra spaces
        assert "Output with" in result.generated_text
        assert "padding" in result.generated_text
    
    def test_disassemble_missing_outputs(self):
        """Test handling of missing outputs."""
        requests = [
            BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            for i in range(3)
        ]
        
        padded_batch = PaddedBatch(
            batch_id="batch_1",
            original_requests=requests,
            padded_prompts=["Test"] * 3,
            attention_mask=mx.ones((3, 1)),
            max_length=1
        )
        
        # Only 2 outputs for 3 requests
        batch_outputs = ["Output 1", "Output 2"]
        
        batch_result = self.disassembler.disassemble(
            padded_batch,
            batch_outputs,
            processing_time_ms=100.0
        )
        
        assert batch_result.success_count == 2
        assert batch_result.failure_count == 1
        
        # Third request should have error
        result = batch_result.request_results["req_2"]
        assert result.success is False
        assert "Missing output" in result.error
    
    def test_disassemble_processing_time_distribution(self):
        """Test processing time is distributed across requests."""
        requests = [
            BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            for i in range(4)
        ]
        
        padded_batch = PaddedBatch(
            batch_id="batch_1",
            original_requests=requests,
            padded_prompts=["Test"] * 4,
            attention_mask=mx.ones((4, 1)),
            max_length=1
        )
        
        batch_outputs = ["Output"] * 4
        total_time = 400.0
        
        batch_result = self.disassembler.disassemble(
            padded_batch,
            batch_outputs,
            processing_time_ms=total_time
        )
        
        # Each request should get 1/4 of total time
        for req_id in [f"req_{i}" for i in range(4)]:
            result = batch_result.request_results[req_id]
            assert result.processing_time_ms == total_time / 4


class TestBatchProcessor:
    """Test batch processor functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = BatchProcessor()
    
    def test_processor_initialization(self):
        """Test processor initialization."""
        assert self.processor.assembler is not None
        assert self.processor.disassembler is not None
        assert self.processor.stats["batches_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_process_batch_success(self):
        """Test successful batch processing."""
        requests = [
            BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test prompt",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            for i in range(3)
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        
        # Mock inference function
        async def mock_inference(prompts, max_tokens):
            return [f"Output {i}" for i in range(len(prompts))]
        
        batch_result = await self.processor.process_batch(batch, mock_inference)
        
        assert batch_result.batch_id == "batch_1"
        assert batch_result.success_count == 3
        assert batch_result.failure_count == 0
        assert len(batch_result.request_results) == 3
    
    @pytest.mark.asyncio
    async def test_process_batch_inference_failure(self):
        """Test batch processing with inference failure."""
        requests = [
            BatchRequest(
                request_id="req_1",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        
        # Mock inference function that raises error
        async def mock_inference(prompts, max_tokens):
            raise RuntimeError("Inference failed")
        
        batch_result = await self.processor.process_batch(batch, mock_inference)
        
        assert batch_result.success_count == 0
        assert batch_result.failure_count == 1
        
        result = batch_result.request_results["req_1"]
        assert result.success is False
        assert "Inference failed" in result.error
    
    @pytest.mark.asyncio
    async def test_process_batch_validation_failure(self):
        """Test batch processing with validation failure."""
        # Create batch with multiple models (invalid)
        requests = [
            BatchRequest(
                request_id="req_1",
                model_name="model-a",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            ),
            BatchRequest(
                request_id="req_2",
                model_name="model-b",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        
        async def mock_inference(prompts, max_tokens):
            return ["Output"] * len(prompts)
        
        batch_result = await self.processor.process_batch(batch, mock_inference)
        
        assert batch_result.success_count == 0
        assert batch_result.failure_count == 2
    
    @pytest.mark.asyncio
    async def test_process_batch_updates_stats(self):
        """Test batch processing updates statistics."""
        requests = [
            BatchRequest(
                request_id=f"req_{i}",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
            for i in range(5)
        ]
        
        batch = Batch(batch_id="batch_1", requests=requests)
        
        async def mock_inference(prompts, max_tokens):
            return ["Output"] * len(prompts)
        
        await self.processor.process_batch(batch, mock_inference)
        
        stats = self.processor.get_stats()
        
        assert stats["batches_processed"] == 1
        assert stats["total_requests"] == 5
        assert stats["successful_requests"] == 5
        assert stats["failed_requests"] == 0
    
    @pytest.mark.asyncio
    async def test_process_multiple_batches(self):
        """Test processing multiple batches."""
        async def mock_inference(prompts, max_tokens):
            return ["Output"] * len(prompts)
        
        # Process 3 batches
        for i in range(3):
            requests = [
                BatchRequest(
                    request_id=f"batch_{i}_req_{j}",
                    model_name="test-model",
                    prompt="Test",
                    max_tokens=50,
                    temperature=0.7,
                    top_p=0.9
                )
                for j in range(4)
            ]
            
            batch = Batch(batch_id=f"batch_{i}", requests=requests)
            await self.processor.process_batch(batch, mock_inference)
        
        stats = self.processor.get_stats()
        
        assert stats["batches_processed"] == 3
        assert stats["total_requests"] == 12
        assert stats["avg_batch_size"] == 4.0
    
    def test_get_stats(self):
        """Test getting processor statistics."""
        stats = self.processor.get_stats()
        
        assert "batches_processed" in stats
        assert "total_requests" in stats
        assert "successful_requests" in stats
        assert "failed_requests" in stats
        assert "success_rate" in stats
        assert "avg_batch_time_ms" in stats
        assert "avg_batch_size" in stats


class TestPaddedBatch:
    """Test PaddedBatch dataclass."""
    
    def test_padded_batch_creation(self):
        """Test creating padded batch."""
        requests = [
            BatchRequest(
                request_id="req_1",
                model_name="test-model",
                prompt="Test",
                max_tokens=50,
                temperature=0.7,
                top_p=0.9
            )
        ]
        
        padded_batch = PaddedBatch(
            batch_id="batch_1",
            original_requests=requests,
            padded_prompts=["Test <pad>"],
            attention_mask=mx.array([[1.0, 0.0]]),
            max_length=2
        )
        
        assert padded_batch.batch_id == "batch_1"
        assert padded_batch.size == 1
        assert padded_batch.max_length == 2


class TestBatchResult:
    """Test BatchResult dataclass."""
    
    def test_batch_result_creation(self):
        """Test creating batch result."""
        result = BatchResult(
            batch_id="batch_1",
            request_results={},
            processing_time_ms=100.0,
            tokens_generated=50,
            success_count=3,
            failure_count=1
        )
        
        assert result.batch_id == "batch_1"
        assert result.success_rate == 0.75


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
