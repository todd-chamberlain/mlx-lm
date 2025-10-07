"""
Batch Processing Pipeline for MLX Distributed Inference

Implements batch assembly, padding, processing, and result distribution
for efficient multi-request inference execution.
"""

import time
import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import mlx.core as mx

from .batch_manager import Batch, BatchRequest
from .models import InferenceResult

logger = logging.getLogger(__name__)


@dataclass
class PaddedBatch:
    """Batch with padding applied for uniform processing."""
    
    batch_id: str
    original_requests: List[BatchRequest]
    padded_prompts: List[str]
    attention_mask: mx.array
    max_length: int
    padding_token: str = "<pad>"
    
    @property
    def size(self) -> int:
        """Get batch size."""
        return len(self.original_requests)


@dataclass
class BatchResult:
    """Result from batch processing."""
    
    batch_id: str
    request_results: Dict[str, InferenceResult]
    processing_time_ms: float
    tokens_generated: int
    success_count: int
    failure_count: int
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class BatchAssembler:
    """
    Assembles batches with padding and masking for uniform processing.
    
    Features:
    - Prompt padding to uniform length
    - Attention mask generation
    - Batch validation
    - Memory-efficient padding
    """
    
    def __init__(
        self,
        padding_token: str = "<pad>",
        max_padding_ratio: float = 0.5  # Max 50% padding overhead
    ):
        """
        Initialize batch assembler.
        
        Args:
            padding_token: Token to use for padding
            max_padding_ratio: Maximum ratio of padding to content
        """
        self.padding_token = padding_token
        self.max_padding_ratio = max_padding_ratio
        
        logger.info(
            f"Initialized batch assembler: "
            f"padding_token='{padding_token}', "
            f"max_padding_ratio={max_padding_ratio}"
        )
    
    def assemble(self, batch: Batch) -> Optional[PaddedBatch]:
        """
        Assemble batch with padding and masking.
        
        Args:
            batch: Batch to assemble
            
        Returns:
            PaddedBatch with padding applied, or None if assembly fails
        """
        if not batch.requests:
            logger.warning(f"Cannot assemble empty batch: {batch.batch_id}")
            return None
        
        # Get prompt lengths
        prompt_lengths = [req.prompt_length for req in batch.requests]
        max_length = max(prompt_lengths)
        min_length = min(prompt_lengths)
        
        # Check padding ratio
        padding_ratio = (max_length - min_length) / max_length if max_length > 0 else 0.0
        if padding_ratio > self.max_padding_ratio:
            logger.warning(
                f"High padding ratio in batch {batch.batch_id}: "
                f"{padding_ratio:.2%} (max={self.max_padding_ratio:.2%})"
            )
        
        # Pad prompts to uniform length
        padded_prompts = []
        attention_mask_list = []
        
        for req in batch.requests:
            # Pad prompt
            padding_needed = max_length - req.prompt_length
            if padding_needed > 0:
                padded_prompt = req.prompt + " " + self.padding_token * padding_needed
            else:
                padded_prompt = req.prompt
            
            padded_prompts.append(padded_prompt)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            mask = [1.0] * req.prompt_length + [0.0] * padding_needed
            attention_mask_list.append(mask)
        
        # Convert attention mask to MLX array
        try:
            attention_mask = mx.array(attention_mask_list, dtype=mx.float32)
        except Exception as e:
            logger.error(f"Failed to create attention mask: {e}")
            return None
        
        padded_batch = PaddedBatch(
            batch_id=batch.batch_id,
            original_requests=batch.requests,
            padded_prompts=padded_prompts,
            attention_mask=attention_mask,
            max_length=max_length,
            padding_token=self.padding_token
        )
        
        logger.debug(
            f"Assembled batch {batch.batch_id}: "
            f"size={padded_batch.size}, max_length={max_length}, "
            f"padding_ratio={padding_ratio:.2%}"
        )
        
        return padded_batch
    
    def validate_batch(self, batch: Batch) -> Tuple[bool, Optional[str]]:
        """
        Validate batch before assembly.
        
        Args:
            batch: Batch to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not batch.requests:
            return False, "Empty batch"
        
        # Check for consistent model names
        model_names = {req.model_name for req in batch.requests}
        if len(model_names) > 1:
            return False, f"Multiple models in batch: {model_names}"
        
        # Check for reasonable prompt lengths
        max_length = max(req.prompt_length for req in batch.requests)
        if max_length > 10000:  # Arbitrary large limit
            return False, f"Prompt too long: {max_length} tokens"
        
        return True, None


class BatchDisassembler:
    """
    Disassembles batch results back to individual request results.
    
    Features:
    - Result extraction and mapping
    - Error handling for partial failures
    - Token count adjustment for padding
    """
    
    def __init__(self):
        """Initialize batch disassembler."""
        logger.info("Initialized batch disassembler")
    
    def disassemble(
        self,
        padded_batch: PaddedBatch,
        batch_outputs: List[str],
        processing_time_ms: float
    ) -> BatchResult:
        """
        Disassemble batch outputs to individual results.
        
        Args:
            padded_batch: Original padded batch
            batch_outputs: List of generated outputs (one per request)
            processing_time_ms: Total processing time
            
        Returns:
            BatchResult with individual request results
        """
        request_results = {}
        success_count = 0
        failure_count = 0
        total_tokens = 0
        
        for i, req in enumerate(padded_batch.original_requests):
            try:
                # Get output for this request
                if i < len(batch_outputs):
                    output = batch_outputs[i]
                    
                    # Remove padding tokens from output
                    output = output.replace(padded_batch.padding_token, "").strip()
                    
                    # Create result
                    result = InferenceResult(
                        request_id=req.request_id,
                        generated_text=output,
                        tokens_generated=len(output.split()),
                        processing_time_ms=processing_time_ms / padded_batch.size,
                        success=True
                    )
                    
                    request_results[req.request_id] = result
                    success_count += 1
                    total_tokens += result.tokens_generated
                else:
                    # Missing output - create error result
                    result = InferenceResult(
                        request_id=req.request_id,
                        generated_text="",
                        tokens_generated=0,
                        processing_time_ms=0.0,
                        success=False,
                        error="Missing output from batch processing"
                    )
                    request_results[req.request_id] = result
                    failure_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to disassemble result for {req.request_id}: {e}")
                result = InferenceResult(
                    request_id=req.request_id,
                    generated_text="",
                    tokens_generated=0,
                    processing_time_ms=0.0,
                    success=False,
                    error=str(e)
                )
                request_results[req.request_id] = result
                failure_count += 1
        
        batch_result = BatchResult(
            batch_id=padded_batch.batch_id,
            request_results=request_results,
            processing_time_ms=processing_time_ms,
            tokens_generated=total_tokens,
            success_count=success_count,
            failure_count=failure_count
        )
        
        logger.debug(
            f"Disassembled batch {padded_batch.batch_id}: "
            f"success={success_count}, failures={failure_count}, "
            f"tokens={total_tokens}"
        )
        
        return batch_result


class BatchProcessor:
    """
    Main batch processing pipeline coordinator.
    
    Features:
    - End-to-end batch processing
    - Assembly, execution, and disassembly
    - Error handling and partial failure recovery
    - Performance metrics collection
    """
    
    def __init__(
        self,
        assembler: Optional[BatchAssembler] = None,
        disassembler: Optional[BatchDisassembler] = None
    ):
        """
        Initialize batch processor.
        
        Args:
            assembler: Batch assembler (creates default if None)
            disassembler: Batch disassembler (creates default if None)
        """
        self.assembler = assembler or BatchAssembler()
        self.disassembler = disassembler or BatchDisassembler()
        
        # Statistics
        self.stats = {
            "batches_processed": 0,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_processing_time_ms": 0.0,
            "total_tokens_generated": 0
        }
        
        logger.info("Initialized batch processor")
    
    async def process_batch(
        self,
        batch: Batch,
        inference_fn: callable
    ) -> BatchResult:
        """
        Process batch through complete pipeline.
        
        Args:
            batch: Batch to process
            inference_fn: Async function to perform inference
                         Should accept (prompts, max_tokens) and return List[str]
            
        Returns:
            BatchResult with individual request results
        """
        start_time = time.time()
        
        try:
            # Validate batch
            is_valid, error_msg = self.assembler.validate_batch(batch)
            if not is_valid:
                logger.error(f"Batch validation failed: {error_msg}")
                return self._create_error_result(batch, error_msg)
            
            # Assemble batch
            padded_batch = self.assembler.assemble(batch)
            if padded_batch is None:
                return self._create_error_result(batch, "Batch assembly failed")
            
            # Execute inference
            try:
                max_tokens = max(req.max_tokens for req in batch.requests)
                batch_outputs = await inference_fn(
                    padded_batch.padded_prompts,
                    max_tokens
                )
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                return self._create_error_result(batch, f"Inference failed: {e}")
            
            # Disassemble results
            processing_time_ms = (time.time() - start_time) * 1000
            batch_result = self.disassembler.disassemble(
                padded_batch,
                batch_outputs,
                processing_time_ms
            )
            
            # Update statistics
            self._update_stats(batch_result)
            
            logger.info(
                f"Processed batch {batch.batch_id}: "
                f"size={batch.size}, time={processing_time_ms:.1f}ms, "
                f"success_rate={batch_result.success_rate:.2%}"
            )
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}", exc_info=True)
            return self._create_error_result(batch, f"Processing failed: {e}")
    
    def _create_error_result(self, batch: Batch, error_msg: str) -> BatchResult:
        """
        Create error result for failed batch.
        
        Args:
            batch: Original batch
            error_msg: Error message
            
        Returns:
            BatchResult with all requests marked as failed
        """
        request_results = {}
        
        for req in batch.requests:
            result = InferenceResult(
                request_id=req.request_id,
                generated_text="",
                tokens_generated=0,
                processing_time_ms=0.0,
                success=False,
                error=error_msg
            )
            request_results[req.request_id] = result
        
        return BatchResult(
            batch_id=batch.batch_id,
            request_results=request_results,
            processing_time_ms=0.0,
            tokens_generated=0,
            success_count=0,
            failure_count=len(batch.requests)
        )
    
    def _update_stats(self, batch_result: BatchResult):
        """
        Update processor statistics.
        
        Args:
            batch_result: Result from batch processing
        """
        self.stats["batches_processed"] += 1
        self.stats["total_requests"] += (
            batch_result.success_count + batch_result.failure_count
        )
        self.stats["successful_requests"] += batch_result.success_count
        self.stats["failed_requests"] += batch_result.failure_count
        self.stats["total_processing_time_ms"] += batch_result.processing_time_ms
        self.stats["total_tokens_generated"] += batch_result.tokens_generated
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get processor statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats["batches_processed"] > 0:
            stats["avg_batch_time_ms"] = (
                stats["total_processing_time_ms"] / stats["batches_processed"]
            )
            stats["avg_batch_size"] = (
                stats["total_requests"] / stats["batches_processed"]
            )
        else:
            stats["avg_batch_time_ms"] = 0.0
            stats["avg_batch_size"] = 0.0
        
        if stats["total_requests"] > 0:
            stats["success_rate"] = (
                stats["successful_requests"] / stats["total_requests"]
            )
            stats["avg_tokens_per_request"] = (
                stats["total_tokens_generated"] / stats["successful_requests"]
                if stats["successful_requests"] > 0 else 0.0
            )
        else:
            stats["success_rate"] = 0.0
            stats["avg_tokens_per_request"] = 0.0
        
        return stats
