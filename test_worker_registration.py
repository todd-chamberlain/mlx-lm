#!/usr/bin/env python3
"""
Test worker registration with coordinator
"""

import asyncio
import sys
import logging

sys.path.insert(0, '.')

from proto import inference_pb2 as pb
from proto import inference_pb2_grpc
from grpc import aio as grpc_aio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_coordinator_connection():
    """Test if we can connect to the coordinator"""
    logger.info("Testing coordinator connection...")
    
    try:
        channel = grpc_aio.insecure_channel('localhost:50051')
        stub = inference_pb2_grpc.CoordinatorServiceStub(channel)
        
        # Test health check
        logger.info("Sending health check...")
        response = await asyncio.wait_for(
            stub.HealthCheck(pb.HealthCheckRequest()),
            timeout=5.0
        )
        logger.info(f"✓ Coordinator is {response.status}")
        logger.info(f"  Workers: {response.healthy_workers}/{response.total_workers}")
        
        await channel.close()
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to connect to coordinator: {e}")
        return False


async def test_worker_registration():
    """Test worker registration"""
    logger.info("\nTesting worker registration...")
    
    try:
        channel = grpc_aio.insecure_channel('localhost:50051')
        stub = inference_pb2_grpc.CoordinatorServiceStub(channel)
        
        # Create registration request
        registration = pb.WorkerRegistration(
            worker_id="test-worker",
            address="localhost:50052",
            device_info=pb.DeviceInfo(
                model="Test Device",
                memory_gb=64,
                gpu_cores=8,
                os_version="macOS 15",
                mlx_version="0.20.0"
            ),
            max_concurrent=4,
            supported_models=["test-model"]
        )
        
        logger.info("Sending registration...")
        response = await asyncio.wait_for(
            stub.RegisterWorker(registration),
            timeout=5.0
        )
        
        if response.success:
            logger.info(f"✓ Registration successful: {response.message}")
            logger.info(f"  Coordinator ID: {response.coordinator_id}")
        else:
            logger.error(f"✗ Registration failed: {response.message}")
        
        await channel.close()
        return response.success
        
    except Exception as e:
        logger.error(f"✗ Registration error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_heartbeat():
    """Test sending a heartbeat"""
    logger.info("\nTesting heartbeat...")
    
    try:
        channel = grpc_aio.insecure_channel('localhost:50051')
        stub = inference_pb2_grpc.CoordinatorServiceStub(channel)
        
        # Create heartbeat request
        heartbeat = pb.HeartbeatRequest(
            worker_id="test-worker",
            status=pb.WorkerStatus(
                status="healthy",
                current_load=0,
                memory_usage_percent=50.0,
                uptime_seconds=100
            ),
            metrics=pb.WorkerMetrics(
                avg_latency_ms=50.0,
                tokens_per_second=20.0,
                total_requests=10,
                failed_requests=0,
                active_requests=0,
                memory_usage_gb=32.0
            )
        )
        
        logger.info("Sending heartbeat...")
        response = await asyncio.wait_for(
            stub.Heartbeat(heartbeat),
            timeout=5.0
        )
        
        if response.acknowledged:
            logger.info(f"✓ Heartbeat acknowledged: {response.message}")
            logger.info(f"  Heartbeat interval: {response.heartbeat_interval}s")
        else:
            logger.error(f"✗ Heartbeat not acknowledged: {response.message}")
        
        await channel.close()
        return response.acknowledged
        
    except Exception as e:
        logger.error(f"✗ Heartbeat error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def check_registered_workers():
    """Check what workers are registered"""
    logger.info("\nChecking registered workers...")
    
    try:
        channel = grpc_aio.insecure_channel('localhost:50051')
        stub = inference_pb2_grpc.CoordinatorServiceStub(channel)
        
        response = await asyncio.wait_for(
            stub.HealthCheck(pb.HealthCheckRequest(include_workers=True)),
            timeout=5.0
        )
        
        logger.info(f"Status: {response.status}")
        logger.info(f"Workers: {response.healthy_workers}/{response.total_workers}")
        
        if response.worker_health:
            logger.info("\nRegistered workers:")
            for worker in response.worker_health:
                logger.info(f"  - {worker.worker_id}")
                logger.info(f"    Status: {worker.status}")
                logger.info(f"    Load: {worker.current_load}")
                logger.info(f"    Last heartbeat: {worker.last_heartbeat_seconds}s ago")
        else:
            logger.info("No workers registered")
        
        await channel.close()
        
    except Exception as e:
        logger.error(f"✗ Error checking workers: {e}")


async def main():
    """Run all tests"""
    print("=" * 70)
    print("Worker Registration Test")
    print("=" * 70)
    
    # Test 1: Coordinator connection
    if not await test_coordinator_connection():
        print("\n❌ Coordinator is not accessible")
        return 1
    
    # Test 2: Worker registration
    if not await test_worker_registration():
        print("\n❌ Worker registration failed")
        return 1
    
    # Test 3: Heartbeat
    if not await test_heartbeat():
        print("\n❌ Heartbeat failed")
        return 1
    
    # Test 4: Check registered workers
    await check_registered_workers()
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
