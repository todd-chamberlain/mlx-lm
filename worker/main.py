"""
MLX Worker Main Service

This is the main entry point for the MLX worker that coordinates all components:
model management, inference engine, gRPC service, coordinator client, and health monitoring.
"""

import asyncio
import logging
import signal
import sys
import os
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .model_manager import ModelManager
from .inference_engine import InferenceEngine
from .grpc_service import WorkerGrpcServer
from .coordinator_client import CoordinatorClient, NetworkDiscovery
from .health_monitor import HealthMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WorkerConfig:
    """Configuration management for MLX worker"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        # Default configuration
        config = {
            'worker_id': os.getenv('WORKER_ID', f'worker-{os.uname().nodename}'),
            'port': int(os.getenv('WORKER_PORT', '8080')),
            'coordinator_host': os.getenv('COORDINATOR_HOST', 'localhost:50051'),
            'model_path': os.getenv('MODEL_PATH', '/models/default'),
            'model_name': os.getenv('MODEL_NAME', 'default'),
            'max_concurrent': int(os.getenv('MAX_CONCURRENT', '4')),
            'heartbeat_interval': int(os.getenv('HEARTBEAT_INTERVAL', '10')),
            'health_check_interval': float(os.getenv('HEALTH_CHECK_INTERVAL', '30.0')),
            'auto_discover_coordinator': os.getenv('AUTO_DISCOVER_COORDINATOR', 'true').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        }
        
        # Load from config file if provided
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    config.update(file_config)
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Failed to load config file {config_path}: {e}")
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dict syntax"""
        return self.config[key]


class MLXWorker:
    """Main MLX Worker service"""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.worker_id = config['worker_id']
        
        # Initialize components
        self.model_manager = ModelManager()
        self.inference_engine = InferenceEngine(self.model_manager, self.worker_id)
        self.grpc_server = WorkerGrpcServer(
            self.model_manager, 
            self.inference_engine, 
            port=config['port']
        )
        self.coordinator_client = CoordinatorClient(
            worker_id=self.worker_id,
            model_manager=self.model_manager,
            inference_engine=self.inference_engine,
            worker_port=config['port'],
            coordinator_host=config['coordinator_host'],
            heartbeat_interval=config['heartbeat_interval']
        )
        self.health_monitor = HealthMonitor(
            self.model_manager,
            self.inference_engine,
            check_interval=config['health_check_interval']
        )
        
        # Shutdown handling
        self.shutdown_event = asyncio.Event()
        self.setup_signal_handlers()
        
        logger.info(f"Initialized MLXWorker: {self.worker_id}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def initialize(self) -> bool:
        """Initialize all components"""
        try:
            logger.info("Initializing MLX Worker...")
            
            # Load model
            model_path = self.config['model_path']
            model_name = self.config['model_name']
            
            if not Path(model_path).exists():
                logger.error(f"Model path does not exist: {model_path}")
                return False
            
            logger.info(f"Loading model: {model_name} from {model_path}")
            if not self.model_manager.load_model(model_path, model_name):
                logger.error("Failed to load model")
                return False
            
            # Discover coordinator if needed
            coordinator_host = self.config['coordinator_host']
            if self.config['auto_discover_coordinator']:
                discovered = await NetworkDiscovery.discover_coordinator()
                if discovered:
                    coordinator_host = discovered
                    self.coordinator_client.coordinator_host = coordinator_host
                    logger.info(f"Using discovered coordinator: {coordinator_host}")
            
            logger.info("MLX Worker initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MLX Worker: {e}")
            return False
    
    async def start(self) -> bool:
        """Start all services"""
        try:
            logger.info("Starting MLX Worker services...")
            
            # Start gRPC server
            await self.grpc_server.start()
            
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Connect to coordinator and register
            if not await self.coordinator_client.start(self.config['max_concurrent']):
                logger.error("Failed to connect to coordinator")
                return False
            
            logger.info("MLX Worker started successfully")
            logger.info(f"Worker ID: {self.worker_id}")
            logger.info(f"Listening on port: {self.config['port']}")
            logger.info(f"Coordinator: {self.coordinator_client.coordinator_host}")
            logger.info(f"Max concurrent requests: {self.config['max_concurrent']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MLX Worker: {e}")
            return False
    
    async def run(self):
        """Main run loop"""
        try:
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Error in main run loop: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown of all services"""
        if self.shutdown_event.is_set():
            return  # Already shutting down
        
        logger.info("Shutting down MLX Worker...")
        self.shutdown_event.set()
        
        try:
            # Stop coordinator client (deregister)
            await self.coordinator_client.stop()
            
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()
            
            # Stop gRPC server
            await self.grpc_server.stop()
            
            logger.info("MLX Worker shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive worker status"""
        return {
            'worker_id': self.worker_id,
            'config': dict(self.config.config),
            'model_manager': {
                'loaded_models': list(self.model_manager.list_loaded_models().keys()),
                'device_info': self.model_manager.get_device_info().__dict__,
                'memory_usage': self.model_manager.get_memory_usage()
            },
            'inference_engine': self.inference_engine.get_metrics(),
            'grpc_server': self.grpc_server.get_service_info(),
            'coordinator_client': self.coordinator_client.get_status(),
            'health': self.health_monitor.get_health_status().__dict__
        }


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='MLX Distributed Inference Worker')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--worker-id', help='Worker ID override')
    parser.add_argument('--port', type=int, help='Port to listen on')
    parser.add_argument('--coordinator', help='Coordinator host:port')
    parser.add_argument('--model-path', help='Path to model directory')
    parser.add_argument('--model-name', help='Model name')
    parser.add_argument('--max-concurrent', type=int, help='Max concurrent requests')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help='Log level')
    
    args = parser.parse_args()
    
    # Load configuration
    config = WorkerConfig(args.config)
    
    # Override with command line arguments
    if args.worker_id:
        config.config['worker_id'] = args.worker_id
    if args.port:
        config.config['port'] = args.port
    if args.coordinator:
        config.config['coordinator_host'] = args.coordinator
    if args.model_path:
        config.config['model_path'] = args.model_path
    if args.model_name:
        config.config['model_name'] = args.model_name
    if args.max_concurrent:
        config.config['max_concurrent'] = args.max_concurrent
    if args.log_level:
        config.config['log_level'] = args.log_level
    
    # Set log level
    log_level = getattr(logging, config['log_level'].upper())
    logging.getLogger().setLevel(log_level)
    
    # Create and run worker
    worker = MLXWorker(config)
    
    try:
        # Initialize
        if not await worker.initialize():
            logger.error("Failed to initialize worker")
            sys.exit(1)
        
        # Start services
        if not await worker.start():
            logger.error("Failed to start worker")
            sys.exit(1)
        
        # Run main loop
        await worker.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        await worker.shutdown()


if __name__ == '__main__':
    asyncio.run(main())