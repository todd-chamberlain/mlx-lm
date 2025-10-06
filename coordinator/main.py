"""
Main entry point for the coordinator service.
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, Any, Optional
import os

from .service import CoordinatorServer
from .metrics_server import MetricsServer
from config.base import load_config_from_file, get_config_path, BaseConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('/tmp/coordinator.log')
    ]
)

logger = logging.getLogger(__name__)


class CoordinatorApp:
    """Main coordinator application."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._load_default_config()
        self.server: Optional[CoordinatorServer] = None
        self.metrics_server: Optional[MetricsServer] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("CoordinatorApp initialized")
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'server': {
                'port': int(os.getenv('COORDINATOR_PORT', '50051')),
                'max_workers': int(os.getenv('COORDINATOR_MAX_WORKERS', '10')),
                'enable_reflection': os.getenv('ENABLE_GRPC_REFLECTION', 'false').lower() == 'true'
            },
            'metrics': {
                'port': int(os.getenv('METRICS_PORT', '9091')),
                'enabled': os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
            },
            'worker_pool': {
                'heartbeat_timeout': float(os.getenv('HEARTBEAT_TIMEOUT', '30.0')),
                'cleanup_interval': float(os.getenv('CLEANUP_INTERVAL', '60.0')),
                'max_workers': int(os.getenv('MAX_WORKERS', '100')),
                'circuit_breaker_failure_threshold': int(os.getenv('CIRCUIT_BREAKER_FAILURE_THRESHOLD', '5')),
                'circuit_breaker_recovery_timeout': float(os.getenv('CIRCUIT_BREAKER_RECOVERY_TIMEOUT', '60.0'))
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL', 'INFO'),
                'file': os.getenv('LOG_FILE', '/tmp/coordinator.log')
            }
        }
    
    async def start(self) -> None:
        """Start the coordinator application."""
        logger.info("Starting coordinator application")
        
        try:
            # Create and configure server
            server_config = self.config['server']
            worker_pool_config = self.config['worker_pool']
            metrics_config = self.config['metrics']
            
            self.server = CoordinatorServer(
                port=server_config['port'],
                max_workers=server_config['max_workers'],
                worker_pool_config=worker_pool_config
            )
            
            # Start metrics server if enabled
            if metrics_config['enabled']:
                self.metrics_server = MetricsServer(port=metrics_config['port'])
                await self.metrics_server.start()
            
            # Set up signal handlers
            self._setup_signal_handlers()
            
            # Start server
            await self.server.start()
            
        except Exception as e:
            logger.error(f"Failed to start coordinator: {e}", exc_info=True)
            raise
    
    async def stop(self) -> None:
        """Stop the coordinator application."""
        logger.info("Stopping coordinator application")
        
        if self.server:
            await self.server.stop()
        
        if self.metrics_server:
            await self.metrics_server.stop()
        
        logger.info("Coordinator application stopped")
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self) -> None:
        """Run the coordinator application."""
        try:
            # Start the application
            start_task = asyncio.create_task(self.start())
            
            # Wait for shutdown signal
            shutdown_task = asyncio.create_task(self._shutdown_event.wait())
            
            # Wait for either start completion or shutdown signal
            done, pending = await asyncio.wait(
                [start_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Check if start failed
            if start_task in done:
                try:
                    await start_task
                except Exception as e:
                    logger.error(f"Coordinator startup failed: {e}")
                    return
            
            # If we get here, we received a shutdown signal
            await self.stop()
            
        except Exception as e:
            logger.error(f"Coordinator application error: {e}", exc_info=True)
            raise


async def main():
    """Main entry point."""
    try:
        # Load configuration
        config_path = os.getenv('COORDINATOR_CONFIG')
        if config_path and os.path.exists(config_path):
            config = load_config(config_path)
        else:
            config = None
        
        # Create and run application
        app = CoordinatorApp(config)
        await app.run()
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())