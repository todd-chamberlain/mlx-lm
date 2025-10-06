"""
HTTP server for Prometheus metrics endpoint in coordinator service.
"""

import asyncio
import logging
from typing import Optional
from aiohttp import web, web_request
from .metrics import get_metrics_data

logger = logging.getLogger(__name__)


class MetricsServer:
    """Simple HTTP server for Prometheus metrics."""
    
    def __init__(self, port: int = 9091):
        self.port = port
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
    
    async def start(self):
        """Start the metrics HTTP server."""
        try:
            self.app = web.Application()
            self.app.router.add_get('/metrics', self._metrics_handler)
            self.app.router.add_get('/health', self._health_handler)
            
            self.runner = web.AppRunner(self.app)
            await self.runner.setup()
            
            self.site = web.TCPSite(self.runner, '0.0.0.0', self.port)
            await self.site.start()
            
            logger.info(f"Metrics server started on port {self.port}")
            
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    async def stop(self):
        """Stop the metrics HTTP server."""
        try:
            if self.site:
                await self.site.stop()
            
            if self.runner:
                await self.runner.cleanup()
            
            logger.info("Metrics server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping metrics server: {e}")
    
    async def _metrics_handler(self, request: web_request.Request) -> web.Response:
        """Handle metrics endpoint."""
        try:
            metrics_data = get_metrics_data()
            return web.Response(
                body=metrics_data,
                content_type='text/plain; version=0.0.4; charset=utf-8'
            )
        except Exception as e:
            logger.error(f"Error generating metrics: {e}")
            return web.Response(
                text="# Error generating metrics\n",
                status=500,
                content_type='text/plain'
            )
    
    async def _health_handler(self, request: web_request.Request) -> web.Response:
        """Handle health check endpoint."""
        return web.Response(
            text="OK",
            content_type='text/plain'
        )