"""
Timeout Handling and Connection Management

Provides timeout utilities, connection pooling, and resource management
for resilient service communication.
"""

import asyncio
import time
import logging
import threading
from typing import Any, Callable, Optional, Dict, Union, ContextManager
from dataclasses import dataclass, field
from contextlib import contextmanager, asynccontextmanager
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import weakref

logger = logging.getLogger(__name__)


@dataclass
class TimeoutConfig:
    """Timeout configuration."""
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 30.0
    total_timeout: float = 60.0
    keepalive_timeout: float = 300.0
    idle_timeout: float = 600.0


class TimeoutError(Exception):
    """Timeout specific exception."""
    pass


class ConnectionTimeoutError(TimeoutError):
    """Connection timeout exception."""
    pass


class ReadTimeoutError(TimeoutError):
    """Read timeout exception."""
    pass


class WriteTimeoutError(TimeoutError):
    """Write timeout exception."""
    pass


class TimeoutManager:
    """Manages timeouts for various operations."""
    
    def __init__(self, config: TimeoutConfig):
        self.config = config
        self._active_timeouts: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def timeout(self, duration: float, operation: str = "operation"):
        """
        Context manager for timeout operations.
        
        Args:
            duration: Timeout duration in seconds
            operation: Operation description for error messages
            
        Raises:
            TimeoutError: When operation times out
        """
        timeout_id = f"{operation}_{id(threading.current_thread())}_{time.time()}"
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"{operation} timed out after {duration}s")
        
        with self._lock:
            self._active_timeouts[timeout_id] = time.time() + duration
        
        # Set up signal handler for timeout
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(duration))
        
        try:
            yield
        finally:
            # Clean up
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            
            with self._lock:
                self._active_timeouts.pop(timeout_id, None)
    
    @asynccontextmanager
    async def async_timeout(self, duration: float, operation: str = "operation"):
        """
        Async context manager for timeout operations.
        
        Args:
            duration: Timeout duration in seconds
            operation: Operation description for error messages
            
        Raises:
            TimeoutError: When operation times out
        """
        timeout_id = f"{operation}_{id(asyncio.current_task())}_{time.time()}"
        
        with self._lock:
            self._active_timeouts[timeout_id] = time.time() + duration
        
        try:
            async with asyncio.timeout(duration):
                yield
        except asyncio.TimeoutError:
            raise TimeoutError(f"{operation} timed out after {duration}s")
        finally:
            with self._lock:
                self._active_timeouts.pop(timeout_id, None)
    
    def call_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """
        Call function with timeout.
        
        Args:
            func: Function to call
            timeout: Timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: When function times out
        """
        with self.timeout(timeout, f"call to {func.__name__}"):
            return func(*args, **kwargs)
    
    async def async_call_with_timeout(self, func: Callable, timeout: float, *args, **kwargs) -> Any:
        """
        Call async function with timeout.
        
        Args:
            func: Async function to call
            timeout: Timeout in seconds
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            TimeoutError: When function times out
        """
        async with self.async_timeout(timeout, f"async call to {func.__name__}"):
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                # Run sync function in executor
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)
    
    def get_active_timeouts(self) -> Dict[str, float]:
        """Get currently active timeouts."""
        current_time = time.time()
        with self._lock:
            return {
                timeout_id: remaining_time - current_time
                for timeout_id, remaining_time in self._active_timeouts.items()
                if remaining_time > current_time
            }
    
    def cancel_all_timeouts(self):
        """Cancel all active timeouts."""
        with self._lock:
            self._active_timeouts.clear()
        
        # Cancel any pending alarms
        signal.alarm(0)


@dataclass
class ConnectionInfo:
    """Connection information and statistics."""
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    is_active: bool = True
    connection_id: str = ""
    remote_address: Optional[str] = None
    
    def mark_used(self):
        """Mark connection as used."""
        self.last_used = time.time()
        self.use_count += 1
    
    def is_expired(self, idle_timeout: float) -> bool:
        """Check if connection is expired."""
        return time.time() - self.last_used > idle_timeout
    
    def age(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at


class ConnectionPool:
    """Generic connection pool with timeout management."""
    
    def __init__(
        self,
        create_connection: Callable,
        close_connection: Callable,
        max_connections: int = 10,
        timeout_config: Optional[TimeoutConfig] = None
    ):
        self.create_connection = create_connection
        self.close_connection = close_connection
        self.max_connections = max_connections
        self.timeout_config = timeout_config or TimeoutConfig()
        
        self._connections: Dict[str, Any] = {}
        self._connection_info: Dict[str, ConnectionInfo] = {}
        self._available_connections: set = set()
        self._lock = threading.Lock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._shutdown = False
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def get_connection(self, connection_key: str = "default") -> Any:
        """
        Get connection from pool.
        
        Args:
            connection_key: Key to identify connection type/destination
            
        Returns:
            Connection object
            
        Raises:
            ConnectionTimeoutError: When connection creation times out
        """
        with self._lock:
            # Try to reuse existing connection
            if connection_key in self._available_connections:
                self._available_connections.remove(connection_key)
                connection = self._connections[connection_key]
                self._connection_info[connection_key].mark_used()
                
                logger.debug(f"Reusing connection: {connection_key}")
                return connection
            
            # Create new connection if under limit
            if len(self._connections) < self.max_connections:
                try:
                    connection = self._create_connection_with_timeout(connection_key)
                    self._connections[connection_key] = connection
                    self._connection_info[connection_key] = ConnectionInfo(
                        connection_id=connection_key
                    )
                    self._connection_info[connection_key].mark_used()
                    
                    logger.debug(f"Created new connection: {connection_key}")
                    return connection
                    
                except Exception as e:
                    logger.error(f"Failed to create connection {connection_key}: {e}")
                    raise ConnectionTimeoutError(f"Failed to create connection: {e}")
            
            # Pool is full, wait for available connection or timeout
            raise ConnectionTimeoutError(
                f"Connection pool full ({self.max_connections} connections)"
            )
    
    def return_connection(self, connection_key: str, connection: Any, is_healthy: bool = True):
        """
        Return connection to pool.
        
        Args:
            connection_key: Connection key
            connection: Connection object
            is_healthy: Whether connection is still healthy
        """
        with self._lock:
            if not is_healthy or connection_key not in self._connections:
                # Close unhealthy or unknown connections
                self._close_connection_safe(connection)
                if connection_key in self._connections:
                    del self._connections[connection_key]
                    del self._connection_info[connection_key]
                logger.debug(f"Closed unhealthy connection: {connection_key}")
                return
            
            # Return healthy connection to available pool
            self._available_connections.add(connection_key)
            logger.debug(f"Returned connection to pool: {connection_key}")
    
    @contextmanager
    def connection(self, connection_key: str = "default"):
        """
        Context manager for getting and returning connections.
        
        Args:
            connection_key: Connection key
            
        Yields:
            Connection object
        """
        conn = None
        try:
            conn = self.get_connection(connection_key)
            yield conn
        except Exception as e:
            # Mark connection as unhealthy on exception
            if conn:
                self.return_connection(connection_key, conn, is_healthy=False)
            raise
        else:
            # Return healthy connection
            if conn:
                self.return_connection(connection_key, conn, is_healthy=True)
    
    def _create_connection_with_timeout(self, connection_key: str) -> Any:
        """Create connection with timeout."""
        timeout_manager = TimeoutManager(self.timeout_config)
        
        return timeout_manager.call_with_timeout(
            self.create_connection,
            self.timeout_config.connect_timeout,
            connection_key
        )
    
    def _close_connection_safe(self, connection: Any):
        """Safely close connection."""
        try:
            self.close_connection(connection)
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_loop():
            while not self._shutdown:
                try:
                    self._cleanup_expired_connections()
                    time.sleep(30)  # Cleanup every 30 seconds
                except Exception as e:
                    logger.error(f"Error in connection cleanup: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_expired_connections(self):
        """Clean up expired connections."""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, info in self._connection_info.items():
                if info.is_expired(self.timeout_config.idle_timeout):
                    expired_keys.append(key)
        
        # Close expired connections
        for key in expired_keys:
            with self._lock:
                if key in self._connections:
                    connection = self._connections[key]
                    del self._connections[key]
                    del self._connection_info[key]
                    self._available_connections.discard(key)
            
            self._close_connection_safe(connection)
            logger.debug(f"Cleaned up expired connection: {key}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        with self._lock:
            return {
                "total_connections": len(self._connections),
                "available_connections": len(self._available_connections),
                "active_connections": len(self._connections) - len(self._available_connections),
                "max_connections": self.max_connections,
                "connection_details": {
                    key: {
                        "age": info.age(),
                        "use_count": info.use_count,
                        "last_used": info.last_used,
                        "is_active": info.is_active
                    }
                    for key, info in self._connection_info.items()
                }
            }
    
    def close_all(self):
        """Close all connections and shutdown pool."""
        self._shutdown = True
        
        with self._lock:
            connections = list(self._connections.values())
            self._connections.clear()
            self._connection_info.clear()
            self._available_connections.clear()
        
        # Close all connections
        for connection in connections:
            self._close_connection_safe(connection)
        
        # Wait for cleanup thread to finish
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        logger.info("Connection pool closed")


class AsyncConnectionPool:
    """Async version of connection pool."""
    
    def __init__(
        self,
        create_connection: Callable,
        close_connection: Callable,
        max_connections: int = 10,
        timeout_config: Optional[TimeoutConfig] = None
    ):
        self.create_connection = create_connection
        self.close_connection = close_connection
        self.max_connections = max_connections
        self.timeout_config = timeout_config or TimeoutConfig()
        
        self._connections: Dict[str, Any] = {}
        self._connection_info: Dict[str, ConnectionInfo] = {}
        self._available_connections: set = set()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        # Start cleanup task
        self._start_cleanup_task()
    
    async def get_connection(self, connection_key: str = "default") -> Any:
        """Get connection from pool (async version)."""
        async with self._lock:
            # Try to reuse existing connection
            if connection_key in self._available_connections:
                self._available_connections.remove(connection_key)
                connection = self._connections[connection_key]
                self._connection_info[connection_key].mark_used()
                
                logger.debug(f"Reusing async connection: {connection_key}")
                return connection
            
            # Create new connection if under limit
            if len(self._connections) < self.max_connections:
                try:
                    connection = await self._create_connection_with_timeout(connection_key)
                    self._connections[connection_key] = connection
                    self._connection_info[connection_key] = ConnectionInfo(
                        connection_id=connection_key
                    )
                    self._connection_info[connection_key].mark_used()
                    
                    logger.debug(f"Created new async connection: {connection_key}")
                    return connection
                    
                except Exception as e:
                    logger.error(f"Failed to create async connection {connection_key}: {e}")
                    raise ConnectionTimeoutError(f"Failed to create connection: {e}")
            
            # Pool is full
            raise ConnectionTimeoutError(
                f"Async connection pool full ({self.max_connections} connections)"
            )
    
    async def return_connection(self, connection_key: str, connection: Any, is_healthy: bool = True):
        """Return connection to pool (async version)."""
        async with self._lock:
            if not is_healthy or connection_key not in self._connections:
                # Close unhealthy or unknown connections
                await self._close_connection_safe(connection)
                if connection_key in self._connections:
                    del self._connections[connection_key]
                    del self._connection_info[connection_key]
                logger.debug(f"Closed unhealthy async connection: {connection_key}")
                return
            
            # Return healthy connection to available pool
            self._available_connections.add(connection_key)
            logger.debug(f"Returned async connection to pool: {connection_key}")
    
    @asynccontextmanager
    async def connection(self, connection_key: str = "default"):
        """Async context manager for connections."""
        conn = None
        try:
            conn = await self.get_connection(connection_key)
            yield conn
        except Exception as e:
            # Mark connection as unhealthy on exception
            if conn:
                await self.return_connection(connection_key, conn, is_healthy=False)
            raise
        else:
            # Return healthy connection
            if conn:
                await self.return_connection(connection_key, conn, is_healthy=True)
    
    async def _create_connection_with_timeout(self, connection_key: str) -> Any:
        """Create connection with timeout (async version)."""
        timeout_manager = TimeoutManager(self.timeout_config)
        
        return await timeout_manager.async_call_with_timeout(
            self.create_connection,
            self.timeout_config.connect_timeout,
            connection_key
        )
    
    async def _close_connection_safe(self, connection: Any):
        """Safely close connection (async version)."""
        try:
            if asyncio.iscoroutinefunction(self.close_connection):
                await self.close_connection(connection)
            else:
                self.close_connection(connection)
        except Exception as e:
            logger.warning(f"Error closing async connection: {e}")
    
    def _start_cleanup_task(self):
        """Start background cleanup task."""
        async def cleanup_loop():
            while not self._shutdown:
                try:
                    await self._cleanup_expired_connections()
                    await asyncio.sleep(30)  # Cleanup every 30 seconds
                except Exception as e:
                    logger.error(f"Error in async connection cleanup: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_expired_connections(self):
        """Clean up expired connections (async version)."""
        expired_keys = []
        
        async with self._lock:
            for key, info in self._connection_info.items():
                if info.is_expired(self.timeout_config.idle_timeout):
                    expired_keys.append(key)
        
        # Close expired connections
        for key in expired_keys:
            async with self._lock:
                if key in self._connections:
                    connection = self._connections[key]
                    del self._connections[key]
                    del self._connection_info[key]
                    self._available_connections.discard(key)
            
            await self._close_connection_safe(connection)
            logger.debug(f"Cleaned up expired async connection: {key}")
    
    async def close_all(self):
        """Close all connections and shutdown pool (async version)."""
        self._shutdown = True
        
        # Cancel cleanup task
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        async with self._lock:
            connections = list(self._connections.values())
            self._connections.clear()
            self._connection_info.clear()
            self._available_connections.clear()
        
        # Close all connections
        for connection in connections:
            await self._close_connection_safe(connection)
        
        logger.info("Async connection pool closed")


# Global timeout manager
_global_timeout_manager = TimeoutManager(TimeoutConfig())


def with_timeout(duration: float, operation: str = "operation"):
    """Global timeout context manager."""
    return _global_timeout_manager.timeout(duration, operation)


def async_with_timeout(duration: float, operation: str = "operation"):
    """Global async timeout context manager."""
    return _global_timeout_manager.async_timeout(duration, operation)