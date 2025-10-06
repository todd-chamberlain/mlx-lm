"""
Tests for comprehensive logging and error tracking system.
"""

import pytest
import logging
import json
import time
from io import StringIO
from unittest.mock import patch, MagicMock

from common.logging_system import (
    setup_logging,
    get_logger,
    request_context,
    log_error,
    get_error_tracker,
    log_performance,
    log_operation,
    StructuredFormatter,
    ContextLogger,
    ErrorTracker,
)


class TestStructuredFormatter:
    """Test structured JSON logging formatter."""
    
    def test_basic_formatting(self):
        """Test basic log record formatting."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"
        assert data["line"] == 10
        assert "timestamp" in data
    
    def test_exception_formatting(self):
        """Test exception information in logs."""
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test error")
        except ValueError:
            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=True
            )
            record.exc_info = (ValueError, ValueError("Test error"), None)
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert "exception" in data
        assert data["exception"]["type"] == "ValueError"
        assert "Test error" in data["exception"]["message"]
    
    def test_extra_fields(self):
        """Test adding extra fields to logs."""
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.extra_fields = {"user_id": "123", "action": "login"}
        
        result = formatter.format(record)
        data = json.loads(result)
        
        assert data["user_id"] == "123"
        assert data["action"] == "login"


class TestContextLogger:
    """Test context-aware logger."""
    
    def test_logger_with_context(self):
        """Test logger adds context to messages."""
        base_logger = logging.getLogger("test")
        logger = ContextLogger(base_logger, {"service": "api", "version": "1.0"})
        
        assert logger.extra["service"] == "api"
        assert logger.extra["version"] == "1.0"
    
    def test_process_adds_context(self):
        """Test process method adds context."""
        base_logger = logging.getLogger("test")
        logger = ContextLogger(base_logger, {"service": "api"})
        
        msg, kwargs = logger.process("Test", {"extra": {"request_id": "123"}})
        
        assert msg == "Test"
        assert "extra" in kwargs
        assert "extra_fields" in kwargs["extra"]
        assert kwargs["extra"]["extra_fields"]["service"] == "api"


class TestErrorTracker:
    """Test error tracking and aggregation."""
    
    def test_track_error(self):
        """Test tracking an error."""
        tracker = ErrorTracker()
        
        tracker.track_error(
            "validation_error",
            "Invalid input",
            context={"field": "email"}
        )
        
        assert len(tracker.errors) == 1
        assert tracker.error_counts["validation_error"] == 1
        assert tracker.errors[0]["error_type"] == "validation_error"
        assert tracker.errors[0]["message"] == "Invalid input"
    
    def test_track_error_with_exception(self):
        """Test tracking error with exception."""
        tracker = ErrorTracker()
        
        try:
            raise ValueError("Test error")
        except ValueError as e:
            tracker.track_error(
                "runtime_error",
                "Operation failed",
                exception=e
            )
        
        assert len(tracker.errors) == 1
        assert "exception" in tracker.errors[0]
        assert tracker.errors[0]["exception"]["type"] == "ValueError"
    
    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        tracker = ErrorTracker()
        
        # Track multiple errors
        for _ in range(5):
            tracker.track_error("test_error", "Test")
            time.sleep(0.01)
        
        rate = tracker.get_error_rate("test_error")
        assert rate == 5  # 5 errors in last minute
    
    def test_total_error_rate(self):
        """Test total error rate across types."""
        tracker = ErrorTracker()
        
        tracker.track_error("error_type_1", "Error 1")
        tracker.track_error("error_type_1", "Error 2")
        tracker.track_error("error_type_2", "Error 3")
        
        total_rate = tracker.get_total_error_rate()
        assert total_rate == 3
    
    def test_error_summary(self):
        """Test error summary generation."""
        tracker = ErrorTracker()
        
        tracker.track_error("error_1", "Message 1")
        tracker.track_error("error_1", "Message 2")
        tracker.track_error("error_2", "Message 3")
        
        summary = tracker.get_error_summary()
        
        assert summary["total_errors"] == 3
        assert summary["error_counts"]["error_1"] == 2
        assert summary["error_counts"]["error_2"] == 1
        assert "error_rates" in summary
    
    def test_recent_errors(self):
        """Test getting recent errors."""
        tracker = ErrorTracker()
        
        for i in range(15):
            tracker.track_error("test", f"Error {i}")
        
        recent = tracker.get_recent_errors(limit=5)
        assert len(recent) == 5
        assert recent[-1]["message"] == "Error 14"
    
    def test_max_errors_limit(self):
        """Test error list size limit."""
        tracker = ErrorTracker(max_errors=10)
        
        for i in range(20):
            tracker.track_error("test", f"Error {i}")
        
        assert len(tracker.errors) == 10
        # Should keep most recent errors
        assert tracker.errors[-1]["message"] == "Error 19"


class TestLoggingSetup:
    """Test logging setup and configuration."""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup."""
        logger = setup_logging("test_service", log_level="INFO")
        
        assert logger.name == "test_service"
        assert logger.level == logging.INFO
        assert len(logger.handlers) > 0
    
    def test_setup_logging_with_file(self, tmp_path):
        """Test logging setup with file output."""
        log_file = tmp_path / "test.log"
        logger = setup_logging(
            "test_service",
            log_level="DEBUG",
            log_file=str(log_file)
        )
        
        logger.info("Test message")
        
        assert log_file.exists()
        content = log_file.read_text()
        assert "Test message" in content
    
    def test_setup_logging_json_format(self):
        """Test JSON formatted logging."""
        logger = setup_logging("test_service", json_format=True)
        
        # Check that handler has StructuredFormatter
        assert any(
            isinstance(h.formatter, StructuredFormatter)
            for h in logger.handlers
        )
    
    def test_setup_logging_plain_format(self):
        """Test plain text formatted logging."""
        logger = setup_logging("test_service", json_format=False)
        
        # Check that handler doesn't have StructuredFormatter
        assert not any(
            isinstance(h.formatter, StructuredFormatter)
            for h in logger.handlers
        )


class TestGetLogger:
    """Test logger retrieval with context."""
    
    def test_get_logger_basic(self):
        """Test getting a basic logger."""
        logger = get_logger("test")
        
        assert isinstance(logger, ContextLogger)
        assert logger.logger.name == "test"
    
    def test_get_logger_with_context(self):
        """Test getting logger with context."""
        logger = get_logger("test", service="api", version="1.0")
        
        assert logger.extra["service"] == "api"
        assert logger.extra["version"] == "1.0"


class TestRequestContext:
    """Test request context management."""
    
    def test_request_context_basic(self):
        """Test basic request context."""
        with request_context() as ctx:
            assert "request_id" in ctx
            assert "trace_id" in ctx
            assert ctx["request_id"] is not None
            assert ctx["trace_id"] is not None
    
    def test_request_context_with_ids(self):
        """Test request context with provided IDs."""
        with request_context(
            request_id="req-123",
            trace_id="trace-456"
        ) as ctx:
            assert ctx["request_id"] == "req-123"
            assert ctx["trace_id"] == "trace-456"
    
    def test_request_context_with_user(self):
        """Test request context with user ID."""
        with request_context(user_id="user-789") as ctx:
            assert ctx["user_id"] == "user-789"
    
    def test_request_context_extra_fields(self):
        """Test request context with extra fields."""
        with request_context(
            request_id="req-123",
            endpoint="/api/chat",
            method="POST"
        ) as ctx:
            assert ctx["endpoint"] == "/api/chat"
            assert ctx["method"] == "POST"
    
    def test_request_context_cleanup(self):
        """Test context is cleaned up after use."""
        from common.logging_system import _request_context
        
        with request_context(request_id="req-123"):
            assert hasattr(_request_context, 'request_id')
        
        # After context, should be cleaned up
        assert not hasattr(_request_context, 'request_id')


class TestLogError:
    """Test error logging functionality."""
    
    def test_log_error_basic(self):
        """Test basic error logging."""
        logger = logging.getLogger("test")
        tracker = get_error_tracker()
        initial_count = len(tracker.errors)
        
        log_error(logger, "test_error", "Test error message")
        
        assert len(tracker.errors) == initial_count + 1
        assert tracker.errors[-1]["error_type"] == "test_error"
    
    def test_log_error_with_exception(self):
        """Test error logging with exception."""
        logger = logging.getLogger("test")
        tracker = get_error_tracker()
        
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            log_error(logger, "runtime_error", "Operation failed", exception=e)
        
        assert tracker.errors[-1]["exception"]["type"] == "ValueError"
    
    def test_log_error_with_context(self):
        """Test error logging with context."""
        logger = logging.getLogger("test")
        tracker = get_error_tracker()
        
        log_error(
            logger,
            "validation_error",
            "Invalid input",
            field="email",
            value="invalid"
        )
        
        assert tracker.errors[-1]["context"]["field"] == "email"
        assert tracker.errors[-1]["context"]["value"] == "invalid"


class TestLogPerformance:
    """Test performance logging."""
    
    def test_log_performance_basic(self, caplog):
        """Test basic performance logging."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        
        with caplog.at_level(logging.INFO):
            log_performance(logger, "test_operation", 123.45)
        
        assert "test_operation" in caplog.text
        assert "123.45ms" in caplog.text
    
    def test_log_performance_with_context(self, caplog):
        """Test performance logging with context."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        
        with caplog.at_level(logging.INFO):
            log_performance(
                logger,
                "api_request",
                250.0,
                endpoint="/api/chat",
                status=200
            )
        
        assert "api_request" in caplog.text
        assert "250.00ms" in caplog.text


class TestLogOperation:
    """Test operation logging context manager."""
    
    def test_log_operation_success(self, caplog):
        """Test successful operation logging."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        
        with caplog.at_level(logging.INFO):
            with log_operation(logger, "test_operation"):
                time.sleep(0.01)
        
        assert "Starting: test_operation" in caplog.text
        assert "Performance: test_operation" in caplog.text or "Completed: test_operation" in caplog.text
    
    def test_log_operation_failure(self, caplog):
        """Test failed operation logging."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        tracker = get_error_tracker()
        initial_count = len(tracker.errors)
        
        with caplog.at_level(logging.INFO):
            with pytest.raises(ValueError):
                with log_operation(logger, "failing_operation"):
                    raise ValueError("Operation failed")
        
        assert "Starting: failing_operation" in caplog.text
        assert len(tracker.errors) > initial_count
    
    def test_log_operation_with_context(self, caplog):
        """Test operation logging with context."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        
        with caplog.at_level(logging.INFO):
            with log_operation(logger, "api_call", endpoint="/test"):
                pass
        
        assert "Starting: api_call" in caplog.text
    
    def test_log_operation_without_performance(self, caplog):
        """Test operation logging without performance metrics."""
        logger = logging.getLogger("test")
        logger.setLevel(logging.INFO)
        
        with caplog.at_level(logging.INFO):
            with log_operation(logger, "test_op", log_performance_metrics=False):
                pass
        
        assert "Starting: test_op" in caplog.text
        assert "Completed: test_op" in caplog.text
        assert "Performance:" not in caplog.text


class TestIntegration:
    """Integration tests for logging system."""
    
    def test_full_request_flow(self, caplog):
        """Test complete request logging flow."""
        logger = setup_logging("api_service", log_level="INFO", json_format=False)
        
        with caplog.at_level(logging.INFO):
            with request_context(request_id="req-123", user_id="user-456"):
                with log_operation(logger, "process_request"):
                    logger.info("Processing request")
                    log_performance(logger, "database_query", 45.2)
        
        assert "Processing request" in caplog.text
        assert "process_request" in caplog.text
    
    def test_error_tracking_integration(self):
        """Test error tracking across multiple operations."""
        logger = setup_logging("test_service", log_level="INFO")
        tracker = get_error_tracker()
        initial_count = len(tracker.errors)
        
        # Simulate multiple errors
        log_error(logger, "error_1", "First error")
        log_error(logger, "error_1", "Second error")
        log_error(logger, "error_2", "Different error")
        
        summary = tracker.get_error_summary()
        
        assert summary["total_errors"] >= initial_count + 3
        assert summary["error_counts"]["error_1"] >= 2
        assert summary["error_counts"]["error_2"] >= 1
    
    def test_concurrent_request_contexts(self):
        """Test multiple concurrent request contexts."""
        logger = setup_logging("test_service")
        
        # Simulate concurrent requests (in same thread for testing)
        contexts = []
        for i in range(3):
            with request_context(request_id=f"req-{i}") as ctx:
                contexts.append(ctx["request_id"])
        
        # Each should have unique request ID
        assert len(set(contexts)) == 3
