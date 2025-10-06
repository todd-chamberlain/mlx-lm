"""
Tests for service recovery and health management system.
"""

import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from common.service_recovery import (
    ServiceHealthMonitor,
    ServiceState,
    HealthCheckResult,
    RecoveryAction
)
from common.resource_monitor import (
    ResourceMonitor,
    ResourceThresholds,
    ResourceUsage,
    CleanupManager
)
from common.data_consistency import (
    DataConsistencyChecker,
    ConsistencyIssue,
    FileIntegrityChecker
)


class TestServiceHealthMonitor:
    """Test service health monitoring."""
    
    @pytest.fixture
    def monitor(self):
        """Create a health monitor instance."""
        return ServiceHealthMonitor(
            service_name="test-service",
            check_interval=0.1,
            unhealthy_threshold=2,
            recovery_threshold=2
        )
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.service_name == "test-service"
        assert monitor.current_state == ServiceState.HEALTHY
        assert monitor.consecutive_failures == 0
        assert monitor.consecutive_successes == 0
    
    def test_add_health_check(self, monitor):
        """Test adding health checks."""
        def check_func():
            return True
        
        monitor.add_health_check(check_func)
        assert len(monitor.health_checks) == 1
        assert monitor.health_checks[0] == check_func

    
    def test_add_recovery_action(self, monitor):
        """Test adding recovery actions."""
        def recovery_func():
            return True
        
        monitor.add_recovery_action("test-recovery", recovery_func, priority=10)
        assert len(monitor.recovery_actions) == 1
        assert monitor.recovery_actions[0].name == "test-recovery"
        assert monitor.recovery_actions[0].priority == 10
    
    @pytest.mark.asyncio
    async def test_perform_health_check_success(self, monitor):
        """Test successful health check."""
        def healthy_check():
            return HealthCheckResult(
                is_healthy=True,
                state=ServiceState.HEALTHY,
                message="All good"
            )
        
        monitor.add_health_check(healthy_check)
        result = await monitor.perform_health_check()
        
        assert result.is_healthy
        assert result.state == ServiceState.HEALTHY
    
    @pytest.mark.asyncio
    async def test_perform_health_check_failure(self, monitor):
        """Test failed health check."""
        def unhealthy_check():
            return HealthCheckResult(
                is_healthy=False,
                state=ServiceState.UNHEALTHY,
                message="Service down"
            )
        
        monitor.add_health_check(unhealthy_check)
        result = await monitor.perform_health_check()
        
        assert not result.is_healthy
        assert result.state == ServiceState.UNHEALTHY
    
    @pytest.mark.asyncio
    async def test_state_transition_to_unhealthy(self, monitor):
        """Test state transition to unhealthy."""
        def failing_check():
            return HealthCheckResult(
                is_healthy=False,
                state=ServiceState.UNHEALTHY,
                message="Failed"
            )
        
        monitor.add_health_check(failing_check)
        
        # First failure - should be degraded
        result = await monitor.perform_health_check()
        await monitor.update_state(result)
        assert monitor.current_state == ServiceState.DEGRADED
        assert monitor.consecutive_failures == 1
        
        # Second failure - should be unhealthy
        result = await monitor.perform_health_check()
        await monitor.update_state(result)
        assert monitor.current_state == ServiceState.UNHEALTHY
        assert monitor.consecutive_failures == 2
    
    @pytest.mark.asyncio
    async def test_state_recovery(self, monitor):
        """Test recovery from unhealthy state."""
        # Set to unhealthy state
        monitor.current_state = ServiceState.UNHEALTHY
        monitor.consecutive_failures = 3
        
        def healthy_check():
            return HealthCheckResult(
                is_healthy=True,
                state=ServiceState.HEALTHY,
                message="Recovered"
            )
        
        monitor.add_health_check(healthy_check)
        
        # First success - should be recovering
        result = await monitor.perform_health_check()
        await monitor.update_state(result)
        assert monitor.current_state == ServiceState.RECOVERING
        assert monitor.consecutive_successes == 1
        
        # Second success - should be healthy
        result = await monitor.perform_health_check()
        await monitor.update_state(result)
        assert monitor.current_state == ServiceState.HEALTHY
        assert monitor.consecutive_successes == 0
    
    @pytest.mark.asyncio
    async def test_trigger_recovery(self, monitor):
        """Test triggering recovery actions."""
        recovery_called = False
        
        def recovery_action():
            nonlocal recovery_called
            recovery_called = True
            return True
        
        monitor.add_recovery_action("test-recovery", recovery_action)
        result = await monitor.trigger_recovery()
        
        assert result
        assert recovery_called
    
    @pytest.mark.asyncio
    async def test_recovery_retry_limit(self, monitor):
        """Test recovery action retry limit."""
        call_count = 0
        
        def failing_recovery():
            nonlocal call_count
            call_count += 1
            return False
        
        monitor.add_recovery_action("failing-recovery", failing_recovery, max_retries=3)
        
        # Trigger recovery multiple times
        for _ in range(5):
            await monitor.trigger_recovery()
        
        # Should only be called max_retries times
        assert call_count == 3
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self, monitor):
        """Test monitoring loop."""
        check_count = 0
        
        def counting_check():
            nonlocal check_count
            check_count += 1
            return HealthCheckResult(
                is_healthy=True,
                state=ServiceState.HEALTHY,
                message="OK"
            )
        
        monitor.add_health_check(counting_check)
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Wait for a few checks
        await asyncio.sleep(0.3)
        
        # Stop monitoring
        await monitor.stop_monitoring()
        
        # Should have performed multiple checks
        assert check_count >= 2
    
    def test_get_health_status(self, monitor):
        """Test getting health status."""
        status = monitor.get_health_status()
        
        assert status["service"] == "test-service"
        assert status["state"] == ServiceState.HEALTHY.value
        assert "consecutive_failures" in status
        assert "consecutive_successes" in status


class TestResourceMonitor:
    """Test resource monitoring."""
    
    @pytest.fixture
    def monitor(self):
        """Create a resource monitor instance."""
        thresholds = ResourceThresholds(
            cpu_percent=80.0,
            memory_percent=85.0,
            disk_percent=90.0
        )
        return ResourceMonitor(thresholds=thresholds, check_interval=0.1)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization."""
        assert monitor.thresholds.cpu_percent == 80.0
        assert monitor.thresholds.memory_percent == 85.0
        assert len(monitor.cleanup_actions) == 0
    
    def test_get_current_usage(self, monitor):
        """Test getting current resource usage."""
        usage = monitor.get_current_usage()
        
        assert isinstance(usage, ResourceUsage)
        assert usage.cpu_percent >= 0
        assert usage.memory_percent >= 0
        assert usage.disk_percent >= 0
        assert usage.disk_free_gb >= 0
    
    def test_check_thresholds_normal(self, monitor):
        """Test threshold checking with normal usage."""
        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_percent=70.0,
            disk_free_gb=100.0,
            timestamp=datetime.now()
        )
        
        exceeded = monitor.check_thresholds(usage)
        assert len(exceeded) == 0
    
    def test_check_thresholds_exceeded(self, monitor):
        """Test threshold checking with exceeded limits."""
        usage = ResourceUsage(
            cpu_percent=85.0,
            memory_percent=90.0,
            memory_used_gb=14.0,
            memory_available_gb=2.0,
            disk_percent=95.0,
            disk_free_gb=2.0,
            timestamp=datetime.now()
        )
        
        exceeded = monitor.check_thresholds(usage)
        assert 'cpu' in exceeded
        assert 'memory' in exceeded
        assert 'disk' in exceeded
    
    @pytest.mark.asyncio
    async def test_trigger_cleanup(self, monitor):
        """Test triggering cleanup actions."""
        cleanup_called = False
        
        def cleanup_action(exceeded):
            nonlocal cleanup_called
            cleanup_called = True
        
        monitor.add_cleanup_action(cleanup_action)
        await monitor.trigger_cleanup({'memory': True})
        
        assert cleanup_called
    
    @pytest.mark.asyncio
    async def test_monitoring_loop(self, monitor):
        """Test resource monitoring loop."""
        await monitor.start_monitoring()
        
        # Wait for a few checks
        await asyncio.sleep(0.3)
        
        await monitor.stop_monitoring()
        
        # Should have collected usage data
        assert len(monitor.usage_history) > 0
    
    def test_get_resource_status(self, monitor):
        """Test getting resource status."""
        # Add some usage data
        usage = ResourceUsage(
            cpu_percent=50.0,
            memory_percent=60.0,
            memory_used_gb=8.0,
            memory_available_gb=8.0,
            disk_percent=70.0,
            disk_free_gb=100.0,
            timestamp=datetime.now()
        )
        monitor.usage_history.append(usage)
        
        status = monitor.get_resource_status()
        
        assert status["cpu_percent"] == 50.0
        assert status["memory_percent"] == 60.0
        assert "timestamp" in status


class TestCleanupManager:
    """Test cleanup manager."""
    
    @pytest.fixture
    def cleanup_manager(self, tmp_path):
        """Create a cleanup manager instance."""
        log_dir = tmp_path / "logs"
        cache_dir = tmp_path / "cache"
        log_dir.mkdir()
        cache_dir.mkdir()
        return CleanupManager(log_dir=log_dir, cache_dir=cache_dir)
    
    def test_cleanup_old_logs(self, cleanup_manager, tmp_path):
        """Test cleaning up old log files."""
        log_dir = tmp_path / "logs"
        
        # Create some test log files
        old_log = log_dir / "old.log"
        old_log.write_text("old log")
        
        # Set modification time to old
        import os
        old_time = time.time() - (8 * 86400)  # 8 days ago
        os.utime(old_log, (old_time, old_time))
        
        # Clean up logs older than 7 days
        count = cleanup_manager.cleanup_old_logs(max_age_days=7)
        
        assert count == 1
        assert not old_log.exists()
    
    def test_cleanup_cache(self, cleanup_manager, tmp_path):
        """Test cleaning up cache directory."""
        cache_dir = tmp_path / "cache"
        
        # Create some cache files
        cache_file1 = cache_dir / "cache1.dat"
        cache_file2 = cache_dir / "cache2.dat"
        cache_file1.write_text("cache data 1")
        cache_file2.write_text("cache data 2")
        
        count = cleanup_manager.cleanup_cache()
        
        assert count == 2
        assert not cache_file1.exists()
        assert not cache_file2.exists()


class TestDataConsistencyChecker:
    """Test data consistency checker."""
    
    @pytest.fixture
    def checker(self, tmp_path):
        """Create a consistency checker instance."""
        return DataConsistencyChecker(data_dir=tmp_path)
    
    def test_checker_initialization(self, checker):
        """Test checker initialization."""
        assert len(checker.consistency_checks) == 0
        assert len(checker.repair_actions) == 0
        assert len(checker.issues) == 0
    
    def test_add_consistency_check(self, checker):
        """Test adding consistency checks."""
        def check_func():
            return []
        
        checker.add_consistency_check(check_func)
        assert len(checker.consistency_checks) == 1
    
    def test_add_repair_action(self, checker):
        """Test adding repair actions."""
        def repair_func(issue):
            return True
        
        checker.add_repair_action("test-issue", repair_func)
        assert "test-issue" in checker.repair_actions
    
    @pytest.mark.asyncio
    async def test_run_consistency_checks(self, checker):
        """Test running consistency checks."""
        def check_func():
            return [ConsistencyIssue(
                issue_type="test-issue",
                severity="low",
                description="Test issue",
                affected_data="test-data",
                timestamp=datetime.now()
            )]
        
        checker.add_consistency_check(check_func)
        issues = await checker.run_consistency_checks()
        
        assert len(issues) == 1
        assert issues[0].issue_type == "test-issue"
    
    @pytest.mark.asyncio
    async def test_repair_issue(self, checker):
        """Test repairing an issue."""
        issue = ConsistencyIssue(
            issue_type="test-issue",
            severity="low",
            description="Test issue",
            affected_data="test-data",
            timestamp=datetime.now()
        )
        
        def repair_func(issue):
            return True
        
        checker.add_repair_action("test-issue", repair_func)
        result = await checker.repair_issue(issue)
        
        assert result.success
        assert len(checker.repair_history) == 1
    
    @pytest.mark.asyncio
    async def test_repair_all_issues(self, checker):
        """Test repairing all issues."""
        # Add some issues
        checker.issues = [
            ConsistencyIssue(
                issue_type="repairable",
                severity="low",
                description="Repairable issue",
                affected_data="data1",
                timestamp=datetime.now(),
                auto_repairable=True
            ),
            ConsistencyIssue(
                issue_type="manual",
                severity="high",
                description="Manual issue",
                affected_data="data2",
                timestamp=datetime.now(),
                auto_repairable=False
            )
        ]
        
        def repair_func(issue):
            return True
        
        checker.add_repair_action("repairable", repair_func)
        
        stats = await checker.repair_all_issues(auto_only=True)
        
        assert stats["total"] == 2
        assert stats["repaired"] == 1
        assert stats["skipped"] == 1
    
    def test_get_consistency_report(self, checker):
        """Test getting consistency report."""
        checker.issues = [
            ConsistencyIssue(
                issue_type="test",
                severity="high",
                description="Test issue",
                affected_data="data",
                timestamp=datetime.now()
            )
        ]
        
        report = checker.get_consistency_report()
        
        assert report["total_issues"] == 1
        assert "high" in report["issues_by_severity"]


class TestFileIntegrityChecker:
    """Test file integrity checker."""
    
    @pytest.fixture
    def checker(self, tmp_path):
        """Create a file integrity checker instance."""
        manifest_file = tmp_path / "manifest.json"
        return FileIntegrityChecker(manifest_file=manifest_file)
    
    def test_checker_initialization(self, checker):
        """Test checker initialization."""
        assert isinstance(checker.manifest, dict)
    
    def test_add_and_verify_file(self, checker, tmp_path):
        """Test adding and verifying a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Add file to manifest
        checker.add_file(test_file)
        assert str(test_file) in checker.manifest
        
        # Verify file
        assert checker.verify_file(test_file)
    
    def test_verify_corrupted_file(self, checker, tmp_path):
        """Test verifying a corrupted file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")
        
        # Add file to manifest
        checker.add_file(test_file)
        
        # Modify file
        test_file.write_text("modified content")
        
        # Verification should fail
        assert not checker.verify_file(test_file)
    
    def test_save_and_load_manifest(self, checker, tmp_path):
        """Test saving and loading manifest."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")
        
        # Add file and save manifest
        checker.add_file(test_file)
        checker.save_manifest()
        
        # Create new checker and load manifest
        new_checker = FileIntegrityChecker(manifest_file=checker.manifest_file)
        
        assert str(test_file) in new_checker.manifest
        assert new_checker.verify_file(test_file)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
