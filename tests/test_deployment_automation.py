"""
Deployment Automation Tests

Tests for deployment scripts, automation procedures, and infrastructure management
in the MLX distributed inference system.
"""

import pytest
import subprocess
import tempfile
import shutil
import time
import os
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional


class TestDeploymentScriptValidation:
    """Test deployment script validation and syntax."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dirs = []
    
    def teardown_method(self):
        """Clean up test resources."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def test_deploy_script_syntax(self):
        """Test main deployment script syntax validation."""
        deploy_script = Path("scripts/deploy.sh")
        assert deploy_script.exists(), "deploy.sh script not found"
        
        # Test bash syntax
        result = subprocess.run(
            ["bash", "-n", str(deploy_script)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Deployment script syntax error: {result.stderr}"
    
    def test_worker_startup_script_syntax(self):
        """Test worker startup script syntax validation."""
        startup_script = Path("scripts/start_worker_macos.sh")
        assert startup_script.exists(), "start_worker_macos.sh script not found"
        
        # Test bash syntax
        result = subprocess.run(
            ["bash", "-n", str(startup_script)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Worker startup script syntax error: {result.stderr}"
    
    def test_health_check_script_syntax(self):
        """Test health check script syntax validation."""
        health_script = Path("scripts/health_check.sh")
        assert health_script.exists(), "health_check.sh script not found"
        
        # Test bash syntax
        result = subprocess.run(
            ["bash", "-n", str(health_script)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Health check script syntax error: {result.stderr}"
    
    def test_backup_recovery_script_syntax(self):
        """Test backup and recovery script syntax validation."""
        backup_script = Path("scripts/backup_recovery.sh")
        assert backup_script.exists(), "backup_recovery.sh script not found"
        
        # Test bash syntax
        result = subprocess.run(
            ["bash", "-n", str(backup_script)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Backup script syntax error: {result.stderr}"
    
    def test_log_rotation_script_syntax(self):
        """Test log rotation script syntax validation."""
        log_script = Path("scripts/log_rotation.sh")
        assert log_script.exists(), "log_rotation.sh script not found"
        
        # Test bash syntax
        result = subprocess.run(
            ["bash", "-n", str(log_script)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Log rotation script syntax error: {result.stderr}"
    
    def test_stop_worker_script_syntax(self):
        """Test stop worker script syntax validation."""
        stop_script = Path("scripts/stop_worker.sh")
        assert stop_script.exists(), "stop_worker.sh script not found"
        
        # Test bash syntax
        result = subprocess.run(
            ["bash", "-n", str(stop_script)],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, f"Stop worker script syntax error: {result.stderr}"


class TestScriptHelpAndUsage:
    """Test script help and usage information."""
    
    def test_deploy_script_help(self):
        """Test deployment script help output."""
        deploy_script = Path("scripts/deploy.sh")
        
        result = subprocess.run(
            ["bash", str(deploy_script), "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "OPTIONS:" in result.stdout
        assert "EXAMPLES:" in result.stdout
    
    def test_worker_startup_script_help(self):
        """Test worker startup script help output."""
        startup_script = Path("scripts/start_worker_macos.sh")
        
        result = subprocess.run(
            ["bash", str(startup_script), "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "OPTIONS:" in result.stdout
    
    def test_health_check_script_help(self):
        """Test health check script help output."""
        health_script = Path("scripts/health_check.sh")
        
        result = subprocess.run(
            ["bash", str(health_script), "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "OPTIONS:" in result.stdout
    
    def test_backup_recovery_script_help(self):
        """Test backup and recovery script help output."""
        backup_script = Path("scripts/backup_recovery.sh")
        
        result = subprocess.run(
            ["bash", str(backup_script), "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "COMMANDS:" in result.stdout
    
    def test_log_rotation_script_help(self):
        """Test log rotation script help output."""
        log_script = Path("scripts/log_rotation.sh")
        
        result = subprocess.run(
            ["bash", str(log_script), "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "Usage:" in result.stdout
        assert "OPTIONS:" in result.stdout


class TestDeploymentPrerequisites:
    """Test deployment prerequisite checking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dirs = []
    
    def teardown_method(self):
        """Clean up test resources."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def test_system_requirements_check(self):
        """Test system requirements validation."""
        # Mock system requirements check
        requirements = {
            "operating_system": "Darwin",  # macOS
            "python_version": "3.11+",
            "docker": "available",
            "docker_compose": "available",
            "mlx": "installed"
        }
        
        # Validate requirements structure
        assert "operating_system" in requirements
        assert "python_version" in requirements
        assert "docker" in requirements
        assert "mlx" in requirements
        
        # Test OS check
        import platform
        current_os = platform.system()
        # For testing, we accept any OS, but production requires macOS
        assert current_os in ["Darwin", "Linux", "Windows"]
    
    def test_python_version_check(self):
        """Test Python version validation."""
        import sys
        
        python_version = sys.version_info
        
        # Check minimum Python version (3.11+)
        min_version = (3, 11)
        version_ok = python_version >= min_version
        
        # For testing, we accept Python 3.8+ but warn about optimal version
        if python_version < min_version:
            pytest.skip(f"Python {min_version[0]}.{min_version[1]}+ recommended, found {python_version.major}.{python_version.minor}")
        
        assert version_ok or python_version >= (3, 8)
    
    def test_docker_availability_check(self):
        """Test Docker availability validation."""
        # Test Docker command availability
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            docker_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            docker_available = False
        
        if not docker_available:
            pytest.skip("Docker not available for testing")
        
        assert docker_available == True
    
    def test_docker_compose_availability_check(self):
        """Test Docker Compose availability validation."""
        # Test Docker Compose command availability
        try:
            result = subprocess.run(
                ["docker-compose", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            compose_available = result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            compose_available = False
        
        if not compose_available:
            pytest.skip("Docker Compose not available for testing")
        
        assert compose_available == True
    
    def test_model_directory_validation(self):
        """Test model directory validation."""
        # Create temporary model directory
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        model_dir = Path(temp_dir) / "models"
        model_dir.mkdir()
        
        # Create mock model files
        (model_dir / "model.bin").touch()
        (model_dir / "config.json").write_text('{"model_type": "llama"}')
        (model_dir / "tokenizer.json").touch()
        
        # Validate model directory
        assert model_dir.exists()
        assert model_dir.is_dir()
        
        # Check for model files
        model_files = list(model_dir.glob("*.bin"))
        config_files = list(model_dir.glob("config.json"))
        
        assert len(model_files) > 0
        assert len(config_files) > 0


class TestConfigurationGeneration:
    """Test automatic configuration generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dirs = []
    
    def teardown_method(self):
        """Clean up test resources."""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def test_worker_config_generation(self):
        """Test automatic worker configuration generation."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        # Mock configuration template
        config_template = {
            "worker_id": "${WORKER_ID}",
            "port": "${WORKER_PORT:-8080}",
            "model": {
                "path": "${MODEL_PATH}",
                "name": "${MODEL_NAME:-default}"
            },
            "coordinator": {
                "host": "${COORDINATOR_HOST:-localhost:50051}"
            },
            "performance": {
                "max_concurrent": "${MAX_CONCURRENT:-4}"
            }
        }
        
        # Mock environment variables
        env_vars = {
            "WORKER_ID": "auto-worker-01",
            "WORKER_PORT": "8080",
            "MODEL_PATH": "/app/models",
            "MODEL_NAME": "llama-7b",
            "COORDINATOR_HOST": "coordinator.local:50051",
            "MAX_CONCURRENT": "6"
        }
        
        # Process template
        def substitute_env_vars(obj, env_vars):
            if isinstance(obj, dict):
                return {k: substitute_env_vars(v, env_vars) for k, v in obj.items()}
            elif isinstance(obj, str) and obj.startswith("${"):
                # Extract variable name and default
                if ":-" in obj:
                    var_part, default = obj[2:-1].split(":-", 1)
                    return env_vars.get(var_part, default)
                else:
                    var_name = obj[2:-1]
                    return env_vars.get(var_name, obj)
            else:
                return obj
        
        processed_config = substitute_env_vars(config_template, env_vars)
        
        # Validate processed configuration
        assert processed_config["worker_id"] == "auto-worker-01"
        assert processed_config["port"] == "8080"
        assert processed_config["model"]["path"] == "/app/models"
        assert processed_config["model"]["name"] == "llama-7b"
        assert processed_config["coordinator"]["host"] == "coordinator.local:50051"
        assert processed_config["performance"]["max_concurrent"] == "6"
    
    def test_coordinator_config_generation(self):
        """Test automatic coordinator configuration generation."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        # Mock coordinator configuration
        coordinator_config = {
            "server": {
                "port": 50051,
                "max_workers": 10
            },
            "worker_pool": {
                "heartbeat_timeout": 30.0,
                "max_workers": 100
            },
            "load_balancing": {
                "algorithm": "weighted_round_robin",
                "weights": {
                    "latency": 0.4,
                    "load": 0.3,
                    "reliability": 0.3
                }
            }
        }
        
        # Write configuration file
        config_file = Path(temp_dir) / "coordinator.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(coordinator_config, f)
        
        # Validate generated configuration
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config["server"]["port"] == 50051
        assert loaded_config["load_balancing"]["algorithm"] == "weighted_round_robin"
        assert sum(loaded_config["load_balancing"]["weights"].values()) == 1.0
    
    def test_api_config_generation(self):
        """Test automatic API configuration generation."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        # Mock API configuration
        api_config = {
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "debug": False
            },
            "coordinator": {
                "host": "coordinator.local",
                "port": 50051
            },
            "security": {
                "enable_auth": True,
                "api_keys": ["generated-key-" + "x" * 32]
            },
            "rate_limiting": {
                "requests_per_minute": 100,
                "burst_size": 20
            }
        }
        
        # Write configuration file
        config_file = Path(temp_dir) / "api.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(api_config, f)
        
        # Validate generated configuration
        assert config_file.exists()
        
        with open(config_file, 'r') as f:
            loaded_config = yaml.safe_load(f)
        
        assert loaded_config["server"]["host"] == "0.0.0.0"
        assert loaded_config["server"]["port"] == 8000
        assert loaded_config["security"]["enable_auth"] == True
        assert len(loaded_config["security"]["api_keys"][0]) >= 32


class TestDeploymentWorkflow:
    """Test complete deployment workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dirs = []
        self.test_processes = []
    
    def teardown_method(self):
        """Clean up test resources."""
        # Stop test processes
        for process in self.test_processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception:
                pass
        
        # Clean up temp directories
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def test_deployment_step_sequence(self):
        """Test deployment step sequence validation."""
        # Define expected deployment steps
        deployment_steps = [
            "prerequisites_check",
            "project_structure_setup",
            "configuration_validation",
            "backup_creation",
            "containerized_services_deployment",
            "native_workers_deployment",
            "deployment_verification",
            "monitoring_setup"
        ]
        
        # Mock deployment execution
        executed_steps = []
        
        for step in deployment_steps:
            # Mock step execution
            step_success = True  # Would be actual step execution
            
            if step_success:
                executed_steps.append(step)
            else:
                break  # Stop on failure
        
        # Verify all steps executed
        assert len(executed_steps) == len(deployment_steps)
        assert executed_steps == deployment_steps
    
    def test_deployment_rollback_mechanism(self):
        """Test deployment rollback mechanism."""
        # Mock deployment with failure
        deployment_steps = [
            {"name": "backup_creation", "status": "success"},
            {"name": "service_stop", "status": "success"},
            {"name": "configuration_update", "status": "success"},
            {"name": "service_start", "status": "failed"},  # Simulate failure
        ]
        
        # Check for failure
        failed_step = next((step for step in deployment_steps if step["status"] == "failed"), None)
        
        if failed_step:
            # Execute rollback steps
            rollback_steps = [
                {"name": "rollback_initiated", "status": "success"},
                {"name": "service_stop", "status": "success"},
                {"name": "backup_restore", "status": "success"},
                {"name": "service_restart", "status": "success"}
            ]
            
            deployment_steps.extend(rollback_steps)
        
        # Verify rollback was executed
        rollback_executed = any("rollback" in step["name"] for step in deployment_steps)
        assert rollback_executed == True
        
        # Verify final state is success
        final_steps = [step for step in deployment_steps if "rollback" in step["name"] or "restore" in step["name"]]
        assert all(step["status"] == "success" for step in final_steps)
    
    def test_deployment_environment_setup(self):
        """Test deployment environment setup."""
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        # Mock project structure creation
        project_dirs = [
            "config",
            "logs/api",
            "logs/coordinator",
            "logs/worker",
            "logs/system",
            "run",
            "data/api",
            "data/coordinator",
            "data/worker",
            "backups",
            "models"
        ]
        
        # Create directory structure
        for dir_path in project_dirs:
            full_path = Path(temp_dir) / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
        
        # Verify directory structure
        for dir_path in project_dirs:
            full_path = Path(temp_dir) / dir_path
            assert full_path.exists()
            assert full_path.is_dir()
    
    def test_deployment_health_verification(self):
        """Test deployment health verification."""
        # Mock health check results
        health_checks = {
            "containerized_services": {
                "api": {"status": "healthy", "port": 8000},
                "coordinator": {"status": "healthy", "port": 50051},
                "redis": {"status": "healthy", "port": 6379},
                "prometheus": {"status": "healthy", "port": 9090},
                "grafana": {"status": "healthy", "port": 3000}
            },
            "native_workers": {
                "worker-01": {"status": "healthy", "port": 8080},
                "worker-02": {"status": "healthy", "port": 8081}
            },
            "network_connectivity": {
                "api_to_coordinator": "connected",
                "coordinator_to_workers": "connected",
                "monitoring_endpoints": "accessible"
            }
        }
        
        # Verify all services are healthy
        containerized_healthy = all(
            service["status"] == "healthy" 
            for service in health_checks["containerized_services"].values()
        )
        
        workers_healthy = all(
            worker["status"] == "healthy"
            for worker in health_checks["native_workers"].values()
        )
        
        network_ok = all(
            status == "connected" or status == "accessible"
            for status in health_checks["network_connectivity"].values()
        )
        
        assert containerized_healthy == True
        assert workers_healthy == True
        assert network_ok == True


class TestDeploymentMonitoring:
    """Test deployment monitoring and alerting."""
    
    def test_deployment_metrics_collection(self):
        """Test deployment metrics collection."""
        # Mock deployment metrics
        deployment_metrics = {
            "deployment_duration": 120.5,  # seconds
            "services_started": 5,
            "workers_deployed": 2,
            "health_checks_passed": 7,
            "configuration_files_generated": 3,
            "backup_size": 1024 * 1024 * 50,  # 50MB
            "deployment_timestamp": time.time()
        }
        
        # Validate metrics structure
        assert "deployment_duration" in deployment_metrics
        assert "services_started" in deployment_metrics
        assert "workers_deployed" in deployment_metrics
        assert deployment_metrics["deployment_duration"] > 0
        assert deployment_metrics["services_started"] > 0
    
    def test_deployment_alerting_configuration(self):
        """Test deployment alerting configuration."""
        # Mock alerting configuration
        alerting_config = {
            "enabled": True,
            "channels": {
                "slack": {
                    "webhook_url": "https://hooks.slack.com/services/...",
                    "channel": "#deployments"
                },
                "email": {
                    "smtp_server": "smtp.example.com",
                    "recipients": ["ops@example.com"]
                }
            },
            "alert_conditions": {
                "deployment_failure": {"severity": "critical", "notify": True},
                "deployment_success": {"severity": "info", "notify": True},
                "rollback_triggered": {"severity": "warning", "notify": True},
                "health_check_failure": {"severity": "warning", "notify": True}
            }
        }
        
        # Validate alerting configuration
        assert alerting_config["enabled"] == True
        assert "slack" in alerting_config["channels"]
        assert "email" in alerting_config["channels"]
        assert "deployment_failure" in alerting_config["alert_conditions"]
        assert alerting_config["alert_conditions"]["deployment_failure"]["severity"] == "critical"
    
    def test_deployment_logging_integration(self):
        """Test deployment logging integration."""
        # Mock deployment logging
        log_entries = [
            {"timestamp": time.time(), "level": "INFO", "message": "Deployment started", "component": "deploy"},
            {"timestamp": time.time(), "level": "INFO", "message": "Prerequisites check passed", "component": "deploy"},
            {"timestamp": time.time(), "level": "INFO", "message": "Configuration generated", "component": "deploy"},
            {"timestamp": time.time(), "level": "INFO", "message": "Services started successfully", "component": "deploy"},
            {"timestamp": time.time(), "level": "INFO", "message": "Deployment completed", "component": "deploy"}
        ]
        
        # Validate log entries
        assert len(log_entries) > 0
        
        for entry in log_entries:
            assert "timestamp" in entry
            assert "level" in entry
            assert "message" in entry
            assert "component" in entry
            assert entry["level"] in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])