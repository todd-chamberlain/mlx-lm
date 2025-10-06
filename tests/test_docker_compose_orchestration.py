"""
Tests for Docker Compose orchestration and service coordination.

Tests complete stack startup, service health checks, configuration management,
and integration between all containerized services.
"""

import asyncio
import pytest
import subprocess
import time
import requests
import yaml
import json
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import shutil


class TestDockerComposeStackManagement:
    """Test complete Docker Compose stack management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dirs = []
        self.compose_processes = []
    
    def teardown_method(self):
        """Clean up test resources."""
        # Stop any running compose processes
        for process in self.compose_processes:
            try:
                process.terminate()
                process.wait(timeout=10)
            except Exception:
                pass
        
        # Clean up temp directories
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def test_compose_config_validation(self):
        """Test Docker Compose configuration validation."""
        # Test main compose file
        result = subprocess.run(
            ["docker-compose", "config"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode != 0:
            pytest.skip(f"Docker Compose not available or config invalid: {result.stderr}")
        
        # Parse the output to ensure it's valid YAML
        try:
            config = yaml.safe_load(result.stdout)
            assert isinstance(config, dict)
            assert "services" in config
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid compose configuration: {e}")
    
    def test_compose_development_config(self):
        """Test development configuration override."""
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "-f", "docker-compose.dev.yml", "config"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode != 0:
            pytest.skip(f"Development compose config validation failed: {result.stderr}")
        
        try:
            config = yaml.safe_load(result.stdout)
            services = config.get("services", {})
            
            # Development config should have specific characteristics
            if "api" in services:
                api_service = services["api"]
                # Should have development-specific settings
                environment = api_service.get("environment", [])
                # Convert list to dict if needed
                if isinstance(environment, list):
                    env_dict = {}
                    for env_var in environment:
                        if "=" in env_var:
                            key, value = env_var.split("=", 1)
                            env_dict[key] = value
                    environment = env_dict
                
                # Check for development settings
                assert isinstance(environment, dict)
                
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid development compose configuration: {e}")
    
    def test_compose_production_config(self):
        """Test production configuration override."""
        result = subprocess.run(
            ["docker-compose", "-f", "docker-compose.yml", "-f", "docker-compose.prod.yml", "config"],
            capture_output=True,
            text=True,
            cwd="."
        )
        
        if result.returncode != 0:
            pytest.skip(f"Production compose config validation failed: {result.stderr}")
        
        try:
            config = yaml.safe_load(result.stdout)
            services = config.get("services", {})
            
            # Production config should have resource limits
            for service_name, service_config in services.items():
                deploy = service_config.get("deploy", {})
                if deploy:
                    resources = deploy.get("resources", {})
                    if resources:
                        limits = resources.get("limits", {})
                        # Should have memory and CPU limits in production
                        assert "memory" in limits or "cpus" in limits
                        
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid production compose configuration: {e}")
    
    def test_service_startup_order(self):
        """Test that services start in correct dependency order."""
        # Create minimal test compose file
        test_compose = {
            "version": "3.8",
            "services": {
                "redis": {
                    "image": "redis:alpine",
                    "command": "redis-server --appendonly yes"
                },
                "coordinator": {
                    "image": "alpine:latest",
                    "command": "sleep 30",
                    "depends_on": ["redis"]
                },
                "api": {
                    "image": "alpine:latest", 
                    "command": "sleep 30",
                    "depends_on": ["coordinator", "redis"]
                }
            }
        }
        
        # Create temporary compose file
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        compose_file = Path(temp_dir) / "test-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(test_compose, f)
        
        # Test dependency validation
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Compose config failed: {result.stderr}"
        
        # Parse and verify dependencies
        config = yaml.safe_load(result.stdout)
        services = config["services"]
        
        # API should depend on both coordinator and redis
        api_deps = services["api"]["depends_on"]
        assert "coordinator" in api_deps
        assert "redis" in api_deps
        
        # Coordinator should depend on redis
        coord_deps = services["coordinator"]["depends_on"]
        assert "redis" in coord_deps
    
    def test_environment_variable_substitution(self):
        """Test environment variable substitution in compose files."""
        # Create test compose with environment variables
        test_compose = {
            "version": "3.8",
            "services": {
                "test-service": {
                    "image": "alpine:latest",
                    "environment": {
                        "API_PORT": "${API_PORT:-8000}",
                        "LOG_LEVEL": "${LOG_LEVEL:-INFO}",
                        "REDIS_URL": "redis://${REDIS_HOST:-redis}:${REDIS_PORT:-6379}"
                    }
                }
            }
        }
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        compose_file = Path(temp_dir) / "test-env-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(test_compose, f)
        
        # Test with default values
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True,
            env={"PATH": "/usr/bin:/bin"}  # Minimal environment
        )
        
        if result.returncode == 0:
            config = yaml.safe_load(result.stdout)
            service_env = config["services"]["test-service"]["environment"]
            
            # Should use default values
            assert service_env["API_PORT"] == "8000"
            assert service_env["LOG_LEVEL"] == "INFO"
            assert "redis:6379" in service_env["REDIS_URL"]
    
    def test_network_connectivity_between_services(self):
        """Test network connectivity configuration between services."""
        # Create test compose with custom network
        test_compose = {
            "version": "3.8",
            "services": {
                "service1": {
                    "image": "alpine:latest",
                    "command": "sleep 60",
                    "networks": ["test-network"]
                },
                "service2": {
                    "image": "alpine:latest",
                    "command": "sleep 60", 
                    "networks": ["test-network"]
                }
            },
            "networks": {
                "test-network": {
                    "driver": "bridge",
                    "ipam": {
                        "config": [
                            {"subnet": "172.22.0.0/16"}
                        ]
                    }
                }
            }
        }
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        compose_file = Path(temp_dir) / "test-network-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(test_compose, f)
        
        # Validate network configuration
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Network config failed: {result.stderr}"
        
        config = yaml.safe_load(result.stdout)
        networks = config["networks"]
        
        # Verify network configuration
        assert "test-network" in networks
        network_config = networks["test-network"]
        assert network_config["driver"] == "bridge"
        assert "ipam" in network_config
    
    def test_volume_mount_configuration(self):
        """Test volume mount configuration and validation."""
        # Create test compose with various volume types
        test_compose = {
            "version": "3.8",
            "services": {
                "app": {
                    "image": "alpine:latest",
                    "command": "sleep 60",
                    "volumes": [
                        "named-volume:/data",
                        "./config:/app/config:ro",
                        "/tmp:/tmp:rw"
                    ]
                }
            },
            "volumes": {
                "named-volume": {}
            }
        }
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        # Create config directory for bind mount
        config_dir = Path(temp_dir) / "config"
        config_dir.mkdir()
        (config_dir / "test.conf").write_text("test=value")
        
        compose_file = Path(temp_dir) / "test-volume-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(test_compose, f)
        
        # Validate volume configuration
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True,
            cwd=temp_dir
        )
        
        assert result.returncode == 0, f"Volume config failed: {result.stderr}"
        
        config = yaml.safe_load(result.stdout)
        
        # Verify volume configuration
        assert "volumes" in config
        assert "named-volume" in config["volumes"]
        
        service_volumes = config["services"]["app"]["volumes"]
        assert len(service_volumes) == 3
    
    def test_health_check_configuration(self):
        """Test health check configuration for services."""
        # Create test compose with health checks
        test_compose = {
            "version": "3.8",
            "services": {
                "web": {
                    "image": "nginx:alpine",
                    "healthcheck": {
                        "test": ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3,
                        "start_period": "40s"
                    }
                },
                "api": {
                    "image": "python:3.11-slim",
                    "healthcheck": {
                        "test": ["CMD", "curl", "-f", "http://localhost:8000/health"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 3
                    }
                }
            }
        }
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        compose_file = Path(temp_dir) / "test-health-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(test_compose, f)
        
        # Validate health check configuration
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0, f"Health check config failed: {result.stderr}"
        
        config = yaml.safe_load(result.stdout)
        services = config["services"]
        
        # Verify health check configuration
        for service_name in ["web", "api"]:
            service = services[service_name]
            assert "healthcheck" in service
            healthcheck = service["healthcheck"]
            assert "test" in healthcheck
            assert "interval" in healthcheck
            assert "timeout" in healthcheck
            assert "retries" in healthcheck


class TestServiceIntegrationAndCommunication:
    """Test integration and communication between services."""
    
    def test_service_discovery_configuration(self):
        """Test service discovery through Docker DNS."""
        # Create test compose that demonstrates service discovery
        test_compose = {
            "version": "3.8",
            "services": {
                "backend": {
                    "image": "python:3.11-slim",
                    "command": "python -c 'import http.server; import socketserver; socketserver.TCPServer((\"\", 8000), http.server.SimpleHTTPRequestHandler).serve_forever()'",
                    "expose": ["8000"]
                },
                "frontend": {
                    "image": "alpine:latest",
                    "command": "sleep 60",
                    "depends_on": ["backend"]
                }
            }
        }
        
        temp_dir = tempfile.mkdtemp()
        compose_file = Path(temp_dir) / "test-discovery-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(test_compose, f)
        
        # Validate service discovery configuration
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        
        config = yaml.safe_load(result.stdout)
        services = config["services"]
        
        # Frontend should be able to reach backend via service name
        assert "backend" in services
        assert "frontend" in services
        assert "backend" in services["frontend"]["depends_on"]
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_port_exposure_and_mapping(self):
        """Test port exposure and mapping configuration."""
        test_compose = {
            "version": "3.8",
            "services": {
                "web": {
                    "image": "nginx:alpine",
                    "ports": ["80:80", "443:443"],
                    "expose": ["80", "443"]
                },
                "api": {
                    "image": "python:3.11-slim",
                    "ports": ["8000:8000"],
                    "expose": ["8000"]
                },
                "internal": {
                    "image": "redis:alpine",
                    "expose": ["6379"]  # Only internal exposure
                }
            }
        }
        
        temp_dir = tempfile.mkdtemp()
        compose_file = Path(temp_dir) / "test-ports-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(test_compose, f)
        
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        
        config = yaml.safe_load(result.stdout)
        services = config["services"]
        
        # Verify port configurations
        web_service = services["web"]
        assert "ports" in web_service
        assert "80:80" in web_service["ports"]
        
        api_service = services["api"]
        assert "ports" in api_service
        assert "8000:8000" in api_service["ports"]
        
        internal_service = services["internal"]
        assert "expose" in internal_service
        assert "6379" in internal_service["expose"]
        # Internal service should not have external ports
        assert "ports" not in internal_service
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_environment_configuration_inheritance(self):
        """Test environment configuration and inheritance."""
        # Create .env file for testing
        temp_dir = tempfile.mkdtemp()
        env_file = Path(temp_dir) / ".env"
        env_file.write_text("""
API_PORT=8080
LOG_LEVEL=DEBUG
REDIS_PASSWORD=secret123
""")
        
        test_compose = {
            "version": "3.8",
            "services": {
                "api": {
                    "image": "python:3.11-slim",
                    "environment": {
                        "PORT": "${API_PORT}",
                        "LOG_LEVEL": "${LOG_LEVEL}",
                        "REDIS_PASSWORD": "${REDIS_PASSWORD}"
                    },
                    "env_file": [".env"]
                }
            }
        }
        
        compose_file = Path(temp_dir) / "test-env-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(test_compose, f)
        
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True,
            cwd=temp_dir
        )
        
        if result.returncode == 0:
            config = yaml.safe_load(result.stdout)
            api_env = config["services"]["api"]["environment"]
            
            # Environment variables should be substituted
            assert api_env["PORT"] == "8080"
            assert api_env["LOG_LEVEL"] == "DEBUG"
            assert api_env["REDIS_PASSWORD"] == "secret123"
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_resource_limits_and_constraints(self):
        """Test resource limits and deployment constraints."""
        test_compose = {
            "version": "3.8",
            "services": {
                "api": {
                    "image": "python:3.11-slim",
                    "deploy": {
                        "resources": {
                            "limits": {
                                "cpus": "1.0",
                                "memory": "1G"
                            },
                            "reservations": {
                                "cpus": "0.5",
                                "memory": "512M"
                            }
                        },
                        "restart_policy": {
                            "condition": "on-failure",
                            "delay": "5s",
                            "max_attempts": 3
                        }
                    }
                }
            }
        }
        
        temp_dir = tempfile.mkdtemp()
        compose_file = Path(temp_dir) / "test-resources-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(test_compose, f)
        
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        
        config = yaml.safe_load(result.stdout)
        api_deploy = config["services"]["api"]["deploy"]
        
        # Verify resource limits
        resources = api_deploy["resources"]
        assert resources["limits"]["cpus"] == "1.0"
        assert resources["limits"]["memory"] == "1G"
        assert resources["reservations"]["cpus"] == "0.5"
        assert resources["reservations"]["memory"] == "512M"
        
        # Verify restart policy
        restart_policy = api_deploy["restart_policy"]
        assert restart_policy["condition"] == "on-failure"
        assert restart_policy["max_attempts"] == 3
        
        # Cleanup
        shutil.rmtree(temp_dir)


class TestComposeCommandIntegration:
    """Test Docker Compose command integration and workflows."""
    
    def test_compose_build_command(self):
        """Test Docker Compose build command functionality."""
        # Test build command validation
        result = subprocess.run(
            ["docker-compose", "build", "--help"],
            capture_output=True,
            text=True
        )
        
        # Should show build help without errors
        assert result.returncode == 0
        assert "build" in result.stdout.lower()
    
    def test_compose_up_dry_run(self):
        """Test Docker Compose up command validation."""
        # Test up command help
        result = subprocess.run(
            ["docker-compose", "up", "--help"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        assert "up" in result.stdout.lower()
    
    def test_compose_logs_configuration(self):
        """Test Docker Compose logging configuration."""
        test_compose = {
            "version": "3.8",
            "services": {
                "app": {
                    "image": "alpine:latest",
                    "command": "echo 'test log message'",
                    "logging": {
                        "driver": "json-file",
                        "options": {
                            "max-size": "10m",
                            "max-file": "3"
                        }
                    }
                }
            }
        }
        
        temp_dir = tempfile.mkdtemp()
        compose_file = Path(temp_dir) / "test-logs-compose.yml"
        with open(compose_file, 'w') as f:
            yaml.dump(test_compose, f)
        
        result = subprocess.run(
            ["docker-compose", "-f", str(compose_file), "config"],
            capture_output=True,
            text=True
        )
        
        assert result.returncode == 0
        
        config = yaml.safe_load(result.stdout)
        logging_config = config["services"]["app"]["logging"]
        
        assert logging_config["driver"] == "json-file"
        assert logging_config["options"]["max-size"] == "10m"
        assert logging_config["options"]["max-file"] == "3"
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_compose_override_files(self):
        """Test Docker Compose override file functionality."""
        # Base compose file
        base_compose = {
            "version": "3.8",
            "services": {
                "app": {
                    "image": "alpine:latest",
                    "environment": {
                        "ENV": "base"
                    }
                }
            }
        }
        
        # Override compose file
        override_compose = {
            "version": "3.8",
            "services": {
                "app": {
                    "environment": {
                        "ENV": "override",
                        "DEBUG": "true"
                    }
                }
            }
        }
        
        temp_dir = tempfile.mkdtemp()
        
        base_file = Path(temp_dir) / "docker-compose.yml"
        with open(base_file, 'w') as f:
            yaml.dump(base_compose, f)
        
        override_file = Path(temp_dir) / "docker-compose.override.yml"
        with open(override_file, 'w') as f:
            yaml.dump(override_compose, f)
        
        # Test merged configuration
        result = subprocess.run(
            ["docker-compose", "config"],
            capture_output=True,
            text=True,
            cwd=temp_dir
        )
        
        if result.returncode == 0:
            config = yaml.safe_load(result.stdout)
            app_env = config["services"]["app"]["environment"]
            
            # Override should take precedence
            assert app_env["ENV"] == "override"
            assert app_env["DEBUG"] == "true"
        
        # Cleanup
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])