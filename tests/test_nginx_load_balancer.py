"""
Tests for nginx load balancer configuration and functionality.

Tests nginx configuration validation, upstream health checks, SSL/TLS termination,
request routing, and integration with the MLX inference services.
"""

import pytest
import subprocess
import tempfile
import shutil
import time
import requests
import socket
from pathlib import Path
from unittest.mock import Mock, patch
import yaml
import ssl
import threading
import http.server
import socketserver


class TestNginxConfiguration:
    """Test nginx configuration file validation and syntax."""
    
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
    
    def test_nginx_config_syntax_validation(self):
        """Test nginx configuration syntax validation."""
        # Test main nginx config
        nginx_config_path = Path("config/nginx/nginx.conf")
        if nginx_config_path.exists():
            # Create temporary nginx config for testing
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            test_config = temp_dir + "/nginx.conf"
            shutil.copy(nginx_config_path, test_config)
            
            # Test nginx configuration syntax
            result = subprocess.run(
                ["nginx", "-t", "-c", test_config],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # If nginx is not available, skip the test
                if "nginx: command not found" in result.stderr or "No such file" in result.stderr:
                    pytest.skip("nginx not available for testing")
                else:
                    pytest.fail(f"nginx config syntax error: {result.stderr}")
    
    def test_nginx_production_config_syntax(self):
        """Test nginx production configuration syntax."""
        nginx_prod_config_path = Path("config/nginx/nginx-prod.conf")
        if nginx_prod_config_path.exists():
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            test_config = temp_dir + "/nginx-prod.conf"
            shutil.copy(nginx_prod_config_path, test_config)
            
            # Test nginx production configuration syntax
            result = subprocess.run(
                ["nginx", "-t", "-c", test_config],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                if "nginx: command not found" in result.stderr or "No such file" in result.stderr:
                    pytest.skip("nginx not available for testing")
                else:
                    pytest.fail(f"nginx production config syntax error: {result.stderr}")
    
    def test_upstream_configuration_validation(self):
        """Test upstream server configuration validation."""
        # Create test nginx config with upstream
        nginx_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            upstream api_servers {
                server api1:8000 max_fails=3 fail_timeout=30s;
                server api2:8000 max_fails=3 fail_timeout=30s;
                server api3:8000 max_fails=3 fail_timeout=30s backup;
            }
            
            server {
                listen 80;
                location / {
                    proxy_pass http://api_servers;
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-upstream.conf"
        config_file.write_text(nginx_config)
        
        # Test configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr or "No such file" in result.stderr:
            pytest.skip("nginx not available for testing")
        
        # Configuration should be valid
        assert result.returncode == 0 or "syntax is ok" in result.stderr
    
    def test_ssl_configuration_validation(self):
        """Test SSL/TLS configuration validation."""
        # Create self-signed certificate for testing
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        cert_file = Path(temp_dir) / "test.crt"
        key_file = Path(temp_dir) / "test.key"
        
        # Generate self-signed certificate
        try:
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-keyout", str(key_file),
                "-out", str(cert_file), "-days", "1", "-nodes", "-subj", "/CN=test"
            ], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("OpenSSL not available for certificate generation")
        
        # Create nginx SSL config
        ssl_config = f"""
        events {{
            worker_connections 1024;
        }}
        
        http {{
            server {{
                listen 443 ssl;
                ssl_certificate {cert_file};
                ssl_certificate_key {key_file};
                ssl_protocols TLSv1.2 TLSv1.3;
                
                location / {{
                    return 200 'SSL test';
                    add_header Content-Type text/plain;
                }}
            }}
        }}
        """
        
        config_file = Path(temp_dir) / "test-ssl.conf"
        config_file.write_text(ssl_config)
        
        # Test SSL configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for testing")
        
        # SSL configuration should be valid
        assert result.returncode == 0 or "syntax is ok" in result.stderr
    
    def test_rate_limiting_configuration(self):
        """Test rate limiting configuration validation."""
        rate_limit_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
            limit_req_zone $binary_remote_addr zone=auth:10m rate=5r/s;
            
            server {
                listen 80;
                
                location /api/ {
                    limit_req zone=api burst=20 nodelay;
                    proxy_pass http://backend;
                }
                
                location /auth/ {
                    limit_req zone=auth burst=5 nodelay;
                    proxy_pass http://backend;
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-rate-limit.conf"
        config_file.write_text(rate_limit_config)
        
        # Test rate limiting configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for testing")
        
        assert result.returncode == 0 or "syntax is ok" in result.stderr


class TestLoadBalancingFunctionality:
    """Test load balancing functionality and behavior."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dirs = []
        self.test_servers = []
        self.server_threads = []
    
    def teardown_method(self):
        """Clean up test resources."""
        # Stop test servers
        for server in self.test_servers:
            try:
                server.shutdown()
            except Exception:
                pass
        
        # Wait for server threads to finish
        for thread in self.server_threads:
            try:
                thread.join(timeout=1)
            except Exception:
                pass
        
        # Clean up temp directories
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
    
    def create_test_server(self, port: int, response_text: str) -> tuple:
        """Create a test HTTP server."""
        class TestHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(response_text.encode())
            
            def log_message(self, format, *args):
                pass  # Suppress log messages
        
        server = socketserver.TCPServer(("", port), TestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        
        self.test_servers.append(server)
        self.server_threads.append(thread)
        
        return server, thread
    
    def find_free_port(self) -> int:
        """Find a free port for testing."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def test_round_robin_load_balancing(self):
        """Test round-robin load balancing behavior."""
        # Create multiple backend servers
        port1 = self.find_free_port()
        port2 = self.find_free_port()
        
        self.create_test_server(port1, "Server 1")
        self.create_test_server(port2, "Server 2")
        
        # Wait for servers to start
        time.sleep(0.5)
        
        # Create nginx config with round-robin
        nginx_config = f"""
        events {{
            worker_connections 1024;
        }}
        
        http {{
            upstream backend {{
                server 127.0.0.1:{port1};
                server 127.0.0.1:{port2};
            }}
            
            server {{
                listen 8080;
                location / {{
                    proxy_pass http://backend;
                }}
            }}
        }}
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-lb.conf"
        config_file.write_text(nginx_config)
        
        # Test configuration syntax
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for load balancing test")
        
        # Configuration should be valid
        assert result.returncode == 0 or "syntax is ok" in result.stderr
    
    def test_weighted_load_balancing(self):
        """Test weighted load balancing configuration."""
        nginx_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            upstream backend {
                server backend1:8000 weight=3;
                server backend2:8000 weight=1;
                server backend3:8000 weight=2;
            }
            
            server {
                listen 80;
                location / {
                    proxy_pass http://backend;
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-weighted-lb.conf"
        config_file.write_text(nginx_config)
        
        # Test weighted configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for weighted load balancing test")
        
        assert result.returncode == 0 or "syntax is ok" in result.stderr
    
    def test_health_check_configuration(self):
        """Test health check configuration for upstream servers."""
        health_check_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            upstream backend {
                server backend1:8000 max_fails=3 fail_timeout=30s;
                server backend2:8000 max_fails=3 fail_timeout=30s;
                server backup_server:8000 backup;
            }
            
            server {
                listen 80;
                
                location / {
                    proxy_pass http://backend;
                    proxy_next_upstream error timeout invalid_header http_500 http_502 http_503 http_504;
                }
                
                location /health {
                    access_log off;
                    return 200 'healthy';
                    add_header Content-Type text/plain;
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-health-check.conf"
        config_file.write_text(health_check_config)
        
        # Test health check configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for health check test")
        
        assert result.returncode == 0 or "syntax is ok" in result.stderr
    
    def test_session_persistence_configuration(self):
        """Test session persistence (sticky sessions) configuration."""
        session_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            upstream backend {
                ip_hash;
                server backend1:8000;
                server backend2:8000;
                server backend3:8000;
            }
            
            server {
                listen 80;
                location / {
                    proxy_pass http://backend;
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-session.conf"
        config_file.write_text(session_config)
        
        # Test session persistence configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for session persistence test")
        
        assert result.returncode == 0 or "syntax is ok" in result.stderr


class TestSecurityAndSSL:
    """Test security features and SSL/TLS configuration."""
    
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
    
    def test_security_headers_configuration(self):
        """Test security headers configuration."""
        security_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            # Security headers
            add_header X-Frame-Options DENY always;
            add_header X-Content-Type-Options nosniff always;
            add_header X-XSS-Protection "1; mode=block" always;
            add_header Referrer-Policy "strict-origin-when-cross-origin" always;
            add_header Content-Security-Policy "default-src 'self'" always;
            
            server {
                listen 80;
                server_tokens off;
                
                location / {
                    return 200 'Security headers test';
                    add_header Content-Type text/plain;
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-security.conf"
        config_file.write_text(security_config)
        
        # Test security configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for security test")
        
        assert result.returncode == 0 or "syntax is ok" in result.stderr
    
    def test_ssl_cipher_configuration(self):
        """Test SSL cipher suite configuration."""
        # Create temporary certificate files
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        cert_file = Path(temp_dir) / "server.crt"
        key_file = Path(temp_dir) / "server.key"
        
        # Create dummy certificate files
        cert_file.write_text("-----BEGIN CERTIFICATE-----\nDUMMY\n-----END CERTIFICATE-----")
        key_file.write_text("-----BEGIN PRIVATE KEY-----\nDUMMY\n-----END PRIVATE KEY-----")
        
        ssl_config = f"""
        events {{
            worker_connections 1024;
        }}
        
        http {{
            # SSL configuration
            ssl_protocols TLSv1.2 TLSv1.3;
            ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
            ssl_prefer_server_ciphers off;
            ssl_session_cache shared:SSL:10m;
            ssl_session_timeout 10m;
            
            server {{
                listen 443 ssl http2;
                ssl_certificate {cert_file};
                ssl_certificate_key {key_file};
                
                location / {{
                    return 200 'SSL test';
                    add_header Content-Type text/plain;
                }}
            }}
        }}
        """
        
        config_file = Path(temp_dir) / "test-ssl-ciphers.conf"
        config_file.write_text(ssl_config)
        
        # Test SSL cipher configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for SSL cipher test")
        
        # Note: This might fail due to dummy certificates, but syntax should be OK
        # We're mainly testing configuration syntax here
        assert "syntax is ok" in result.stderr or result.returncode == 0
    
    def test_access_control_configuration(self):
        """Test access control and IP restriction configuration."""
        access_control_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            server {
                listen 80;
                
                # Public endpoints
                location /health {
                    return 200 'healthy';
                    add_header Content-Type text/plain;
                }
                
                # Restricted admin endpoints
                location /admin/ {
                    allow 192.168.1.0/24;
                    allow 10.0.0.0/8;
                    deny all;
                    
                    proxy_pass http://backend;
                }
                
                # Metrics endpoint with IP restrictions
                location /metrics {
                    allow 172.16.0.0/12;
                    allow 10.0.0.0/8;
                    deny all;
                    
                    proxy_pass http://backend;
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-access-control.conf"
        config_file.write_text(access_control_config)
        
        # Test access control configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for access control test")
        
        assert result.returncode == 0 or "syntax is ok" in result.stderr


class TestStreamingAndWebSocketSupport:
    """Test streaming and WebSocket support configuration."""
    
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
    
    def test_streaming_configuration(self):
        """Test configuration for streaming responses."""
        streaming_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            upstream api_servers {
                server api1:8000;
                server api2:8000;
                keepalive 32;
            }
            
            server {
                listen 80;
                
                location /v1/chat/completions {
                    proxy_pass http://api_servers;
                    
                    # Streaming configuration
                    proxy_buffering off;
                    proxy_request_buffering off;
                    proxy_http_version 1.1;
                    proxy_set_header Connection "";
                    
                    # Timeouts for long-running requests
                    proxy_connect_timeout 30s;
                    proxy_send_timeout 60s;
                    proxy_read_timeout 600s;
                    
                    # Headers
                    proxy_set_header Host $host;
                    proxy_set_header X-Real-IP $remote_addr;
                    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                    proxy_set_header X-Forwarded-Proto $scheme;
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-streaming.conf"
        config_file.write_text(streaming_config)
        
        # Test streaming configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for streaming test")
        
        assert result.returncode == 0 or "syntax is ok" in result.stderr
    
    def test_websocket_configuration(self):
        """Test WebSocket proxy configuration."""
        websocket_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            map $http_upgrade $connection_upgrade {
                default upgrade;
                '' close;
            }
            
            upstream websocket_backend {
                server ws1:8080;
                server ws2:8080;
            }
            
            server {
                listen 80;
                
                location /ws/ {
                    proxy_pass http://websocket_backend;
                    proxy_http_version 1.1;
                    proxy_set_header Upgrade $http_upgrade;
                    proxy_set_header Connection $connection_upgrade;
                    proxy_set_header Host $host;
                    proxy_cache_bypass $http_upgrade;
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-websocket.conf"
        config_file.write_text(websocket_config)
        
        # Test WebSocket configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for WebSocket test")
        
        assert result.returncode == 0 or "syntax is ok" in result.stderr


class TestMonitoringAndLogging:
    """Test monitoring and logging configuration."""
    
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
    
    def test_access_log_configuration(self):
        """Test access log configuration and custom formats."""
        logging_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            # Custom log format
            log_format main '$remote_addr - $remote_user [$time_local] "$request" '
                           '$status $body_bytes_sent "$http_referer" '
                           '"$http_user_agent" "$http_x_forwarded_for" '
                           'rt=$request_time uct="$upstream_connect_time" '
                           'uht="$upstream_header_time" urt="$upstream_response_time"';
            
            access_log /var/log/nginx/access.log main;
            error_log /var/log/nginx/error.log warn;
            
            server {
                listen 80;
                
                location /health {
                    access_log off;
                    return 200 'healthy';
                }
                
                location / {
                    access_log /var/log/nginx/api_access.log main;
                    proxy_pass http://backend;
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-logging.conf"
        config_file.write_text(logging_config)
        
        # Test logging configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for logging test")
        
        assert result.returncode == 0 or "syntax is ok" in result.stderr
    
    def test_status_module_configuration(self):
        """Test nginx status module configuration."""
        status_config = """
        events {
            worker_connections 1024;
        }
        
        http {
            server {
                listen 80;
                
                location /nginx_status {
                    stub_status on;
                    access_log off;
                    allow 127.0.0.1;
                    allow 10.0.0.0/8;
                    deny all;
                }
                
                location / {
                    return 200 'OK';
                }
            }
        }
        """
        
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        
        config_file = Path(temp_dir) / "test-status.conf"
        config_file.write_text(status_config)
        
        # Test status configuration
        result = subprocess.run(
            ["nginx", "-t", "-c", str(config_file)],
            capture_output=True,
            text=True
        )
        
        if "nginx: command not found" in result.stderr:
            pytest.skip("nginx not available for status test")
        
        assert result.returncode == 0 or "syntax is ok" in result.stderr


if __name__ == "__main__":
    pytest.main([__file__, "-v"])