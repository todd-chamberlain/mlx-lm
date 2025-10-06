# Docker Deployment Guide

This guide covers deploying the MLX Distributed Inference system using Docker Compose.

## Quick Start

### Development Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your configuration
nano .env

# Start development stack
make docker-up

# Or manually:
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

### Production Environment

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with production values

# Build and start production stack
make docker-up-prod

# Or manually:
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## Architecture

The system consists of the following containerized services:

### Core Services
- **API Server** (`api`): FastAPI server providing OpenAI-compatible endpoints
- **Coordinator** (`coordinator`): gRPC service managing worker pool and request routing
- **Redis** (`redis`): State persistence and caching
- **Nginx** (`nginx`): Load balancer and reverse proxy

### Monitoring Stack
- **Monitoring** (`monitoring`): Combined Prometheus + Grafana container
- **Jaeger** (`jaeger`): Distributed tracing

### Networks
- **mlx-network**: Bridge network (172.20.0.0/16) for service communication

### Volumes
- **redis_data**: Redis data persistence
- **prometheus_data**: Prometheus metrics storage
- **grafana_data**: Grafana dashboards and configuration

## Configuration

### Environment Variables

Key environment variables (see `.env.example`):

```bash
# Security
API_KEYS=key1,key2,key3
GRAFANA_PASSWORD=secure-password
REDIS_PASSWORD=secure-password

# Performance
MAX_WORKERS=500
WORKER_TIMEOUT=60
RATE_LIMIT_REQUESTS_PER_MINUTE=1000

# External Services
ELASTICSEARCH_URL=http://elasticsearch:9200
```

### Service Configuration

Each service has dedicated configuration files in `config/`:

- `config/redis/redis.conf` - Redis development config
- `config/redis/redis-prod.conf` - Redis production config
- `config/nginx/nginx.conf` - Nginx development config
- `config/nginx/nginx-prod.conf` - Nginx production config with SSL
- `config/prometheus/prometheus.yml` - Prometheus scraping config
- `config/grafana/` - Grafana dashboards and provisioning

## Deployment Modes

### Development (`docker-compose.dev.yml`)
- Exposed ports for direct access
- Debug logging enabled
- Source code mounted for hot reload
- Anonymous Grafana access
- Single API worker for debugging

### Production (`docker-compose.prod.yml`)
- Resource limits and reservations
- Multiple replicas for HA
- SSL/TLS termination
- Authentication enabled
- Optimized logging and retention

## SSL/TLS Setup

For production deployment with SSL:

1. Generate or obtain SSL certificates:
```bash
# Self-signed (development only)
mkdir -p config/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout config/nginx/ssl/nginx.key \
  -out config/nginx/ssl/nginx.crt

# Or copy your certificates
cp your-cert.crt config/nginx/ssl/nginx.crt
cp your-key.key config/nginx/ssl/nginx.key
```

2. Update nginx configuration to use SSL
3. Configure environment variables for certificate paths

## Health Checks

All services include comprehensive health checks:

- **API**: `GET /health`
- **Coordinator**: gRPC channel ready check
- **Redis**: `redis-cli ping`
- **Nginx**: `wget /health`
- **Monitoring**: Prometheus and Grafana health endpoints
- **Jaeger**: Web UI availability

## Monitoring and Observability

### Access Points
- **API**: http://localhost (or https in production)
- **Grafana**: http://localhost:3000 (admin/admin or configured password)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

### Metrics Collection
- API server metrics: Request rates, latency, errors
- Coordinator metrics: Worker health, routing decisions
- System metrics: CPU, memory, disk usage
- Custom metrics: Inference performance, token rates

### Dashboards
Pre-configured Grafana dashboards:
- MLX Overview: System-wide metrics
- Worker Performance: Individual worker statistics
- API Performance: Request handling metrics

## Scaling

### Horizontal Scaling
```bash
# Scale API servers
docker-compose up -d --scale api=3

# Scale coordinators (production)
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --scale coordinator=2
```

### Resource Limits
Production configuration includes:
- Memory limits: 512M-2G per service
- CPU limits: 0.25-1.0 cores per service
- Restart policies: `on-failure` with backoff

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 80, 443, 3000, 6379, 9090, 16686, 50051 are available
2. **Memory issues**: Increase Docker memory limits for large models
3. **Network issues**: Check Docker network configuration and firewall rules

### Debugging Commands

```bash
# View logs
docker-compose logs -f [service_name]

# Check service health
docker-compose ps

# Execute commands in containers
docker-compose exec api bash
docker-compose exec coordinator python -c "import grpc; print('gRPC available')"

# Network debugging
docker network inspect mlx-distributed-inference_mlx-network
```

### Log Locations
- Container logs: `docker-compose logs`
- Nginx logs: `/var/log/nginx/` (in container)
- Application logs: stdout/stderr (captured by Docker)

## Backup and Recovery

### Data Backup
```bash
# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
docker cp mlx-redis:/data/dump.rdb ./backup/

# Backup Prometheus data
docker run --rm -v mlx-distributed-inference_prometheus_data:/data -v $(pwd)/backup:/backup alpine tar czf /backup/prometheus-$(date +%Y%m%d).tar.gz -C /data .

# Backup Grafana data
docker run --rm -v mlx-distributed-inference_grafana_data:/data -v $(pwd)/backup:/backup alpine tar czf /backup/grafana-$(date +%Y%m%d).tar.gz -C /data .
```

### Recovery
```bash
# Restore Redis data
docker cp ./backup/dump.rdb mlx-redis:/data/
docker-compose restart redis

# Restore Prometheus data
docker run --rm -v mlx-distributed-inference_prometheus_data:/data -v $(pwd)/backup:/backup alpine tar xzf /backup/prometheus-20231201.tar.gz -C /data

# Restore Grafana data
docker run --rm -v mlx-distributed-inference_grafana_data:/data -v $(pwd)/backup:/backup alpine tar xzf /backup/grafana-20231201.tar.gz -C /data
```

## Security Considerations

### Production Security Checklist
- [ ] Change default passwords (Grafana, Redis)
- [ ] Configure API keys for authentication
- [ ] Enable SSL/TLS with valid certificates
- [ ] Restrict metrics endpoint access
- [ ] Configure firewall rules
- [ ] Enable Redis authentication
- [ ] Review and update security headers
- [ ] Set up log monitoring and alerting

### Network Security
- Services communicate over internal Docker network
- Only necessary ports exposed to host
- Rate limiting configured in nginx
- Security headers added to all responses

## Performance Tuning

### Resource Optimization
- Adjust worker counts based on available CPU cores
- Configure memory limits based on model sizes
- Tune Redis memory policies for your workload
- Optimize nginx worker processes and connections

### Monitoring Performance
- Watch Grafana dashboards for bottlenecks
- Monitor container resource usage
- Track API response times and error rates
- Observe worker utilization and queue depths