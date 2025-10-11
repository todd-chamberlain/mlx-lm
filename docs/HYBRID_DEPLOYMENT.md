# Hybrid Deployment Guide

## Architecture

The **Hybrid Deployment** runs coordination services in Docker while MLX workers run natively on Apple Silicon for optimal GPU access.

```
┌─────────────────────────────────────┐
│      Docker Services (mac-studio-1) │
│                                     │
│  ┌──────────┐  ┌──────────┐        │
│  │ Redis    │  │Coordinator│        │
│  └──────────┘  └──────────┘        │
│                                     │
│  ┌──────────┐  ┌──────────┐        │
│  │   API    │  │Prometheus │        │
│  └──────────┘  └──────────┘        │
│                                     │
│  ┌──────────┐                       │
│  │ Grafana  │                       │
│  └──────────┘                       │
└─────────────────────────────────────┘
           │
           │ gRPC (50051)
           │
    ┌──────┴────────┐
    │               │
    ▼               ▼
┌─────────┐    ┌─────────┐
│ Worker  │    │ Worker  │
│Studio 1 │    │Studio 2 │
│ (Native)│    │ (Native)│
│  MLX    │    │  MLX    │
└─────────┘    └─────────┘
```

## Why Hybrid?

- **Docker services**: Easy deployment, scaling, monitoring
- **Native MLX workers**: Direct GPU access on Apple Silicon
- **Best of both worlds**: Container orchestration + native performance

## Quick Start

### 1. Configure Nodes

Edit `config/nodes.yaml`:

```yaml
workers:
  - id: "worker-studio1"
    host: "10.201.20.10"
    port: 8080
    connection_type: "local"

  - id: "worker-studio2"
    host: "10.201.20.12"
    port: 8080
    connection_type: "ssh"
    ssh_user: "your_username"
    work_dir: "~/mlx-distributed-inference"

  # Add more workers:
  - id: "worker-studio3"
    host: "10.201.20.13"
    port: 8080
    connection_type: "thunderbolt"
    ssh_user: "username"
    work_dir: "~/mlx-distributed-inference"
```

### 2. Configure Model Paths

Edit `config/worker_models.yaml`:

```yaml
models:
  "qwen3-coder-480b-mlx":
    worker-studio1: "/Volumes/thunderbay/models/qwen3-coder-480b-mlx"
    worker-studio2: "/Volumes/models/mlx/qwen3-coder-480b-mlx"
    worker-studio3: "/Volumes/models/mlx/qwen3-coder-480b-mlx"
```

### 3. Start the System

```bash
# Simple way
./mlx-cli start-hybrid

# Or manually
./scripts/hybrid_orchestrator.py
```

### 4. Verify Health

```bash
./mlx-cli health

# Or
curl http://localhost:8000/health | jq .
```

## Services

| Service | URL | Description |
|---------|-----|-------------|
| API | http://localhost:8000 | OpenAI-compatible API |
| Coordinator | localhost:50051 | gRPC coordination service |
| Prometheus | http://localhost:9090 | Metrics collection |
| Grafana | http://localhost:3000 | Dashboards (admin/admin) |

## Scaling to N Nodes

### Adding a New Worker

1. **Install MLX on the new machine:**
   ```bash
   ssh new-machine
   git clone <repo>
   cd mlx-distributed-inference
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Add to `config/nodes.yaml`:**
   ```yaml
   workers:
     - id: "worker-new"
       host: "10.201.20.XX"
       port: 8080
       connection_type: "ssh"
       ssh_user: "username"
       work_dir: "~/mlx-distributed-inference"
   ```

3. **Add model paths to `config/worker_models.yaml`:**
   ```yaml
   models:
     "qwen3-coder-480b-mlx":
       worker-new: "/path/to/model"
   ```

4. **Restart the system:**
   ```bash
   ./mlx-cli stop-hybrid
   ./mlx-cli start-hybrid
   ```

## Thunderbolt Networking

For thunderbolt-connected nodes:

1. **Configure thunderbolt bridge:**
   - System Settings → Network → Thunderbolt Bridge
   - Note the IP address (usually 169.254.x.x)

2. **Update `config/nodes.yaml`:**
   ```yaml
   - id: "worker-thunderbolt"
     host: "169.254.x.x"
     connection_type: "thunderbolt"
   ```

3. **Test connectivity:**
   ```bash
   ping 169.254.x.x
   ssh user@169.254.x.x
   ```

## Loading Sharded Models

With all workers running:

```bash
# Load Qwen 3 Coder 480B with automatic sharding
python scripts/download_qwen3_with_sharding.py
```

This will:
1. Check available workers
2. Create shard plan (layer distribution)
3. Tell each worker to load its assigned layers
4. Monitor progress

## Monitoring

### Logs

```bash
# Docker services
docker logs mlx-coordinator
docker logs mlx-api

# Native workers
tail -f logs/worker-studio1.log
ssh studio2 'tail -f ~/mlx-distributed-inference/logs/worker-studio2.log'
```

### Metrics

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000

Default dashboards show:
- Worker health
- Request latency
- Token generation speed
- Memory usage

## Troubleshooting

### Workers Not Connecting

```bash
# Check coordinator is accessible
telnet localhost 50051

# Check worker logs
tail -f logs/worker-*.log

# Verify network connectivity
ping <worker-host>
```

### Docker Services Not Starting

```bash
# Check Docker is running
docker info

# View service logs
docker-compose -f docker-compose.hybrid.yml logs

# Restart services
docker-compose -f docker-compose.hybrid.yml restart
```

### Model Not Loading

```bash
# Verify model paths in config
cat config/worker_models.yaml

# Check worker can access model
ssh worker-host "ls -la /path/to/model"

# View worker logs during load
tail -f logs/worker-*.log
```

## Stopping the System

```bash
# Stop everything
./mlx-cli stop-hybrid

# Or manually
./scripts/stop_hybrid_system.sh
```

## Advanced Configuration

### Custom Docker Network

Edit `docker-compose.hybrid.yml` to change subnet:

```yaml
networks:
  mlx-network:
    ipam:
      config:
        - subnet: 172.21.0.0/16
```

### Resource Limits

Add to docker-compose services:

```yaml
coordinator:
  deploy:
    resources:
      limits:
        memory: 2G
        cpus: "2.0"
```

### Authentication

Enable API authentication:

```yaml
api:
  environment:
    - ENABLE_AUTH=true
    - API_KEYS=your-secret-key
```

## Production Checklist

- [ ] Configure firewall rules for ports 8000, 50051
- [ ] Set up SSL/TLS for API endpoint
- [ ] Enable authentication
- [ ] Configure backup for Redis data
- [ ] Set up log rotation
- [ ] Configure Grafana alerts
- [ ] Test failover scenarios
- [ ] Document recovery procedures
