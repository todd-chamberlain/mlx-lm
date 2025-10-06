# 🚀 Quick Start - Run on Mac Studios

Get the MLX Distributed Inference System running in 5 minutes!

## Prerequisites Check

```bash
# Check you have everything
python3 --version  # Should be 3.9+
python3 -c "import mlx.core; print('✓ MLX installed')"
docker --version   # Only on Mac Studio 1
```

## Install Dependencies

```bash
# Install MLX if needed
pip3 install mlx mlx-lm

# Install project dependencies
pip3 install -r requirements.txt
```

## Option 1: Quick Test (Single Machine)

Run everything on one Mac Studio for testing:

```bash
# 1. Start containerized services
docker-compose up -d

# 2. Wait for services to start
sleep 10

# 3. Check services
docker-compose ps

# 4. Start a local worker
python3 -m worker.main \
  --worker-id worker-local \
  --coordinator localhost:50051 \
  --model-path /Users/Shared/models \
  --port 50052

# 5. Test it
./test_deployment.sh
```

## Option 2: Full Deployment (Two Mac Studios)

### On Mac Studio 1 (Coordinator Host)

```bash
# 1. Get your IP address
export STUDIO1_IP=$(ipconfig getifaddr en0)
echo "Mac Studio 1 IP: $STUDIO1_IP"

# 2. Start containerized services
docker-compose up -d

# 3. Wait for services
sleep 10

# 4. Start local worker
python3 -m worker.main \
  --worker-id worker-studio1 \
  --coordinator localhost:50051 \
  --model-path /Users/Shared/models \
  --port 50052 &

# 5. Check health
curl http://localhost:8000/health
```

### On Mac Studio 2 (Worker Host)

```bash
# 1. Set Studio 1's IP (replace with actual IP)
export STUDIO1_IP="192.168.1.100"

# 2. Start worker pointing to Studio 1
python3 -m worker.main \
  --worker-id worker-studio2 \
  --coordinator ${STUDIO1_IP}:50051 \
  --model-path /Users/Shared/models \
  --port 50052 &

# 3. Check it registered
# (Run on Studio 1)
curl http://localhost:8000/health
# Should show 2 workers
```

## Test It!

```bash
# Run the test script (on Mac Studio 1)
./test_deployment.sh

# Or test manually
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-api-key-here" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Access Monitoring

```bash
# Grafana (username: admin, password: admin)
open http://localhost:3000

# Prometheus
open http://localhost:9090

# API Health
open http://localhost:8000/health
```

## Stop Everything

```bash
# Stop workers
pkill -f "python3 -m worker.main"

# Stop containerized services
docker-compose down
```

## Troubleshooting

### "Connection refused" error

```bash
# Check coordinator is running
docker-compose ps coordinator

# Check firewall
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

# Check network connectivity
ping <other-mac-ip>
nc -zv <other-mac-ip> 50051
```

### "Model not found" error

```bash
# Download a model first
mkdir -p /Users/Shared/models
cd /Users/Shared/models

# Download Llama 3.2 3B (4-bit quantized)
huggingface-cli download mlx-community/Llama-3.2-3B-Instruct-4bit \
  --local-dir Llama-3.2-3B-Instruct-4bit
```

### Worker not registering

```bash
# Check coordinator logs
docker-compose logs coordinator

# Check worker logs
# (Worker logs go to stdout, check terminal)

# Verify coordinator address
echo $STUDIO1_IP
nc -zv $STUDIO1_IP 50051
```

## What's Running?

After successful deployment:

**Mac Studio 1:**
- ✅ API Server (port 8000)
- ✅ Coordinator (port 50051)
- ✅ Redis (port 6379)
- ✅ Prometheus (port 9090)
- ✅ Grafana (port 3000)
- ✅ Worker 1 (native MLX)

**Mac Studio 2:**
- ✅ Worker 2 (native MLX)

## Next Steps

1. **Load test**: `python tests/load_testing_scenarios.py`
2. **Performance test**: `python tests/test_performance_benchmarking.py`
3. **View metrics**: Open Grafana at http://localhost:3000
4. **Read full guide**: See `DEPLOYMENT_GUIDE_MAC_STUDIOS.md`

## Need Help?

- Check logs: `docker-compose logs -f`
- Check health: `curl http://localhost:8000/health`
- View workers: `curl http://localhost:8000/metrics | grep worker`
- Full documentation: See `DEPLOYMENT_GUIDE_MAC_STUDIOS.md`

---

**Ready? Let's go!** 🚀

```bash
# One command to rule them all (Mac Studio 1)
docker-compose up -d && sleep 10 && ./test_deployment.sh
```
