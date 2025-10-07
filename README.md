# MLX Distributed Inference

Production-ready distributed layer inference for Apple Silicon.

## Quick Start

```bash
# Deploy
./deploy_native_only.sh

# Test
curl http://localhost:8000/health

# Inference
curl -X POST http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
```

## Features

- ✅ **Distributed Layers** - Split models across nodes
- ✅ **Hot Model Swapping** - Zero downtime model changes
- ✅ **Background Downloads** - Pre-cache models
- ✅ **OpenAI-Compatible** - Drop-in API replacement
- ✅ **Universal Models** - Works with any MLX model

## Documentation

📖 **[Complete Guide](DISTRIBUTED_INFERENCE_GUIDE.md)** - Everything you need to know

## Architecture

```
Client → API (8000) → Coordinator (50051) → Workers (50052)
                                              ├─ Node 1: Layers 0-N/2
                                              └─ Node 2: Layers N/2-N
```

## Admin API

```bash
# Download model (background)
curl -X POST "http://localhost:8000/admin/download_model?model_name=MODEL"

# Check download
curl http://localhost:8000/admin/download_status

# Hot-swap model
curl -X POST "http://localhost:8000/admin/load_model?model_name=MODEL"
```

## Configuration

Edit `config/distributed_config.yaml`:
```yaml
nodes:
  - id: "node-1"
    address: "10.201.20.10:50052"
  - id: "node-2"
    address: "10.201.20.12:50052"
```

## License

MIT
