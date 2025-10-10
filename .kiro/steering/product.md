# Product Overview

MLX Distributed Inference System is a production-ready distributed inference platform for large language models on Apple Silicon. It enables running models that exceed single-machine memory capacity by distributing layers across multiple Mac machines.

## Core Value Proposition

- **Distribute large models** (70B-800B parameters) across multiple Mac Studios
- **Reduce memory usage** by 50% per node through layer-level partitioning
- **Increase throughput** by 2-3x with intelligent batching and caching
- **OpenAI-compatible API** for seamless integration

## Architecture

Three-tier distributed system:
1. **API Server** (FastAPI) - OpenAI-compatible REST endpoints with streaming support
2. **Coordinator** - Worker management, request routing, health monitoring, cache coordination
3. **Workers** - MLX inference engines running model layer partitions on Apple Silicon

## Key Features

- Layer-level model partitioning across nodes
- Multi-level caching (activation cache, KV cache)
- Adaptive batch processing
- Memory optimization with pooling and GC tuning
- Comprehensive observability (Prometheus, Grafana, Jaeger)
- Automatic service discovery and health monitoring
- Hot-swappable models with graceful draining
