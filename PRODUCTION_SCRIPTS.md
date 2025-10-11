# Production Scripts Reference

**Last Cleanup:** 2025-01-10
**Status:** Production Ready ✅

This document describes the **production-ready scripts** for the MLX Distributed Inference System. All obsolete/experimental scripts have been archived.

---

## 📁 Directory Structure

```
mlx-lm/
├── scripts/               # Production scripts
│   ├── archive/          # Obsolete scripts (archived)
│   ├── deploy_to_nodes.py
│   ├── manage_dependencies.py
│   ├── hybrid_orchestrator.py
│   ├── download_qwen3_with_sharding.py
│   └── ...
├── tests/                # Test scripts
│   └── performance/      # Performance tests
├── maintenance/          # Maintenance utilities
├── mlx-cli              # Main CLI (symlink to mlx_distributed_cli.py)
└── docker-compose.hybrid.yml  # Production Docker config
```

---

## 🚀 Production Scripts

### **Core System Management**

| Script | Purpose | Usage |
|--------|---------|-------|
| `mlx_distributed_cli.py` | **Main CLI** - Unified interface | `./mlx-cli [command]` |
| `hybrid_orchestrator.py` | System startup orchestration | Called by `mlx-cli start-hybrid` |
| `start_hybrid_system.sh` | Bash wrapper for startup | Alternative to orchestrator |
| `stop_hybrid_system.sh` | System shutdown | Called by `mlx-cli stop-hybrid` |

### **Deployment & Updates**

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy_to_nodes.py` | Deploy code to N workers | `./mlx-cli deploy` |
| `manage_dependencies.py` | Dependency management | `./mlx-cli sync-deps` |

### **Model Management**

| Script | Purpose | Usage |
|--------|---------|-------|
| `download_qwen3_with_sharding.py` | Load sharded models | `python scripts/download_qwen3_with_sharding.py` |

### **Utilities**

| Script | Purpose | Usage |
|--------|---------|-------|
| `health_check.sh` | Quick health check | `./scripts/health_check.sh` |
| `run_tests.sh` | Run test suite | `./scripts/run_tests.sh` |
| `test_qwen3_coder_480b.py` | Test specific model | `python scripts/test_qwen3_coder_480b.py` |
| `generate_proto.py` | Regenerate protobuf | `python scripts/generate_proto.py` |

---

## 🧪 Test Scripts

Located in `tests/` directory:

### **Performance Tests**
- `tests/performance/measure_memory_savings.py` - Memory optimization metrics
- `tests/performance/measure_performance.py` - Performance benchmarking
- `tests/performance/run_performance_baseline.py` - Baseline measurements

### **Integration Tests**
- `tests/run_core_tests.sh` - Core functionality tests
- `tests/test_unified_system.py` - End-to-end system tests

See `tests/` directory for full test suite (50+ test files).

---

## 🔧 Maintenance Scripts

Located in `maintenance/` directory:

| Script | Purpose |
|--------|---------|
| `backup_recovery.sh` | Backup and recovery utilities |
| `log_rotation.sh` | Log file management |

---

## 📦 Archived Scripts

Located in `scripts/archive/` - **DO NOT USE IN PRODUCTION**

These are obsolete implementations from earlier development phases:
- Old Docker configs (replaced by `docker-compose.hybrid.yml`)
- Manual deployment scripts (replaced by `deploy_to_nodes.py`)
- Individual worker startup (replaced by orchestrator)
- Partial sharding attempts (replaced by model registry)
- Temporary debug scripts

See `scripts/archive/README.md` for details.

---

## 🎯 Quick Reference

### Daily Operations

```bash
# Start system
./mlx-cli start-hybrid

# Check health
./mlx-cli health

# Check version sync
./mlx-cli check-sync

# Check dependencies
./mlx-cli check-deps

# Stop system
./mlx-cli stop-hybrid
```

### Deployment

```bash
# Deploy code updates
./mlx-cli deploy

# Deploy with dry-run preview
./mlx-cli deploy --dry-run

# Sync dependencies
./mlx-cli sync-deps
```

### Troubleshooting

```bash
# Check worker logs
tail -f logs/worker-studio1.log

# Check Docker services
docker logs mlx-coordinator
docker logs mlx-api

# Run health check
./scripts/health_check.sh

# Run tests
./scripts/run_tests.sh
```

---

## 🗑️ What Was Removed

### Cleanup Summary (2025-01-10)

**Archived:** 30+ obsolete files including:
- 3 old docker-compose configs
- 8 manual deployment scripts
- 7 individual worker startup scripts
- 8 partial sharding implementations
- 7 temporary debug scripts
- 1 old distributed system startup

**Organized:**
- 5 test scripts → `tests/performance/`
- 2 maintenance scripts → `maintenance/`

**Result:** Clean, focused production system with **only 10 production scripts**.

---

## 📚 Documentation

Full documentation available in `docs/`:

| Document | Purpose |
|----------|---------|
| `docs/HYBRID_DEPLOYMENT.md` | Architecture & deployment guide |
| `docs/DEPLOYMENT.md` | Code deployment workflows |
| `docs/DEPENDENCY_MANAGEMENT.md` | Dependency management guide |
| `README.md` | Project overview |

---

## ✅ Production Checklist

Before deploying to production:

- [ ] All workers configured in `config/nodes.yaml`
- [ ] Model paths set in `config/worker_models.yaml`
- [ ] Dependencies locked: `./mlx-cli freeze-deps`
- [ ] Dry-run deployment: `./mlx-cli deploy --dry-run`
- [ ] Tests passing: `./scripts/run_tests.sh`
- [ ] Health check working: `./mlx-cli health`
- [ ] Monitoring accessible (Grafana, Prometheus)
- [ ] Backup strategy in place

---

## 🆘 Support

### Getting Help

1. Check documentation in `docs/`
2. Run `./mlx-cli --help` for CLI reference
3. Check archived scripts if looking for old implementations
4. Review test files for usage examples

### Reporting Issues

When reporting issues, include:
- Output of `./mlx-cli health`
- Output of `./mlx-cli check-sync`
- Output of `./mlx-cli check-deps`
- Relevant logs from `logs/` or `docker logs`

---

## 🔄 Maintenance

### Monthly
- Review and rotate logs (`maintenance/log_rotation.sh`)
- Check for dependency updates
- Backup configurations (`maintenance/backup_recovery.sh`)

### Quarterly
- Review archived scripts - can old ones be deleted?
- Update documentation
- Review test coverage

### Annual
- Major dependency upgrades
- Architecture review
- Capacity planning
