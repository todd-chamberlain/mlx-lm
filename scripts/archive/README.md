# Archived Scripts

This directory contains scripts that were part of earlier implementations
but have been superseded by the current production system.

## Why Archived

These scripts are **obsolete** and replaced by:
- **Hybrid deployment system**: `hybrid_orchestrator.py`
- **Unified CLI**: `mlx-cli` (mlx_distributed_cli.py)
- **Deployment tool**: `deploy_to_nodes.py`
- **Dependency manager**: `manage_dependencies.py`

## Contents

### Obsolete Docker Configs
- Old development/production compose files replaced by `docker-compose.hybrid.yml`

### Old Deployment Scripts
- Manual deployment scripts replaced by automated `deploy_to_nodes.py`

### Old Worker Startup
- Individual worker startup scripts replaced by orchestrator

### Partial Sharding Implementations
- Early sharding attempts replaced by proper model registry + coordinator

### Debug Scripts
- Temporary debugging scripts no longer needed

## Recovery

If you need any of these scripts, they're preserved here.
**DO NOT use these in production** - use the current system instead.
