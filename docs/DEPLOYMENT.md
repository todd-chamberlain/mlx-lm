# Deployment & Software Management Guide

## Overview

The MLX Distributed Inference system uses **automated deployment** to push software updates across N worker nodes. The coordinator and API run in Docker on a single control node, while workers run natively on each Mac Studio for GPU access.

## Deployment Architecture

```
┌─────────────────────────────────────┐
│  Control Node (mac-studio-1)        │
│  ┌───────────────────────────────┐  │
│  │   Docker Services             │  │
│  │   - Coordinator               │  │
│  │   - API                       │  │
│  │   - Monitoring                │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌───────────────────────────────┐  │
│  │   Deployment Manager          │  │
│  │   - rsync code to workers     │  │
│  │   - Install dependencies      │  │
│  │   - Version tracking          │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘
           │
           │ SSH/rsync
           ├──────────────────┬────────────────┐
           ▼                  ▼                ▼
    ┌──────────┐      ┌──────────┐    ┌──────────┐
    │ Worker 1 │      │ Worker 2 │    │ Worker N │
    │ (Native) │      │ (Native) │    │ (Native) │
    └──────────┘      └──────────┘    └──────────┘
```

## Initial Setup

### 1. Prerequisites

All worker nodes must have:
- **macOS** with Apple Silicon
- **Python 3.9+** installed
- **SSH access** from control node
- **Network connectivity** (Ethernet or Thunderbolt)

### 2. SSH Key Setup

Set up passwordless SSH to all worker nodes:

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy key to each worker
ssh-copy-id user@10.201.20.12  # mac-studio-2
ssh-copy-id user@10.201.20.13  # mac-studio-3
# ... etc

# Test connection
ssh user@10.201.20.12 "echo 'SSH OK'"
```

### 3. Configure Nodes

Edit `config/nodes.yaml`:

```yaml
workers:
  - id: "worker-studio1"
    host: "10.201.20.10"
    port: 8080
    connection_type: "local"  # Runs on control node

  - id: "worker-studio2"
    host: "10.201.20.12"
    port: 8080
    connection_type: "ssh"
    ssh_user: "username"
    work_dir: "~/mlx-distributed-inference"

  - id: "worker-studio3"
    host: "10.201.20.13"
    port: 8080
    connection_type: "thunderbolt"
    ssh_user: "username"
    work_dir: "~/mlx-distributed-inference"
```

### 4. Initial Deployment

Deploy to all nodes for the first time:

```bash
# This will:
# - Create directory structure
# - Set up Python virtual environment
# - Install dependencies
# - Copy code
./mlx-cli deploy --setup
```

Output:
```
🚀 Deploying to All Worker Nodes
============================================================

============================================================
Setting up worker-studio2 on 10.201.20.12
============================================================
  [Creating work directory]...
    ✅ Done
  [Creating logs directory]...
    ✅ Done
  [Creating config directory]...
    ✅ Done
  [Checking Python 3]...
    ✅ Done
  [Creating virtual environment]...
    ✅ Virtual environment created

  ✅ worker-studio2 setup complete

  Deploying to worker-studio2 (10.201.20.12)...
    ✅ Code deployed

  Installing dependencies on worker-studio2...
    ✅ Dependencies installed
```

## Regular Updates

### Deploying Code Changes

After making changes to the codebase:

```bash
# Preview what will be deployed
./mlx-cli deploy --dry-run

# Deploy to all nodes
./mlx-cli deploy

# Deploy without updating dependencies (faster)
./mlx-cli deploy --skip-deps
```

### Check Version Sync

Verify all nodes are running the same code version:

```bash
./mlx-cli check-sync
```

Output:
```
Checking Version Sync
============================================================

  ✅ worker-studio1: f05e30b (local: f05e30b)
  ✅ worker-studio2: f05e30b (local: f05e30b)
  ⚠️  worker-studio3: 2bd55d5 (local: f05e30b)
```

If nodes are out of sync, deploy updates:

```bash
./mlx-cli deploy
```

## What Gets Deployed

The deployment system syncs these files/directories:

- `worker/` - Worker code
- `common/` - Shared utilities
- `proto/` - Protocol definitions
- `requirements.txt` - Python dependencies
- `config/worker_models.yaml` - Model path mappings

**Excluded from sync:**
- `*.pyc`, `__pycache__/` - Compiled Python
- `*.log`, `logs/` - Log files
- `venv/` - Virtual environment (managed separately)
- `.git/` - Git repository
- `.DS_Store` - macOS metadata

## Deployment Workflow

### Standard Update Flow

```bash
# 1. Make code changes locally
vim worker/inference_engine.py

# 2. Test locally
./mlx-cli start-system  # or start-hybrid

# 3. Check what will be deployed
./mlx-cli deploy --dry-run

# 4. Deploy to all nodes
./mlx-cli deploy

# 5. Restart the system
./mlx-cli stop-hybrid
./mlx-cli start-hybrid

# 6. Verify health
./mlx-cli health
```

### Quick Update (Skip Dependencies)

For code-only changes that don't require new dependencies:

```bash
./mlx-cli deploy --skip-deps
./mlx-cli stop-hybrid && ./mlx-cli start-hybrid
```

### Git-Based Workflow

If using Git for version control:

```bash
# 1. Commit changes
git add .
git commit -m "Update inference engine"

# 2. Deploy to nodes
./mlx-cli deploy

# 3. Verify all nodes are on the same commit
./mlx-cli check-sync
```

## Adding a New Worker Node

### 1. Physical Setup

- Connect Mac Studio to network
- Note its IP address
- Set up SSH access

### 2. Add to Configuration

Edit `config/nodes.yaml`:

```yaml
workers:
  - id: "worker-studio4"
    host: "10.201.20.14"
    port: 8080
    connection_type: "ssh"
    ssh_user: "username"
    work_dir: "~/mlx-distributed-inference"
```

### 3. Add Model Paths

Edit `config/worker_models.yaml`:

```yaml
models:
  "qwen3-coder-480b-mlx":
    worker-studio4: "/Volumes/models/mlx/qwen3-coder-480b-mlx"
```

### 4. Deploy

```bash
# Initial setup of the new node
./mlx-cli deploy --setup

# Restart system to register new worker
./mlx-cli stop-hybrid
./mlx-cli start-hybrid

# Verify it connected
./mlx-cli health
```

## Removing a Worker Node

### 1. Stop the Worker

```bash
ssh worker-host "pkill -f worker.main"
```

### 2. Remove from Configuration

Edit `config/nodes.yaml` and remove the worker entry.

### 3. Restart System

```bash
./mlx-cli stop-hybrid
./mlx-cli start-hybrid
```

## Troubleshooting

### Deployment Fails

**Problem**: rsync errors or connection failures

**Solution**:
```bash
# Test SSH connectivity
ssh user@worker-host "echo OK"

# Check disk space on worker
ssh user@worker-host "df -h"

# Manually sync to one node for debugging
rsync -avz --dry-run worker/ user@worker-host:~/mlx-distributed-inference/worker/
```

### Version Mismatch

**Problem**: Workers running different code versions

**Solution**:
```bash
# Force re-deploy to all nodes
./mlx-cli deploy

# Check sync status
./mlx-cli check-sync
```

### Dependency Errors

**Problem**: Workers fail to start due to missing dependencies

**Solution**:
```bash
# Re-install dependencies on specific node
ssh user@worker-host "cd ~/mlx-distributed-inference && source venv/bin/activate && pip install -r requirements.txt"

# Or re-deploy with dependencies
./mlx-cli deploy
```

### Permission Denied

**Problem**: Can't write to worker directories

**Solution**:
```bash
# Fix ownership on worker
ssh user@worker-host "sudo chown -R $USER ~/mlx-distributed-inference"

# Ensure work directory is writable
ssh user@worker-host "chmod -R u+w ~/mlx-distributed-inference"
```

## Advanced Topics

### Custom Deployment Locations

To use a different directory on workers:

```yaml
workers:
  - id: "worker-custom"
    work_dir: "/custom/path/to/mlx"
```

### Deployment Hooks

Create pre/post deployment scripts:

```bash
# On each worker: ~/mlx-distributed-inference/hooks/pre-deploy.sh
#!/bin/bash
echo "Running pre-deployment checks..."
python -m pytest tests/

# Make executable
chmod +x hooks/pre-deploy.sh
```

### Rollback

If a deployment breaks something:

```bash
# On each worker, revert to previous commit
ssh user@worker-host "cd ~/mlx-distributed-inference && git reset --hard HEAD~1"

# Or restore from backup
ssh user@worker-host "cp -r ~/mlx-distributed-inference.backup ~/mlx-distributed-inference"

# Restart workers
./mlx-cli stop-hybrid && ./mlx-cli start-hybrid
```

## Best Practices

### 1. Always Test Locally First

```bash
./mlx-cli start-system  # Test in all-native mode
# Run tests, verify changes work
./mlx-cli stop-system
```

### 2. Use Dry Run

```bash
./mlx-cli deploy --dry-run  # See what would change
```

### 3. Deploy During Low Traffic

Schedule updates during maintenance windows or low-usage periods.

### 4. Version Control Everything

```bash
git commit -am "Your changes"
git push origin main
./mlx-cli deploy
```

### 5. Monitor After Deployment

```bash
./mlx-cli health
./mlx-cli check-sync

# Check worker logs
tail -f logs/worker-*.log
```

### 6. Keep Dependencies Minimal

Only update `requirements.txt` when necessary. Code-only updates deploy faster with `--skip-deps`.

## Security Considerations

### SSH Key Management

- Use ed25519 keys (stronger, faster)
- Protect private keys: `chmod 600 ~/.ssh/id_ed25519`
- Consider using SSH agent
- Rotate keys periodically

### Network Security

- Use firewall rules to restrict access
- Consider VPN for remote workers
- Use SSH port forwarding for management

### Code Integrity

- Sign commits with GPG
- Use Git hooks to prevent accidental commits
- Review code before deployment

## Monitoring Deployment Health

### Automated Checks

Add to your monitoring:

```bash
# Cron job to check sync status
*/30 * * * * /path/to/mlx-cli check-sync || mail -s "Version mismatch" admin@example.com
```

### Grafana Dashboard

Create panels for:
- Worker version numbers
- Last deployment time
- Deployment success rate
- Time since last sync
