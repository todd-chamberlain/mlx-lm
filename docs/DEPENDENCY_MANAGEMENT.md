# Dependency Management Guide

## Overview

Managing Python dependencies across N remote worker nodes requires:
- **Version locking** to ensure consistency
- **Snapshot/rollback** capability for safety
- **Consistency checking** to detect drift
- **Automated sync** to all workers

## Quick Reference

```bash
# Lock current versions
./mlx-cli freeze-deps

# Check consistency across workers
./mlx-cli check-deps

# Sync to all workers
./mlx-cli sync-deps

# Create rollback point
./mlx-cli snapshot-deps --name before-upgrade

# Force reinstall everything
./mlx-cli sync-deps --force
```

---

## Dependency Workflow

### Initial Setup

When setting up workers for the first time:

```bash
# 1. Install dependencies locally
pip install -r requirements.txt

# 2. Lock versions
./mlx-cli freeze-deps

# 3. Deploy to workers (includes deps)
./mlx-cli deploy --setup
```

### Adding New Dependencies

When you need to add a new package:

```bash
# 1. Install locally
pip install new-package

# 2. Update requirements.txt
echo "new-package==1.2.3" >> requirements.txt

# 3. Lock all versions
./mlx-cli freeze-deps

# 4. Sync to workers
./mlx-cli sync-deps

# 5. Restart system
./mlx-cli stop-hybrid && ./mlx-cli start-hybrid
```

### Upgrading Dependencies

When upgrading existing packages:

```bash
# 1. Create snapshot before upgrade
./mlx-cli snapshot-deps --name before-mlx-upgrade

# 2. Upgrade locally
pip install --upgrade mlx-lm

# 3. Test locally
./mlx-cli start-system  # Test with new version

# 4. If tests pass, lock and sync
./mlx-cli freeze-deps
./mlx-cli sync-deps

# 5. Restart workers
./mlx-cli stop-hybrid && ./mlx-cli start-hybrid

# 6. Verify
./mlx-cli check-deps
```

### Checking Consistency

Regularly check if all workers have matching dependencies:

```bash
./mlx-cli check-deps
```

**Output:**
```
🔍 Checking dependency consistency...

  Checking worker-studio1...
    ✅ Local worker (skipped)

  Checking worker-studio2...
    ✅ Consistent (247 packages)

  Checking worker-studio3...
    ⚠️  Inconsistencies found:
      Version mismatches:
        mlx-lm: 0.4.0 -> 0.5.0
        numpy: 1.24.0 -> 1.25.0
      Missing packages: 2
      Extra packages: 0

✅ All workers have consistent dependencies
```

### Rollback from Snapshot

If an upgrade causes problems:

```bash
# 1. List available snapshots
./scripts/manage_dependencies.py list-snapshots

# Output:
# Available snapshots:
#   before-mlx-upgrade (247 packages)
#   20250110_120000 (245 packages)

# 2. Restore from snapshot
./scripts/manage_dependencies.py restore before-mlx-upgrade

# 3. Sync to workers
./mlx-cli sync-deps --force

# 4. Restart
./mlx-cli stop-hybrid && ./mlx-cli start-hybrid
```

---

## Files Explained

### `requirements.txt`
**Purpose**: Human-readable list of dependencies
- Edited manually when adding/removing packages
- May use version ranges (`>=`, `~=`)
- Source of truth for what SHOULD be installed

**Example:**
```
mlx-lm>=0.5.0
numpy~=1.25.0
transformers>=4.30.0
```

### `requirements.lock`
**Purpose**: Machine-generated exact versions
- Created by `./mlx-cli freeze-deps`
- Pin ALL dependencies (including transitive)
- Used by `./mlx-cli sync-deps` for consistency

**Example:**
```
mlx-lm==0.5.2
numpy==1.25.1
transformers==4.30.2
tokenizers==0.13.3
...
```

### `.dependency-snapshots/`
**Purpose**: Rollback points
- Created by `./mlx-cli snapshot-deps`
- Each snapshot is a frozen state
- Timestamped automatically

**Structure:**
```
.dependency-snapshots/
├── snapshot_before-mlx-upgrade.txt
├── snapshot_20250110_120000.txt
└── snapshot_20250109_150000.txt
```

---

## Common Scenarios

### Scenario 1: New Worker Added

```bash
# Worker already has Python, venv created by deploy --setup
# Just sync dependencies
./mlx-cli sync-deps

# Verify
./mlx-cli check-deps
```

### Scenario 2: Version Mismatch Detected

```bash
# Check shows mismatches
./mlx-cli check-deps

# Sync to fix
./mlx-cli sync-deps

# Force reinstall if corruption suspected
./mlx-cli sync-deps --force
```

### Scenario 3: Major Upgrade (e.g., MLX 0.4 → 0.5)

```bash
# 1. Snapshot current state
./mlx-cli snapshot-deps --name before-mlx-0.5

# 2. Test upgrade on ONE worker first
ssh worker-studio2 "cd ~/mlx-distributed-inference && \
  source venv/bin/activate && \
  pip install --upgrade mlx-lm"

# 3. Test that worker
# ... run tests ...

# 4. If successful, upgrade locally
pip install --upgrade mlx-lm

# 5. Lock and sync to ALL workers
./mlx-cli freeze-deps
./mlx-cli sync-deps

# 6. If problems, rollback
./scripts/manage_dependencies.py restore before-mlx-0.5
./mlx-cli sync-deps --force
```

### Scenario 4: Dependency Conflict

```bash
# Problem: Package A requires numpy>=1.25, Package B requires numpy<1.25

# 1. Check what's installed
pip list | grep numpy

# 2. Find compatible versions
pip index versions package-a
pip index versions package-b

# 3. Update requirements.txt with compatible versions
vim requirements.txt

# 4. Install and test locally
pip install -r requirements.txt

# 5. If works, sync
./mlx-cli freeze-deps
./mlx-cli sync-deps
```

### Scenario 5: Clean Install

To wipe and reinstall all dependencies:

```bash
# On each worker (via SSH or script)
ssh worker-host "cd ~/mlx-distributed-inference && \
  rm -rf venv && \
  python3 -m venv venv"

# Then sync
./mlx-cli sync-deps --force
```

---

## Best Practices

### 1. Always Lock Before Syncing

```bash
# Good
./mlx-cli freeze-deps
./mlx-cli sync-deps

# Bad (inconsistent versions)
./mlx-cli sync-deps --no-lock
```

### 2. Snapshot Before Major Changes

```bash
./mlx-cli snapshot-deps --name descriptive-name
```

### 3. Check Consistency Regularly

Add to CI/CD or cron:
```bash
*/30 * * * * /path/to/mlx-cli check-deps || notify-admin
```

### 4. Use `--force` Sparingly

Only use when:
- Corruption suspected
- Rollback from bad state
- Clean install needed

Normal updates don't need `--force`.

### 5. Test Locally First

```bash
# Install and test locally
pip install new-package
./mlx-cli start-system

# If tests pass, deploy
./mlx-cli freeze-deps
./mlx-cli deploy
./mlx-cli sync-deps
```

---

## Troubleshooting

### Issue: Sync Fails with "Permission Denied"

**Solution:**
```bash
ssh worker-host "chmod -R u+w ~/mlx-distributed-inference/venv"
./mlx-cli sync-deps
```

### Issue: "requirements.lock not found"

**Solution:**
```bash
./mlx-cli freeze-deps  # Create it
./mlx-cli sync-deps
```

### Issue: Workers Have Different Versions

**Diagnosis:**
```bash
./mlx-cli check-deps
```

**Solution:**
```bash
./mlx-cli sync-deps --force
```

### Issue: Package Installation Timeout

**Solution:**
```bash
# Increase timeout in manage_dependencies.py (line ~250)
# Or install problematic package separately:
ssh worker-host "cd ~/mlx-distributed-inference && \
  source venv/bin/activate && \
  pip install --no-deps problematic-package"

# Then sync rest
./mlx-cli sync-deps
```

### Issue: Circular Dependency

**Solution:**
```bash
# Install in specific order
ssh worker-host "cd ~/mlx-distributed-inference && \
  source venv/bin/activate && \
  pip install package-a && \
  pip install package-b"

# Update lock file
./mlx-cli freeze-deps
```

---

## Advanced Usage

### Manual Dependency Check

```bash
# Check specific worker
./scripts/manage_dependencies.py check

# Compare specific package across workers
for host in worker1 worker2 worker3; do
  echo "$host:"
  ssh $host "source ~/mlx-distributed-inference/venv/bin/activate && pip show mlx-lm | grep Version"
done
```

### Upgrade Single Package

```bash
# Use built-in upgrade command
./scripts/manage_dependencies.py upgrade mlx-lm --version 0.5.0

# Or manually
pip install --upgrade mlx-lm
./mlx-cli freeze-deps
./mlx-cli sync-deps
```

### Export Dependency Report

```bash
# Create detailed dependency report
./scripts/manage_dependencies.py check > dependency-report.txt

# Share with team or attach to issue
```

### Automated Dependency Updates

```bash
# Cron job to check and notify
0 9 * * 1 /path/to/mlx-cli check-deps || \
  mail -s "Dependency drift detected" team@example.com
```

---

## Integration with CI/CD

### Pre-deployment Check

```bash
#!/bin/bash
# .github/workflows/deploy.yml or similar

# Check dependencies are locked
if [ ! -f requirements.lock ]; then
  echo "Error: requirements.lock missing"
  exit 1
fi

# Verify local matches lock
pip freeze > /tmp/current.txt
if ! diff requirements.lock /tmp/current.txt; then
  echo "Error: Local dependencies don't match lock file"
  exit 1
fi

# Deploy
./mlx-cli deploy
./mlx-cli sync-deps
```

### Post-deployment Verification

```bash
#!/bin/bash
# Verify all workers synced

if ! ./mlx-cli check-deps; then
  echo "ERROR: Dependency sync failed!"
  # Rollback or alert
  exit 1
fi
```

---

## Security Considerations

### Private Package Indices

If using private PyPI:

```bash
# Configure pip on each worker
ssh worker-host "cat >> ~/mlx-distributed-inference/venv/pip.conf << EOF
[global]
index-url = https://pypi.private.com/simple
trusted-host = pypi.private.com
EOF"
```

### Hash Verification

For maximum security:

```bash
# Generate hashes
pip hash mlx-lm==0.5.0

# Add to requirements.txt
mlx-lm==0.5.0 --hash=sha256:abc123...

# Sync with hash verification
./mlx-cli sync-deps
```

### Dependency Scanning

```bash
# Check for vulnerabilities
pip install safety
safety check -r requirements.lock

# Run on all workers
for worker in $(yq '.workers[].host' config/nodes.yaml); do
  echo "Checking $worker"
  ssh $worker "cd ~/mlx-distributed-inference && \
    source venv/bin/activate && \
    safety check"
done
```
