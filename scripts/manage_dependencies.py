#!/usr/bin/env python3
"""
Dependency Management for Remote Worker Nodes

Manages Python dependencies across all workers with:
- Version locking and consistency
- Freeze/restore capability
- Conflict detection
- Rollback support
"""

import os
import sys
import yaml
import subprocess
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Colors
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


class DependencyManager:
    """Manages Python dependencies across worker nodes"""

    def __init__(self, config_path: str = "config/nodes.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.workers = self.config.get('workers', [])

        # Lock file for consistent versions
        self.lock_file = Path("requirements.lock")
        self.freeze_dir = Path(".dependency-snapshots")
        self.freeze_dir.mkdir(exist_ok=True)

    def _load_config(self) -> dict:
        """Load node configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _print(self, msg: str, color: str = Colors.NC):
        """Print colored message"""
        print(f"{color}{msg}{Colors.NC}")

    def _run_remote_command(self, worker: dict, command: str, timeout: int = 60) -> Tuple[bool, str, str]:
        """
        Run command on remote worker.
        Returns: (success, stdout, stderr)
        """
        connection_type = worker.get('connection_type')
        if connection_type == 'local':
            # Run locally
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr

        # Remote execution via SSH
        ssh_user = worker.get('ssh_user', os.getenv('USER'))
        host = worker['host']
        work_dir = worker.get('work_dir', '~/mlx-distributed-inference')
        ssh_target = f"{ssh_user}@{host}" if ssh_user else host

        ssh_cmd = f"ssh {ssh_target} 'cd {work_dir} && {command}'"

        try:
            result = subprocess.run(
                ssh_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timeout"

    def freeze_local(self) -> bool:
        """
        Freeze local dependencies to requirements.lock.
        This creates a snapshot of exact versions.
        """
        self._print("📸 Freezing local dependencies...", Colors.CYAN)

        try:
            # Activate venv and freeze
            result = subprocess.run(
                "source venv/bin/activate && pip freeze",
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )

            # Write to lock file
            with open(self.lock_file, 'w') as f:
                f.write(result.stdout)

            # Calculate hash for verification
            lock_hash = hashlib.sha256(result.stdout.encode()).hexdigest()[:8]

            self._print(f"  ✅ Frozen {len(result.stdout.splitlines())} packages", Colors.GREEN)
            self._print(f"  Lock hash: {lock_hash}", Colors.CYAN)
            self._print(f"  Saved to: {self.lock_file}", Colors.CYAN)

            return True

        except subprocess.CalledProcessError as e:
            self._print(f"  ❌ Failed to freeze: {e.stderr}", Colors.RED)
            return False

    def get_local_packages(self) -> Dict[str, str]:
        """Get locally installed packages as dict {name: version}"""
        try:
            result = subprocess.run(
                "source venv/bin/activate && pip freeze",
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )

            packages = {}
            for line in result.stdout.splitlines():
                if '==' in line:
                    name, version = line.split('==', 1)
                    packages[name.strip()] = version.strip()

            return packages

        except:
            return {}

    def get_remote_packages(self, worker: dict) -> Dict[str, str]:
        """Get packages installed on remote worker"""
        success, stdout, stderr = self._run_remote_command(
            worker,
            "source venv/bin/activate && pip freeze",
            timeout=30
        )

        if not success:
            return {}

        packages = {}
        for line in stdout.splitlines():
            if '==' in line:
                name, version = line.split('==', 1)
                packages[name.strip()] = version.strip()

        return packages

    def check_consistency(self) -> bool:
        """
        Check if all workers have consistent dependencies with local.
        """
        self._print("🔍 Checking dependency consistency...\n", Colors.BLUE)

        local_packages = self.get_local_packages()
        all_consistent = True

        for worker in self.workers:
            worker_id = worker['id']
            self._print(f"  Checking {worker_id}...", Colors.CYAN)

            if worker.get('connection_type') == 'local':
                self._print(f"    ✅ Local worker (skipped)", Colors.GREEN)
                continue

            remote_packages = self.get_remote_packages(worker)

            if not remote_packages:
                self._print(f"    ❌ Could not get package list", Colors.RED)
                all_consistent = False
                continue

            # Compare versions
            mismatches = []
            for pkg, local_ver in local_packages.items():
                remote_ver = remote_packages.get(pkg)
                if remote_ver != local_ver:
                    mismatches.append((pkg, local_ver, remote_ver))

            # Check for missing packages
            missing = set(local_packages.keys()) - set(remote_packages.keys())
            extra = set(remote_packages.keys()) - set(local_packages.keys())

            if mismatches or missing or extra:
                all_consistent = False
                self._print(f"    ⚠️  Inconsistencies found:", Colors.YELLOW)

                if mismatches:
                    self._print(f"      Version mismatches:", Colors.YELLOW)
                    for pkg, local_ver, remote_ver in mismatches[:5]:  # Show first 5
                        self._print(f"        {pkg}: {remote_ver or 'missing'} -> {local_ver}", Colors.YELLOW)
                    if len(mismatches) > 5:
                        self._print(f"        ... and {len(mismatches) - 5} more", Colors.YELLOW)

                if missing:
                    self._print(f"      Missing packages: {len(missing)}", Colors.YELLOW)

                if extra:
                    self._print(f"      Extra packages: {len(extra)}", Colors.YELLOW)
            else:
                self._print(f"    ✅ Consistent ({len(local_packages)} packages)", Colors.GREEN)

        print()
        if all_consistent:
            self._print("✅ All workers have consistent dependencies", Colors.GREEN)
        else:
            self._print("⚠️  Some workers have dependency mismatches", Colors.YELLOW)
            self._print("Run: ./mlx-cli sync-deps to fix", Colors.CYAN)

        return all_consistent

    def sync_to_workers(self, use_lock: bool = True, force: bool = False) -> bool:
        """
        Sync dependencies to all workers.

        Args:
            use_lock: Use requirements.lock instead of requirements.txt
            force: Force reinstall even if up to date
        """
        requirements_file = self.lock_file if use_lock else "requirements.txt"

        if use_lock and not self.lock_file.exists():
            self._print(f"❌ {self.lock_file} not found. Run: ./mlx-cli freeze-deps", Colors.RED)
            return False

        self._print(f"🔄 Syncing dependencies to workers...", Colors.BLUE)
        self._print(f"Using: {requirements_file}\n", Colors.CYAN)

        # Copy requirements file to workers first
        for worker in self.workers:
            worker_id = worker['id']

            if worker.get('connection_type') == 'local':
                self._print(f"  ⏭️  {worker_id} (local - skipped)", Colors.YELLOW)
                continue

            self._print(f"  Syncing {worker_id}...", Colors.CYAN)

            # Copy requirements file
            ssh_user = worker.get('ssh_user', os.getenv('USER'))
            host = worker['host']
            work_dir = worker.get('work_dir', '~/mlx-distributed-inference')
            ssh_target = f"{ssh_user}@{host}" if ssh_user else host

            # Copy requirements file
            scp_cmd = f"scp {requirements_file} {ssh_target}:{work_dir}/"
            result = subprocess.run(scp_cmd, shell=True, capture_output=True)

            if result.returncode != 0:
                self._print(f"    ❌ Failed to copy requirements file", Colors.RED)
                continue

            # Install dependencies
            install_cmd = (
                f"source venv/bin/activate && "
                f"pip install {'--force-reinstall' if force else ''} "
                f"-r {requirements_file.name}"
            )

            self._print(f"    Installing packages...", Colors.CYAN)
            success, stdout, stderr = self._run_remote_command(
                worker,
                install_cmd,
                timeout=600  # 10 minutes
            )

            if success:
                self._print(f"    ✅ Dependencies installed", Colors.GREEN)
            else:
                self._print(f"    ❌ Installation failed", Colors.RED)
                if stderr:
                    self._print(f"    Error: {stderr[:200]}", Colors.RED)

        print()
        self._print("✅ Dependency sync complete", Colors.GREEN)
        self._print("Restart workers to use updated dependencies", Colors.CYAN)
        return True

    def create_snapshot(self, name: Optional[str] = None) -> bool:
        """
        Create a snapshot of current dependencies for rollback.
        """
        import datetime

        if name is None:
            name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        snapshot_file = self.freeze_dir / f"snapshot_{name}.txt"

        self._print(f"📸 Creating dependency snapshot: {name}", Colors.CYAN)

        try:
            result = subprocess.run(
                "source venv/bin/activate && pip freeze",
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )

            with open(snapshot_file, 'w') as f:
                f.write(result.stdout)

            self._print(f"  ✅ Snapshot saved: {snapshot_file}", Colors.GREEN)
            return True

        except Exception as e:
            self._print(f"  ❌ Failed: {e}", Colors.RED)
            return False

    def list_snapshots(self):
        """List available dependency snapshots"""
        snapshots = sorted(self.freeze_dir.glob("snapshot_*.txt"))

        if not snapshots:
            self._print("No snapshots found", Colors.YELLOW)
            return

        self._print("Available snapshots:", Colors.CYAN)
        for snapshot in snapshots:
            name = snapshot.stem.replace("snapshot_", "")
            size = len(snapshot.read_text().splitlines())
            self._print(f"  {name} ({size} packages)", Colors.NC)

    def restore_snapshot(self, name: str) -> bool:
        """
        Restore dependencies from a snapshot.
        """
        snapshot_file = self.freeze_dir / f"snapshot_{name}.txt"

        if not snapshot_file.exists():
            self._print(f"❌ Snapshot not found: {name}", Colors.RED)
            self.list_snapshots()
            return False

        self._print(f"🔄 Restoring snapshot: {name}", Colors.CYAN)

        # Copy snapshot to requirements.lock
        import shutil
        shutil.copy(snapshot_file, self.lock_file)

        self._print(f"  ✅ Restored to {self.lock_file}", Colors.GREEN)
        self._print(f"Run: ./mlx-cli sync-deps to apply to workers", Colors.CYAN)
        return True

    def upgrade_package(self, package: str, version: Optional[str] = None) -> bool:
        """
        Upgrade a specific package across all workers.
        """
        if version:
            pkg_spec = f"{package}=={version}"
            self._print(f"📦 Upgrading {package} to version {version}", Colors.CYAN)
        else:
            pkg_spec = package
            self._print(f"📦 Upgrading {package} to latest", Colors.CYAN)

        # Upgrade locally first
        self._print("  Upgrading locally...", Colors.CYAN)
        try:
            subprocess.run(
                f"source venv/bin/activate && pip install --upgrade {pkg_spec}",
                shell=True,
                check=True,
                capture_output=True
            )
            self._print("    ✅ Local upgrade complete", Colors.GREEN)
        except:
            self._print("    ❌ Local upgrade failed", Colors.RED)
            return False

        # Update lock file
        self.freeze_local()

        # Sync to workers
        return self.sync_to_workers(use_lock=True)


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Manage Python dependencies across worker nodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Freeze local dependencies to lock file
  ./scripts/manage_dependencies.py freeze

  # Check consistency across all workers
  ./scripts/manage_dependencies.py check

  # Sync dependencies to all workers
  ./scripts/manage_dependencies.py sync

  # Force reinstall all dependencies
  ./scripts/manage_dependencies.py sync --force

  # Create snapshot for rollback
  ./scripts/manage_dependencies.py snapshot

  # Restore from snapshot
  ./scripts/manage_dependencies.py restore snapshot_20250110_120000

  # Upgrade specific package
  ./scripts/manage_dependencies.py upgrade mlx-lm --version 0.5.0
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Freeze command
    subparsers.add_parser('freeze', help='Freeze local dependencies to requirements.lock')

    # Check command
    subparsers.add_parser('check', help='Check dependency consistency across workers')

    # Sync command
    sync_parser = subparsers.add_parser('sync', help='Sync dependencies to all workers')
    sync_parser.add_argument('--force', action='store_true', help='Force reinstall')
    sync_parser.add_argument('--no-lock', action='store_true', help='Use requirements.txt instead of lock')

    # Snapshot command
    snapshot_parser = subparsers.add_parser('snapshot', help='Create dependency snapshot')
    snapshot_parser.add_argument('--name', help='Snapshot name (default: timestamp)')

    # List snapshots
    subparsers.add_parser('list-snapshots', help='List available snapshots')

    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore from snapshot')
    restore_parser.add_argument('name', help='Snapshot name to restore')

    # Upgrade command
    upgrade_parser = subparsers.add_parser('upgrade', help='Upgrade a package')
    upgrade_parser.add_argument('package', help='Package name')
    upgrade_parser.add_argument('--version', help='Specific version (default: latest)')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    manager = DependencyManager()

    if args.command == 'freeze':
        success = manager.freeze_local()
    elif args.command == 'check':
        success = manager.check_consistency()
    elif args.command == 'sync':
        success = manager.sync_to_workers(
            use_lock=not args.no_lock,
            force=args.force
        )
    elif args.command == 'snapshot':
        success = manager.create_snapshot(args.name)
    elif args.command == 'list-snapshots':
        manager.list_snapshots()
        success = True
    elif args.command == 'restore':
        success = manager.restore_snapshot(args.name)
    elif args.command == 'upgrade':
        success = manager.upgrade_package(args.package, args.version)
    else:
        parser.print_help()
        return 1

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
