#!/usr/bin/env python3
"""
Deploy MLX Worker Software to Remote Nodes

Manages deployment, updates, and synchronization of worker software
across all configured nodes.
"""

import os
import sys
import yaml
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict

# Colors
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


class NodeDeployer:
    """Deploys and manages worker software on remote nodes"""

    def __init__(self, config_path: str = "config/nodes.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.workers = self.config.get('workers', [])

        # Files/directories to sync
        self.sync_items = [
            'worker/',
            'common/',
            'proto/',
            'requirements.txt',
            'config/worker_models.yaml',
        ]

        # Files to exclude
        self.excludes = [
            '*.pyc',
            '__pycache__/',
            '*.log',
            'logs/',
            'venv/',
            '.git/',
            '.DS_Store',
        ]

    def _load_config(self) -> dict:
        """Load node configuration"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _print(self, msg: str, color: str = Colors.NC):
        """Print colored message"""
        print(f"{color}{msg}{Colors.NC}")

    def _build_rsync_excludes(self) -> List[str]:
        """Build rsync exclude arguments"""
        excludes = []
        for pattern in self.excludes:
            excludes.extend(['--exclude', pattern])
        return excludes

    def setup_node(self, worker: dict, force: bool = False) -> bool:
        """
        Initial setup of a new worker node.
        Creates directory structure, virtual environment, installs dependencies.
        """
        worker_id = worker['id']
        host = worker['host']
        connection_type = worker.get('connection_type')

        if connection_type == 'local':
            self._print(f"  Skipping {worker_id} (local node - already set up)", Colors.YELLOW)
            return True

        ssh_user = worker.get('ssh_user', os.getenv('USER'))
        work_dir = worker.get('work_dir', '~/mlx-distributed-inference')
        ssh_target = f"{ssh_user}@{host}" if ssh_user else host

        self._print(f"\n{'='*60}", Colors.BLUE)
        self._print(f"Setting up {worker_id} on {host}", Colors.BLUE)
        self._print(f"{'='*60}", Colors.BLUE)

        # Check if already set up
        if not force:
            check_cmd = f"ssh {ssh_target} 'test -d {work_dir} && echo exists'"
            result = subprocess.run(check_cmd, shell=True, capture_output=True, text=True)
            if 'exists' in result.stdout:
                self._print(f"  ⚠️  {work_dir} already exists. Use --force to reinstall", Colors.YELLOW)
                return True

        steps = [
            ("Creating work directory", f"ssh {ssh_target} 'mkdir -p {work_dir}'"),
            ("Creating logs directory", f"ssh {ssh_target} 'mkdir -p {work_dir}/logs'"),
            ("Creating config directory", f"ssh {ssh_target} 'mkdir -p {work_dir}/config'"),
            ("Checking Python 3", f"ssh {ssh_target} 'python3 --version'"),
        ]

        for step_name, cmd in steps:
            self._print(f"  [{step_name}]...", Colors.CYAN)
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                self._print(f"    ❌ Failed: {result.stderr}", Colors.RED)
                return False
            self._print(f"    ✅ Done", Colors.GREEN)

        # Create virtual environment
        self._print(f"  [Creating virtual environment]...", Colors.CYAN)
        venv_cmd = f"ssh {ssh_target} 'cd {work_dir} && python3 -m venv venv'"
        result = subprocess.run(venv_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            self._print(f"    ❌ Failed: {result.stderr}", Colors.RED)
            return False
        self._print(f"    ✅ Virtual environment created", Colors.GREEN)

        self._print(f"\n  ✅ {worker_id} setup complete", Colors.GREEN)
        return True

    def deploy_code(self, worker: dict, dry_run: bool = False) -> bool:
        """
        Deploy code to a worker node using rsync.
        """
        worker_id = worker['id']
        host = worker['host']
        connection_type = worker.get('connection_type')

        if connection_type == 'local':
            self._print(f"  Skipping {worker_id} (local node)", Colors.YELLOW)
            return True

        ssh_user = worker.get('ssh_user', os.getenv('USER'))
        work_dir = worker.get('work_dir', '~/mlx-distributed-inference')
        ssh_target = f"{ssh_user}@{host}" if ssh_user else host

        self._print(f"\n  Deploying to {worker_id} ({host})...", Colors.CYAN)

        # Build rsync command
        rsync_cmd = [
            'rsync',
            '-avz',  # archive, verbose, compress
            '--delete',  # delete files not in source
            '--progress' if not dry_run else '--dry-run',
        ]
        rsync_cmd.extend(self._build_rsync_excludes())
        rsync_cmd.extend(self.sync_items)
        rsync_cmd.append(f"{ssh_target}:{work_dir}/")

        # Execute rsync
        try:
            result = subprocess.run(
                rsync_cmd,
                check=True,
                capture_output=True,
                text=True
            )

            if dry_run:
                self._print(f"    [DRY RUN] Would sync:", Colors.YELLOW)
                print(result.stdout)
            else:
                self._print(f"    ✅ Code deployed", Colors.GREEN)

            return True

        except subprocess.CalledProcessError as e:
            self._print(f"    ❌ rsync failed: {e.stderr}", Colors.RED)
            return False

    def install_dependencies(self, worker: dict) -> bool:
        """
        Install Python dependencies on remote node.
        """
        worker_id = worker['id']
        host = worker['host']
        connection_type = worker.get('connection_type')

        if connection_type == 'local':
            self._print(f"  Skipping {worker_id} (local node)", Colors.YELLOW)
            return True

        ssh_user = worker.get('ssh_user', os.getenv('USER'))
        work_dir = worker.get('work_dir', '~/mlx-distributed-inference')
        ssh_target = f"{ssh_user}@{host}" if ssh_user else host

        self._print(f"\n  Installing dependencies on {worker_id}...", Colors.CYAN)

        # Install requirements
        pip_cmd = (
            f"ssh {ssh_target} 'cd {work_dir} && "
            f"source venv/bin/activate && "
            f"pip install --upgrade pip && "
            f"pip install -r requirements.txt'"
        )

        try:
            result = subprocess.run(
                pip_cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            self._print(f"    ✅ Dependencies installed", Colors.GREEN)
            return True
        except subprocess.TimeoutExpired:
            self._print(f"    ❌ Timeout installing dependencies", Colors.RED)
            return False
        except subprocess.CalledProcessError as e:
            self._print(f"    ❌ Failed: {e.stderr}", Colors.RED)
            return False

    def restart_worker(self, worker: dict) -> bool:
        """
        Restart worker service on remote node.
        """
        worker_id = worker['id']
        host = worker['host']
        connection_type = worker.get('connection_type')

        if connection_type == 'local':
            self._print(f"  Restart {worker_id} manually (local node)", Colors.YELLOW)
            return True

        ssh_user = worker.get('ssh_user', os.getenv('USER'))
        ssh_target = f"{ssh_user}@{host}" if ssh_user else host

        self._print(f"\n  Restarting {worker_id}...", Colors.CYAN)

        # Kill existing worker
        kill_cmd = f"ssh {ssh_target} 'pkill -f \"worker.main.*{worker_id}\" || true'"
        subprocess.run(kill_cmd, shell=True, capture_output=True)

        self._print(f"    ✅ Worker will be started by orchestrator", Colors.GREEN)
        return True

    def check_version(self, worker: dict) -> bool:
        """
        Check software version on remote node.
        """
        worker_id = worker['id']
        host = worker['host']
        connection_type = worker.get('connection_type')

        if connection_type == 'local':
            return True

        ssh_user = worker.get('ssh_user', os.getenv('USER'))
        work_dir = worker.get('work_dir', '~/mlx-distributed-inference')
        ssh_target = f"{ssh_user}@{host}" if ssh_user else host

        # Get git commit hash if available
        git_cmd = f"ssh {ssh_target} 'cd {work_dir} && git rev-parse --short HEAD 2>/dev/null || echo unknown'"
        result = subprocess.run(git_cmd, shell=True, capture_output=True, text=True)

        remote_version = result.stdout.strip() if result.returncode == 0 else "unknown"

        # Get local version
        local_result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                                     capture_output=True, text=True)
        local_version = local_result.stdout.strip() if local_result.returncode == 0 else "unknown"

        in_sync = remote_version == local_version
        status = "✅" if in_sync else "⚠️"

        self._print(f"  {status} {worker_id}: {remote_version} (local: {local_version})",
                   Colors.GREEN if in_sync else Colors.YELLOW)

        return in_sync

    def deploy_all(self, setup: bool = False, dry_run: bool = False,
                   skip_deps: bool = False):
        """
        Deploy to all configured workers.
        """
        self._print("🚀 Deploying to All Worker Nodes", Colors.BLUE)
        self._print(f"{'='*60}\n", Colors.BLUE)

        results = {
            'setup': [],
            'deploy': [],
            'deps': [],
        }

        for worker in self.workers:
            worker_id = worker['id']

            # Setup (if requested)
            if setup:
                success = self.setup_node(worker, force=False)
                results['setup'].append((worker_id, success))

            # Deploy code
            success = self.deploy_code(worker, dry_run=dry_run)
            results['deploy'].append((worker_id, success))

            # Install dependencies (if not skipped)
            if not skip_deps and not dry_run:
                success = self.install_dependencies(worker)
                results['deps'].append((worker_id, success))

        # Print summary
        self._print(f"\n{'='*60}", Colors.BLUE)
        self._print("Deployment Summary", Colors.BLUE)
        self._print(f"{'='*60}\n", Colors.BLUE)

        if results['setup']:
            self._print("Setup:", Colors.CYAN)
            for worker_id, success in results['setup']:
                status = "✅" if success else "❌"
                self._print(f"  {status} {worker_id}",
                           Colors.GREEN if success else Colors.RED)

        self._print("\nCode Deployment:", Colors.CYAN)
        for worker_id, success in results['deploy']:
            status = "✅" if success else "❌"
            self._print(f"  {status} {worker_id}",
                       Colors.GREEN if success else Colors.RED)

        if results['deps']:
            self._print("\nDependencies:", Colors.CYAN)
            for worker_id, success in results['deps']:
                status = "✅" if success else "❌"
                self._print(f"  {status} {worker_id}",
                           Colors.GREEN if success else Colors.RED)

        # Check if we should restart
        if not dry_run:
            self._print(f"\n{'='*60}", Colors.YELLOW)
            self._print("⚠️  Workers need to be restarted", Colors.YELLOW)
            self._print("Run: ./mlx-cli stop-hybrid && ./mlx-cli start-hybrid", Colors.YELLOW)
            self._print(f"{'='*60}\n", Colors.YELLOW)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Deploy MLX worker software to remote nodes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initial setup of all nodes
  ./scripts/deploy_to_nodes.py --setup

  # Deploy code updates to all nodes
  ./scripts/deploy_to_nodes.py

  # Dry run (see what would be deployed)
  ./scripts/deploy_to_nodes.py --dry-run

  # Deploy without updating dependencies
  ./scripts/deploy_to_nodes.py --skip-deps

  # Check version sync status
  ./scripts/deploy_to_nodes.py --check-version
        """
    )

    parser.add_argument('--setup', action='store_true',
                       help='Initial setup of new nodes')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deployed without doing it')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--check-version', action='store_true',
                       help='Check version sync status across nodes')
    parser.add_argument('--config', default='config/nodes.yaml',
                       help='Path to nodes configuration file')

    args = parser.parse_args()

    deployer = NodeDeployer(config_path=args.config)

    if args.check_version:
        print(f"\n{Colors.BLUE}Checking Version Sync{Colors.NC}")
        print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")
        all_synced = all(deployer.check_version(w) for w in deployer.workers)
        sys.exit(0 if all_synced else 1)
    else:
        deployer.deploy_all(
            setup=args.setup,
            dry_run=args.dry_run,
            skip_deps=args.skip_deps
        )


if __name__ == "__main__":
    main()
