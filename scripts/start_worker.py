#!/usr/bin/env python3
"""
MLX Worker Startup Script

Simple script to start an MLX worker with common configurations.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from worker.main import main

if __name__ == '__main__':
    # Set default environment variables if not set
    os.environ.setdefault('WORKER_ID', f'worker-{os.uname().nodename}')
    os.environ.setdefault('WORKER_PORT', '8080')
    os.environ.setdefault('COORDINATOR_HOST', 'localhost:50051')
    os.environ.setdefault('MODEL_PATH', '/models/default')
    os.environ.setdefault('MODEL_NAME', 'default')
    os.environ.setdefault('MAX_CONCURRENT', '4')
    os.environ.setdefault('LOG_LEVEL', 'INFO')
    
    # Run the worker
    asyncio.run(main())