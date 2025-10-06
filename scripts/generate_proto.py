#!/usr/bin/env python3
"""
Script to generate Python gRPC stubs from protocol buffer definitions.
"""

import subprocess
import sys
import os
from pathlib import Path

def generate_grpc_stubs():
    """Generate Python gRPC stubs from .proto files."""
    
    # Ensure we're in the project root
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Create proto output directory if it doesn't exist
    proto_dir = project_root / "proto"
    proto_dir.mkdir(exist_ok=True)
    
    # Generate Python stubs
    proto_file = "proto/inference.proto"
    
    if not Path(proto_file).exists():
        print(f"Error: {proto_file} not found")
        sys.exit(1)
    
    cmd = [
        sys.executable, "-m", "grpc_tools.protoc",
        f"--proto_path={project_root}",
        f"--python_out={project_root}",
        f"--grpc_python_out={project_root}",
        proto_file
    ]
    
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Successfully generated gRPC stubs:")
        print(f"  - proto/inference_pb2.py")
        print(f"  - proto/inference_pb2_grpc.py")
        
        # Fix import paths in generated files
        fix_import_paths()
        
    except subprocess.CalledProcessError as e:
        print(f"Error generating gRPC stubs: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: grpcio-tools not installed. Install with: pip install grpcio-tools")
        sys.exit(1)

def fix_import_paths():
    """Fix import paths in generated gRPC files."""
    
    # Fix inference_pb2_grpc.py imports
    grpc_file = Path("proto/inference_pb2_grpc.py")
    if grpc_file.exists():
        content = grpc_file.read_text()
        # Replace relative import with absolute import
        content = content.replace(
            "import inference_pb2 as inference__pb2",
            "from . import inference_pb2 as inference__pb2"
        )
        grpc_file.write_text(content)
        print("Fixed import paths in inference_pb2_grpc.py")

if __name__ == "__main__":
    generate_grpc_stubs()