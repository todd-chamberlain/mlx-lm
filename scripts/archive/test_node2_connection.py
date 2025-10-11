#!/usr/bin/env python3
"""
Test connection to mac-studio-2 and help troubleshoot worker startup
"""

import subprocess
import socket
import time

def test_ssh_connection():
    """Test SSH connection to mac-studio-2"""
    print("🔍 Testing SSH connection to mac-studio-2...")
    try:
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "10.201.20.12", "echo 'SSH connection successful'"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ SSH connection successful")
            return True
        else:
            print(f"❌ SSH connection failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ SSH connection error: {e}")
        return False

def test_network_connectivity():
    """Test basic network connectivity"""
    print("🔍 Testing network connectivity to mac-studio-2...")
    try:
        # Test ping
        result = subprocess.run(
            ["ping", "-c", "3", "10.201.20.12"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            print("✅ Network ping successful")
            return True
        else:
            print(f"❌ Network ping failed")
            return False
    except Exception as e:
        print(f"❌ Network test error: {e}")
        return False

def check_coordinator_port():
    """Check if coordinator port is accessible from mac-studio-2"""
    print("🔍 Testing coordinator port accessibility...")
    try:
        # Test if we can connect to coordinator port from this machine
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 50051))
        sock.close()
        
        if result == 0:
            print("✅ Coordinator port 50051 is accessible locally")
            return True
        else:
            print("❌ Coordinator port 50051 is not accessible locally")
            return False
    except Exception as e:
        print(f"❌ Port test error: {e}")
        return False

def get_remote_worker_status():
    """Check if worker process is running on mac-studio-2"""
    print("🔍 Checking worker process on mac-studio-2...")
    try:
        result = subprocess.run([
            "ssh", "10.201.20.12", 
            "ps aux | grep 'python.*worker.main' | grep -v grep"
        ], capture_output=True, text=True, timeout=10)
        
        if result.stdout.strip():
            print("✅ Worker process found on mac-studio-2:")
            print(f"   {result.stdout.strip()}")
            return True
        else:
            print("❌ No worker process found on mac-studio-2")
            return False
    except Exception as e:
        print(f"❌ Remote process check error: {e}")
        return False

def show_startup_commands():
    """Show the exact commands to run on mac-studio-2"""
    print("\n📋 Commands to run on mac-studio-2:")
    print("=" * 50)
    print("ssh 10.201.20.12")
    print("cd ~/mlx-distributed-inference")
    print("source venv/bin/activate")
    print("python -m worker.main --worker-id worker-studio2 --port 8080 --coordinator 10.201.20.10:50051")
    print("=" * 50)

def main():
    print("🚀 Testing mac-studio-2 connection and worker setup...")
    print("=" * 60)
    
    # Test basic connectivity
    network_ok = test_network_connectivity()
    ssh_ok = test_ssh_connection()
    coordinator_ok = check_coordinator_port()
    worker_running = get_remote_worker_status()
    
    print("\n📊 Summary:")
    print(f"Network connectivity: {'✅' if network_ok else '❌'}")
    print(f"SSH connection: {'✅' if ssh_ok else '❌'}")
    print(f"Coordinator accessible: {'✅' if coordinator_ok else '❌'}")
    print(f"Worker running on node2: {'✅' if worker_running else '❌'}")
    
    if not worker_running and ssh_ok:
        show_startup_commands()
        print("\n💡 If the worker fails to start, check:")
        print("   - Virtual environment is activated")
        print("   - Project directory exists: ~/mlx-distributed-inference")
        print("   - Coordinator IP is reachable: 10.201.20.10:50051")
        print("   - No firewall blocking port 50051")
    
    if not ssh_ok:
        print("\n💡 SSH troubleshooting:")
        print("   - Check if mac-studio-2 is powered on")
        print("   - Verify IP address: 10.201.20.12")
        print("   - Check SSH service is running on mac-studio-2")
        print("   - Verify SSH keys or password authentication")

if __name__ == "__main__":
    main()