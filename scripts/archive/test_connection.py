#!/usr/bin/env python3

import grpc
import sys

def test_connection(host, port):
    try:
        channel = grpc.insecure_channel(f'{host}:{port}')
        # Try to connect with a timeout
        grpc.channel_ready_future(channel).result(timeout=5)
        print(f"✅ Successfully connected to {host}:{port}")
        channel.close()
        return True
    except grpc.FutureTimeoutError:
        print(f"❌ Connection timeout to {host}:{port}")
        return False
    except Exception as e:
        print(f"❌ Connection failed to {host}:{port}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python test_connection.py <host> <port>")
        sys.exit(1)
    
    host = sys.argv[1]
    port = sys.argv[2]
    
    success = test_connection(host, port)
    sys.exit(0 if success else 1)