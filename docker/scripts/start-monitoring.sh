#!/bin/bash
set -e

# MLX Monitoring Stack startup script

echo "Starting MLX Monitoring Stack..."

# Create necessary directories
mkdir -p /var/lib/prometheus/data
mkdir -p /var/lib/grafana/data
mkdir -p /var/log/prometheus
mkdir -p /var/log/grafana

# Set permissions
chown -R monitoring:monitoring /var/lib/prometheus /var/lib/grafana /var/log/prometheus /var/log/grafana

# Prometheus configuration
export PROMETHEUS_CONFIG=${PROMETHEUS_CONFIG:-/etc/prometheus/prometheus.yml}
export PROMETHEUS_DATA_DIR=${PROMETHEUS_DATA_DIR:-/var/lib/prometheus/data}
export PROMETHEUS_RETENTION=${PROMETHEUS_RETENTION:-15d}
export PROMETHEUS_STORAGE_TSDB_PATH=${PROMETHEUS_DATA_DIR}

# Grafana configuration
export GF_PATHS_DATA=${GF_PATHS_DATA:-/var/lib/grafana/data}
export GF_PATHS_LOGS=${GF_PATHS_LOGS:-/var/log/grafana}
export GF_PATHS_PLUGINS=${GF_PATHS_PLUGINS:-/var/lib/grafana/plugins}
export GF_PATHS_PROVISIONING=${GF_PATHS_PROVISIONING:-/etc/grafana/provisioning}

# Grafana security settings
export GF_SECURITY_ADMIN_USER=${GF_SECURITY_ADMIN_USER:-admin}
export GF_SECURITY_ADMIN_PASSWORD=${GF_SECURITY_ADMIN_PASSWORD:-admin}
export GF_SECURITY_SECRET_KEY=${GF_SECURITY_SECRET_KEY:-$(openssl rand -hex 16)}

# Grafana server settings
export GF_SERVER_HTTP_PORT=${GF_SERVER_HTTP_PORT:-3000}
export GF_SERVER_DOMAIN=${GF_SERVER_DOMAIN:-localhost}

# Anonymous access for development
export GF_AUTH_ANONYMOUS_ENABLED=${GF_AUTH_ANONYMOUS_ENABLED:-true}
export GF_AUTH_ANONYMOUS_ORG_ROLE=${GF_AUTH_ANONYMOUS_ORG_ROLE:-Viewer}

echo "Prometheus configuration: $PROMETHEUS_CONFIG"
echo "Prometheus data directory: $PROMETHEUS_DATA_DIR"
echo "Prometheus retention: $PROMETHEUS_RETENTION"
echo "Grafana data directory: $GF_PATHS_DATA"
echo "Grafana admin user: $GF_SECURITY_ADMIN_USER"

# Validate Prometheus configuration
if [ -f "$PROMETHEUS_CONFIG" ]; then
    echo "Validating Prometheus configuration..."
    /opt/prometheus/promtool check config "$PROMETHEUS_CONFIG"
else
    echo "Warning: Prometheus configuration file not found at $PROMETHEUS_CONFIG"
fi

# Start Prometheus in background
echo "Starting Prometheus..."
/opt/prometheus/prometheus \
    --config.file="$PROMETHEUS_CONFIG" \
    --storage.tsdb.path="$PROMETHEUS_DATA_DIR" \
    --storage.tsdb.retention.time="$PROMETHEUS_RETENTION" \
    --web.console.libraries=/opt/prometheus/console_libraries \
    --web.console.templates=/opt/prometheus/consoles \
    --web.listen-address=0.0.0.0:9090 \
    --web.enable-lifecycle \
    --log.level=info &

PROMETHEUS_PID=$!

# Wait a moment for Prometheus to start
sleep 5

# Start Grafana in background
echo "Starting Grafana..."
/opt/grafana/bin/grafana-server \
    --homepath=/opt/grafana \
    --config=/etc/grafana/grafana.ini &

GRAFANA_PID=$!

# Function to handle shutdown
shutdown() {
    echo "Shutting down monitoring services..."
    kill $PROMETHEUS_PID $GRAFANA_PID 2>/dev/null || true
    wait $PROMETHEUS_PID $GRAFANA_PID 2>/dev/null || true
    echo "Monitoring services stopped"
    exit 0
}

# Set up signal handlers
trap shutdown SIGTERM SIGINT

# Wait for services to be ready
echo "Waiting for services to be ready..."
timeout=60
while [ $timeout -gt 0 ]; do
    prometheus_ready=false
    grafana_ready=false
    
    # Check Prometheus
    if curl -s http://localhost:9090/-/ready >/dev/null 2>&1; then
        prometheus_ready=true
    fi
    
    # Check Grafana
    if curl -s http://localhost:3000/api/health >/dev/null 2>&1; then
        grafana_ready=true
    fi
    
    if [ "$prometheus_ready" = true ] && [ "$grafana_ready" = true ]; then
        echo "All monitoring services are ready!"
        break
    fi
    
    echo "Waiting for services to be ready... (Prometheus: $prometheus_ready, Grafana: $grafana_ready)"
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    echo "Warning: Timeout waiting for services to be ready"
fi

echo "Monitoring stack started successfully"
echo "Prometheus: http://localhost:9090"
echo "Grafana: http://localhost:3000 (admin/$GF_SECURITY_ADMIN_PASSWORD)"

# Keep the script running and wait for processes
wait $PROMETHEUS_PID $GRAFANA_PID