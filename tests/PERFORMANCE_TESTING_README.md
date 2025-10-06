# Performance Testing and Load Testing Suite

Comprehensive performance testing framework for MLX distributed inference system with benchmarking, load testing, and regression detection capabilities.

## Overview

The performance testing suite provides:

- **Performance Benchmarking**: Measure latency, throughput, and resource usage
- **Load Testing**: Simulate realistic traffic patterns and user behavior
- **Stress Testing**: Find system capacity limits and breaking points
- **Regression Detection**: Automatically detect performance degradations
- **Trend Analysis**: Track performance metrics over time
- **Reporting**: Generate comprehensive HTML reports with visualizations

## Components

### 1. Performance Benchmarking (`test_performance_benchmarking.py`)

Comprehensive performance tests covering:

- **Single Request Performance**
  - Latency measurement (min, mean, median, p95, p99, max)
  - Throughput measurement (tokens/sec, bytes/sec)
  - Resource usage tracking (CPU, memory)

- **Concurrent Request Handling**
  - 10 concurrent users
  - 50 concurrent users
  - 100+ concurrent users
  - Scalability testing

- **Sustained Load Performance**
  - 5-minute sustained load
  - 15-minute sustained load
  - 1-hour sustained load (configurable)
  - Varying request rates

- **Burst Traffic Handling**
  - Sudden traffic spikes
  - Recovery after spikes
  - Queue management

- **Stress and Capacity Testing**
  - Find capacity limits
  - Memory usage under load
  - Graceful degradation verification

#### Usage

```python
from tests.test_performance_benchmarking import PerformanceTestRunner

# Create runner
runner = PerformanceTestRunner(base_url="http://localhost:8000")

# Run single request test
def mock_request():
    # Your request logic
    return {"status": "success", "tokens": 50}

latencies = []
successes = 0

for _ in range(100):
    latency, result = runner.measure_latency(mock_request)
    latencies.append(latency)
    if result["status"] == "success":
        successes += 1

# Calculate metrics
metrics = runner.calculate_metrics(
    "my_test",
    latencies,
    successes,
    0,
    duration=10.0,
    tokens=5000
)

runner.print_summary(metrics)
```

#### Running Tests

```bash
# Run all performance tests
pytest tests/test_performance_benchmarking.py -v -s

# Run specific test class
pytest tests/test_performance_benchmarking.py::TestConcurrentRequestHandling -v -s

# Run with custom markers
pytest tests/test_performance_benchmarking.py -m "not slow" -v
```

### 2. Performance Regression Detection (`performance_regression.py`)

Automated regression detection with baseline comparison:

- **Baseline Management**
  - Create and store performance baselines
  - Load historical baselines
  - Update baselines as needed

- **Regression Detection**
  - Compare current metrics against baseline
  - Configurable thresholds (RPS, latency, error rate)
  - Identify both regressions and improvements
  - Percentage change calculation

- **Trend Analysis**
  - Track metrics over time
  - Identify trends (increasing, decreasing, stable)
  - Historical data visualization
  - Trend plots with matplotlib

- **Reporting**
  - HTML report generation
  - Regression highlights
  - Improvement tracking
  - Metric comparison tables

#### Usage

```python
from tests.performance_regression import (
    PerformanceRegressionDetector,
    PerformanceTrendAnalyzer,
    PerformanceReporter
)

# Initialize detector
detector = PerformanceRegressionDetector()

# Current test metrics
current_metrics = {
    "requests_per_second": 95.0,
    "latency_median": 55.0,
    "latency_p95": 110.0,
    "latency_p99": 160.0,
    "error_rate": 0.02,
    "cpu_usage_percent": 50.0,
    "memory_usage_mb": 550.0
}

# Detect regression
result = detector.detect_regression("my_test", current_metrics)
print(result.get_summary())

# Analyze trends
analyzer = PerformanceTrendAnalyzer()
analyzer.add_result("my_test", current_metrics)
print(analyzer.generate_trend_report("my_test"))

# Generate plots
analyzer.plot_trends("my_test", "performance_trends.png")

# Generate HTML report
reporter = PerformanceReporter()
reporter.generate_report([current_metrics], "performance_report.html")
```

#### Baseline Management

```python
# Create baseline from current metrics
baseline = detector.create_baseline("my_test", current_metrics)

# Load existing baselines
detector.load_baselines()

# Save baselines
detector.save_baselines()
```

### 3. Load Testing Scenarios (`load_testing_scenarios.py`)

Realistic load testing with various traffic patterns:

- **Constant Load Scenario**
  - Fixed number of concurrent users
  - Configurable think time
  - Sustained request rate

- **Ramp-Up Scenario**
  - Gradual load increase
  - Configurable ramp-up duration
  - Smooth transition to full load

- **Spike Scenario**
  - Sudden traffic bursts
  - Recovery testing
  - Normal → Spike → Normal pattern

- **Stress Test Scenario**
  - Find system breaking point
  - Gradual load increase until failure
  - Capacity limit identification

#### Usage

```python
from tests.load_testing_scenarios import (
    LoadTestConfig,
    ConstantLoadScenario,
    RampUpScenario,
    SpikeScenario,
    StressTestScenario,
    LoadTestRunner
)

# Configure load test
config = LoadTestConfig(
    base_url="http://localhost:8000",
    duration_seconds=60,
    concurrent_users=50,
    think_time_seconds=1.0,
    ramp_up_seconds=10,
    timeout_seconds=30
)

# Run constant load test
scenario = ConstantLoadScenario(config)
result = asyncio.run(scenario.run_async())
result.print_summary()

# Run all scenarios
runner = LoadTestRunner(base_url="http://localhost:8000")
runner.run_all_scenarios()
```

#### Request Patterns

The load testing framework supports realistic request distributions:

```python
config = LoadTestConfig(
    request_distribution={
        "short": 0.6,   # 60% short requests (< 50 tokens)
        "medium": 0.3,  # 30% medium requests (50-200 tokens)
        "long": 0.1     # 10% long requests (> 200 tokens)
    }
)
```

## Performance Metrics

### Latency Metrics

- **Min**: Minimum latency observed
- **Mean**: Average latency
- **Median (P50)**: 50th percentile
- **P75**: 75th percentile
- **P90**: 90th percentile
- **P95**: 95th percentile
- **P99**: 99th percentile
- **Max**: Maximum latency observed

### Throughput Metrics

- **Requests per Second**: Total requests / duration
- **Tokens per Second**: Total tokens generated / duration
- **Bytes per Second**: Total bytes transferred / duration

### Resource Metrics

- **CPU Usage**: Percentage of CPU utilization
- **Memory Usage**: Memory consumption in MB
- **Memory Growth**: Memory increase over test duration

### Error Metrics

- **Error Rate**: Failed requests / total requests
- **Timeout Count**: Number of timeout errors
- **Error Types**: Distribution of error types

## Regression Detection

### Thresholds

Default regression detection thresholds:

- **RPS Degradation**: 10% decrease
- **Latency Increase**: 20% increase
- **Error Rate Increase**: 5% increase
- **CPU Usage Increase**: 30% increase
- **Memory Usage Increase**: 30% increase

### Customizing Thresholds

```python
baseline = PerformanceBaseline(
    test_name="my_test",
    baseline_date="2024-01-01",
    requests_per_second=100.0,
    latency_p50=50.0,
    latency_p95=100.0,
    latency_p99=150.0,
    error_rate=0.01,
    cpu_usage_percent=45.0,
    memory_usage_mb=512.0,
    rps_threshold=15.0,      # 15% threshold
    latency_threshold=25.0,  # 25% threshold
    error_rate_threshold=3.0 # 3% threshold
)
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Tests

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v2
      
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-asyncio aiohttp matplotlib
      
      - name: Run performance tests
        run: |
          pytest tests/test_performance_benchmarking.py -v
      
      - name: Check for regressions
        run: |
          python tests/performance_regression.py
      
      - name: Upload performance report
        uses: actions/upload-artifact@v2
        with:
          name: performance-report
          path: performance_report.html
```

### Jenkins Pipeline Example

```groovy
pipeline {
    agent any
    
    stages {
        stage('Performance Tests') {
            steps {
                sh 'pytest tests/test_performance_benchmarking.py -v'
            }
        }
        
        stage('Regression Detection') {
            steps {
                sh 'python tests/performance_regression.py'
            }
        }
        
        stage('Generate Report') {
            steps {
                publishHTML([
                    reportDir: '.',
                    reportFiles: 'performance_report.html',
                    reportName: 'Performance Report'
                ])
            }
        }
    }
    
    post {
        always {
            archiveArtifacts artifacts: 'performance_*.json,performance_*.html,performance_*.png'
        }
    }
}
```

## Best Practices

### 1. Baseline Management

- Create baselines from stable, known-good builds
- Update baselines when intentional performance changes are made
- Keep historical baselines for trend analysis
- Document baseline creation conditions

### 2. Test Environment

- Use consistent hardware for performance tests
- Minimize background processes during testing
- Use dedicated test environments
- Control network conditions

### 3. Test Duration

- Run tests long enough to reach steady state
- Include warm-up period for JIT compilation
- Test for at least 5 minutes for sustained load
- Use longer durations for stress tests

### 4. Metrics Collection

- Collect metrics at regular intervals
- Monitor system resources during tests
- Track both client-side and server-side metrics
- Log detailed error information

### 5. Regression Detection

- Set appropriate thresholds based on system characteristics
- Review regressions before failing builds
- Track improvements as well as regressions
- Maintain historical trend data

## Troubleshooting

### High Latency

- Check system resource usage (CPU, memory)
- Verify network connectivity
- Review application logs for errors
- Check for database or external service bottlenecks

### High Error Rate

- Review error messages and types
- Check timeout settings
- Verify service health
- Review rate limiting configuration

### Memory Leaks

- Run extended duration tests
- Monitor memory growth over time
- Use memory profiling tools
- Review object lifecycle management

### Inconsistent Results

- Ensure consistent test environment
- Run multiple iterations
- Check for background processes
- Verify network stability

## Example Workflow

Complete performance testing workflow:

```bash
# 1. Run performance benchmarks
pytest tests/test_performance_benchmarking.py -v -s

# 2. Run load testing scenarios
python tests/load_testing_scenarios.py

# 3. Check for regressions
python tests/performance_regression.py

# 4. Generate reports
python -c "
from tests.performance_regression import PerformanceReporter
import json

with open('performance_results.json') as f:
    results = json.load(f)

reporter = PerformanceReporter()
reporter.generate_report(results, 'performance_report.html')
"

# 5. View results
open performance_report.html
open performance_trends.png
```

## Additional Resources

- [Performance Testing Best Practices](https://martinfowler.com/articles/practical-test-pyramid.html)
- [Load Testing Guidelines](https://www.nginx.com/blog/load-testing-best-practices/)
- [Percentile Metrics](https://www.elastic.co/blog/averages-can-dangerous-use-percentile)
- [Regression Detection](https://netflixtechblog.com/automated-canary-analysis-at-netflix-with-kayenta-3260bc7acc69)

## Support

For issues or questions about performance testing:

1. Check this README for common scenarios
2. Review test output and error messages
3. Consult the main project documentation
4. Open an issue with performance test results attached
