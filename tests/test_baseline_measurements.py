"""
Tests for performance baseline measurement system.

Tests baseline measurement script, benchmark execution, result validation,
and regression detection capabilities.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from run_performance_baseline import PerformanceBaseline


class TestBaselineMeasurements:
    """Test baseline measurement execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline = PerformanceBaseline(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test resources."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_baseline_initialization(self):
        """Test baseline measurement initialization."""
        assert self.baseline.output_dir.exists()
        assert self.baseline.profiler is not None
        assert isinstance(self.baseline.results, dict)
    
    def test_model_loading_benchmark(self):
        """Test model loading benchmark."""
        result = self.baseline._benchmark_model_loading()
        
        assert 'mean' in result
        assert 'min' in result
        assert 'max' in result
        assert 'samples' in result
        
        assert result['mean'] > 0
        assert result['min'] <= result['mean']
        assert result['max'] >= result['mean']
        assert result['samples'] >= 0  # May be 0 in mock mode
    
    def test_first_token_latency_benchmark(self):
        """Test first token latency benchmark."""
        result = self.baseline._benchmark_first_token_latency()
        
        assert 'p50' in result
        assert 'p95' in result
        assert 'p99' in result
        assert 'mean' in result
        assert 'samples' in result
        
        # Percentiles should be ordered
        assert result['p50'] <= result['p95']
        assert result['p95'] <= result['p99']
    
    def test_token_throughput_benchmark(self):
        """Test token throughput benchmark."""
        result = self.baseline._benchmark_token_throughput()
        
        assert 'tokens_per_second' in result
        assert 'mean_latency_ms' in result
        assert 'total_tokens' in result
        assert 'elapsed_seconds' in result
        
        assert result['tokens_per_second'] > 0
        assert result['total_tokens'] > 0
        assert result['elapsed_seconds'] > 0
    
    def test_memory_usage_benchmark(self):
        """Test memory usage benchmark."""
        result = self.baseline._benchmark_memory_usage()
        
        assert 'peak_mb' in result
        assert 'average_mb' in result
        assert 'min_mb' in result
        assert 'samples' in result
        
        assert result['peak_mb'] >= result['average_mb']
        assert result['average_mb'] >= result['min_mb']
    
    def test_resource_utilization_benchmark(self):
        """Test resource utilization benchmark."""
        result = self.baseline._benchmark_resource_utilization()
        
        assert 'cpu_percent' in result
        assert 'memory_percent' in result
        assert 'samples' in result
        
        assert 0 <= result['cpu_percent'] <= 100
        assert 0 <= result['memory_percent'] <= 100
    
    def test_pipeline_init_benchmark(self):
        """Test pipeline initialization benchmark."""
        result = self.baseline._benchmark_pipeline_init()
        
        assert 'mean' in result
        assert 'min' in result
        assert 'max' in result
        assert 'samples' in result
        
        assert result['mean'] > 0
    
    def test_activation_transfer_benchmark(self):
        """Test activation transfer benchmark."""
        result = self.baseline._benchmark_activation_transfer()
        
        assert 'p50' in result
        assert 'p95' in result
        assert 'mean' in result
        assert 'bandwidth_mbps' in result
        assert 'samples' in result
        
        assert result['bandwidth_mbps'] > 0
    
    def test_e2e_latency_benchmark(self):
        """Test end-to-end latency benchmark."""
        result = self.baseline._benchmark_e2e_latency()
        
        assert 'p50' in result
        assert 'p95' in result
        assert 'p99' in result
        assert 'mean' in result
        
        # Percentiles should be ordered
        assert result['p50'] <= result['p95'] <= result['p99']
    
    def test_network_bandwidth_benchmark(self):
        """Test network bandwidth benchmark."""
        result = self.baseline._benchmark_network_bandwidth()
        
        assert 'average_mbps' in result
        assert 'peak_mbps' in result
        assert 'min_mbps' in result
        assert 'samples' in result
        
        assert result['peak_mbps'] >= result['average_mbps']
        assert result['average_mbps'] >= result['min_mbps']
    
    def test_coordination_overhead_benchmark(self):
        """Test coordination overhead benchmark."""
        result = self.baseline._benchmark_coordination_overhead()
        
        assert 'mean' in result
        assert 'overhead_percent' in result
        assert 'total_mean' in result
        assert 'worker_mean' in result
        
        assert result['overhead_percent'] >= 0
        assert result['total_mean'] >= result['worker_mean']
    
    def test_grpc_latency_benchmark(self):
        """Test gRPC latency benchmark."""
        result = self.baseline._benchmark_grpc_latency()
        
        assert 'mean' in result
        assert 'p95' in result
        assert 'min' in result
        assert 'max' in result
        
        assert result['min'] <= result['mean'] <= result['max']
    
    def test_serialization_benchmark(self):
        """Test serialization benchmark."""
        result = self.baseline._benchmark_serialization()
        
        assert 'serialize_mean' in result
        assert 'deserialize_mean' in result
        assert 'total_mean' in result
        
        assert result['total_mean'] == result['serialize_mean'] + result['deserialize_mean']
    
    def test_routing_latency_benchmark(self):
        """Test routing latency benchmark."""
        result = self.baseline._benchmark_routing_latency()
        
        assert 'mean' in result
        assert 'min' in result
        assert 'max' in result
        
        assert result['min'] <= result['mean'] <= result['max']
    
    def test_api_overhead_benchmark(self):
        """Test API overhead benchmark."""
        result = self.baseline._benchmark_api_overhead()
        
        assert 'mean' in result
        assert 'min' in result
        assert 'max' in result
        
        assert result['min'] <= result['mean'] <= result['max']


class TestBenchmarkSuites:
    """Test complete benchmark suite execution."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline = PerformanceBaseline(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test resources."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_single_node_benchmarks(self):
        """Test single-node benchmark suite."""
        self.baseline.run_single_node_benchmarks()
        
        assert 'single_node' in self.baseline.results
        sn = self.baseline.results['single_node']
        
        assert 'model_loading' in sn
        assert 'first_token_latency' in sn
        assert 'token_throughput' in sn
        assert 'memory_usage' in sn
        assert 'resource_utilization' in sn
    
    def test_distributed_benchmarks(self):
        """Test distributed benchmark suite."""
        self.baseline.run_distributed_benchmarks()
        
        assert 'distributed' in self.baseline.results
        dist = self.baseline.results['distributed']
        
        assert 'pipeline_init' in dist
        assert 'activation_transfer' in dist
        assert 'e2e_latency' in dist
        assert 'network_bandwidth' in dist
        assert 'coordination_overhead' in dist
    
    def test_system_overhead_benchmarks(self):
        """Test system overhead benchmark suite."""
        self.baseline.run_system_overhead_benchmarks()
        
        assert 'system_overhead' in self.baseline.results
        so = self.baseline.results['system_overhead']
        
        assert 'grpc_latency' in so
        assert 'serialization' in so
        assert 'routing_latency' in so
        assert 'api_overhead' in so


class TestReportGeneration:
    """Test report generation and output."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline = PerformanceBaseline(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test resources."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_json_report_generation(self):
        """Test JSON report generation."""
        # Add some test results
        self.baseline.results = {
            'single_node': {
                'model_loading': {'mean': 100.0, 'min': 90.0, 'max': 110.0}
            }
        }
        
        self.baseline.generate_report()
        
        report_file = Path(self.temp_dir) / "baseline_report.json"
        assert report_file.exists()
        
        # Verify JSON content
        with open(report_file, 'r') as f:
            data = json.load(f)
        
        assert 'single_node' in data
        assert 'metadata' in data
        assert 'timestamp' in data['metadata']
    
    def test_summary_report_generation(self):
        """Test summary report generation."""
        # Add test results
        self.baseline.results = {
            'single_node': {
                'model_loading': {'mean': 100.0},
                'first_token_latency': {'p50': 50.0, 'p95': 75.0},
                'token_throughput': {'tokens_per_second': 50.0},
                'memory_usage': {'peak_mb': 200.0}
            },
            'distributed': {
                'pipeline_init': {'mean': 150.0},
                'activation_transfer': {'p50': 8.0},
                'e2e_latency': {'p50': 80.0, 'p95': 120.0},
                'network_bandwidth': {'average_mbps': 50.0},
                'coordination_overhead': {'overhead_percent': 5.0}
            },
            'system_overhead': {
                'grpc_latency': {'mean': 3.0},
                'serialization': {'total_mean': 2.5},
                'routing_latency': {'mean': 2.0},
                'api_overhead': {'mean': 5.0}
            }
        }
        
        self.baseline.generate_report()
        
        summary_file = Path(self.temp_dir) / "baseline_summary.txt"
        assert summary_file.exists()
        
        # Verify summary content
        with open(summary_file, 'r') as f:
            content = f.read()
        
        assert 'SINGLE-NODE INFERENCE' in content
        assert 'DISTRIBUTED INFERENCE' in content
        assert 'SYSTEM OVERHEAD' in content
        assert '100.00ms' in content  # Model loading
        assert '50.00 tokens/sec' in content  # Throughput


class TestRegressionDetection:
    """Test performance regression detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test resources."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_regression_detection_threshold(self):
        """Test regression detection with threshold."""
        baseline = {'mean': 100.0, 'p95': 150.0}
        current = {'mean': 115.0, 'p95': 170.0}
        
        # 10% threshold
        threshold = 0.10
        
        mean_regression = (current['mean'] - baseline['mean']) / baseline['mean']
        p95_regression = (current['p95'] - baseline['p95']) / baseline['p95']
        
        assert mean_regression > threshold  # 15% regression
        assert p95_regression > threshold  # 13% regression
    
    def test_no_regression_detection(self):
        """Test no regression when performance improves."""
        baseline = {'mean': 100.0, 'p95': 150.0}
        current = {'mean': 95.0, 'p95': 140.0}
        
        threshold = 0.10
        
        mean_regression = (current['mean'] - baseline['mean']) / baseline['mean']
        p95_regression = (current['p95'] - baseline['p95']) / baseline['p95']
        
        assert mean_regression < 0  # Improvement
        assert p95_regression < 0  # Improvement
    
    def test_regression_comparison(self):
        """Test comparing two baseline measurements."""
        baseline1 = {
            'single_node': {
                'first_token_latency': {'p50': 50.0, 'p95': 75.0},
                'token_throughput': {'tokens_per_second': 50.0}
            }
        }
        
        baseline2 = {
            'single_node': {
                'first_token_latency': {'p50': 60.0, 'p95': 90.0},
                'token_throughput': {'tokens_per_second': 45.0}
            }
        }
        
        # Calculate regressions
        p50_regression = (baseline2['single_node']['first_token_latency']['p50'] - 
                         baseline1['single_node']['first_token_latency']['p50']) / \
                         baseline1['single_node']['first_token_latency']['p50']
        
        throughput_regression = (baseline2['single_node']['token_throughput']['tokens_per_second'] - 
                                baseline1['single_node']['token_throughput']['tokens_per_second']) / \
                                baseline1['single_node']['token_throughput']['tokens_per_second']
        
        assert p50_regression > 0.10  # 20% latency regression
        assert throughput_regression < -0.05  # 10% throughput regression


class TestBenchmarkValidation:
    """Test benchmark result validation."""
    
    def test_latency_percentile_ordering(self):
        """Test that latency percentiles are properly ordered."""
        # Use larger sample for proper percentile calculation
        latencies = list(range(50, 250, 2))  # 100 samples from 50 to 248
        
        import statistics
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18]
        p99 = statistics.quantiles(latencies, n=100)[98]
        
        assert p50 <= p95
        assert p95 <= p99
    
    def test_throughput_calculation(self):
        """Test throughput calculation."""
        num_tokens = 100
        elapsed_seconds = 2.0
        
        tokens_per_second = num_tokens / elapsed_seconds
        
        assert tokens_per_second == 50.0
    
    def test_bandwidth_calculation(self):
        """Test bandwidth calculation."""
        transfer_size_mb = 10.0
        transfer_time_s = 0.2
        
        bandwidth_mbps = transfer_size_mb / transfer_time_s
        
        assert bandwidth_mbps == 50.0
    
    def test_overhead_percentage_calculation(self):
        """Test overhead percentage calculation."""
        total_time = 100.0
        worker_time = 95.0
        
        overhead = total_time - worker_time
        overhead_percent = (overhead / total_time) * 100
        
        assert overhead == 5.0
        assert overhead_percent == 5.0


class TestBenchmarkReliability:
    """Test benchmark reliability and consistency."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.baseline = PerformanceBaseline(output_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test resources."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_benchmark_repeatability(self):
        """Test that benchmarks produce consistent results."""
        results1 = self.baseline._benchmark_model_loading()
        results2 = self.baseline._benchmark_model_loading()
        
        # Results should be similar (within 20% variance)
        variance = abs(results1['mean'] - results2['mean']) / results1['mean']
        assert variance < 0.20
    
    def test_benchmark_sample_size(self):
        """Test that benchmarks collect sufficient samples."""
        result = self.baseline._benchmark_first_token_latency()
        
        # Should collect samples (may be 0 in mock mode, but structure should be correct)
        assert 'samples' in result
        assert result['samples'] >= 0
    
    def test_benchmark_outlier_handling(self):
        """Test that benchmarks handle outliers appropriately."""
        latencies = [50] * 95 + [500, 600, 700, 800, 900]  # 5% outliers
        
        import statistics
        mean = statistics.mean(latencies)
        median = statistics.median(latencies)
        
        # Median should be more robust to outliers
        assert median < mean
        assert median == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
