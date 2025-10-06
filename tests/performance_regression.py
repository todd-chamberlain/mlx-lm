"""
Performance Regression Detection and Reporting

Automated performance regression detection with baseline comparison,
trend analysis, and alerting capabilities.
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


@dataclass
class PerformanceBaseline:
    """Performance baseline for regression detection."""
    
    test_name: str
    baseline_date: str
    requests_per_second: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    error_rate: float
    cpu_usage_percent: float
    memory_usage_mb: float
    
    # Thresholds for regression detection (percentage)
    rps_threshold: float = 10.0  # 10% degradation
    latency_threshold: float = 20.0  # 20% increase
    error_rate_threshold: float = 5.0  # 5% increase
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RegressionResult:
    """Regression detection result."""
    
    test_name: str
    has_regression: bool
    regressions: List[str]
    improvements: List[str]
    baseline_metrics: Dict[str, float]
    current_metrics: Dict[str, float]
    percentage_changes: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_summary(self) -> str:
        """Get human-readable summary."""
        lines = [f"\nRegression Analysis: {self.test_name}"]
        lines.append("=" * 60)
        
        if self.has_regression:
            lines.append("\n⚠️  PERFORMANCE REGRESSIONS DETECTED:")
            for regression in self.regressions:
                lines.append(f"  - {regression}")
        else:
            lines.append("\n✓ No performance regressions detected")
        
        if self.improvements:
            lines.append("\n✓ Performance Improvements:")
            for improvement in self.improvements:
                lines.append(f"  - {improvement}")
        
        lines.append("\nMetric Changes:")
        for metric, change in self.percentage_changes.items():
            symbol = "↑" if change > 0 else "↓"
            lines.append(f"  {metric}: {symbol} {abs(change):.1f}%")
        
        lines.append("=" * 60)
        return "\n".join(lines)


class PerformanceRegressionDetector:
    """Detect performance regressions by comparing against baselines."""
    
    def __init__(self, baseline_file: str = "performance_baseline.json"):
        self.baseline_file = Path(baseline_file)
        self.baselines: Dict[str, PerformanceBaseline] = {}
        self.load_baselines()
    
    def load_baselines(self):
        """Load performance baselines from file."""
        if self.baseline_file.exists():
            with open(self.baseline_file, 'r') as f:
                data = json.load(f)
                for test_name, baseline_data in data.items():
                    self.baselines[test_name] = PerformanceBaseline(**baseline_data)
    
    def save_baselines(self):
        """Save performance baselines to file."""
        data = {name: baseline.to_dict() for name, baseline in self.baselines.items()}
        
        self.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_baseline(self, test_name: str, metrics: Dict[str, float]):
        """Create a new performance baseline."""
        baseline = PerformanceBaseline(
            test_name=test_name,
            baseline_date=datetime.now().isoformat(),
            requests_per_second=metrics.get("requests_per_second", 0),
            latency_p50=metrics.get("latency_median", 0),
            latency_p95=metrics.get("latency_p95", 0),
            latency_p99=metrics.get("latency_p99", 0),
            error_rate=metrics.get("error_rate", 0),
            cpu_usage_percent=metrics.get("cpu_usage_percent", 0),
            memory_usage_mb=metrics.get("memory_usage_mb", 0)
        )
        
        self.baselines[test_name] = baseline
        self.save_baselines()
        
        return baseline
    
    def detect_regression(
        self,
        test_name: str,
        current_metrics: Dict[str, float]
    ) -> RegressionResult:
        """Detect performance regression by comparing with baseline."""
        if test_name not in self.baselines:
            # No baseline exists, create one
            self.create_baseline(test_name, current_metrics)
            return RegressionResult(
                test_name=test_name,
                has_regression=False,
                regressions=[],
                improvements=[],
                baseline_metrics={},
                current_metrics=current_metrics,
                percentage_changes={}
            )
        
        baseline = self.baselines[test_name]
        regressions = []
        improvements = []
        percentage_changes = {}
        
        # Check requests per second (lower is worse)
        if baseline.requests_per_second > 0:
            rps_change = ((current_metrics.get("requests_per_second", 0) - baseline.requests_per_second) 
                         / baseline.requests_per_second * 100)
            percentage_changes["requests_per_second"] = rps_change
            
            if rps_change < -baseline.rps_threshold:
                regressions.append(
                    f"Requests/sec decreased by {abs(rps_change):.1f}% "
                    f"({baseline.requests_per_second:.1f} → {current_metrics.get('requests_per_second', 0):.1f})"
                )
            elif rps_change > baseline.rps_threshold:
                improvements.append(
                    f"Requests/sec improved by {rps_change:.1f}%"
                )
        
        # Check latency P95 (higher is worse)
        if baseline.latency_p95 > 0:
            p95_change = ((current_metrics.get("latency_p95", 0) - baseline.latency_p95) 
                         / baseline.latency_p95 * 100)
            percentage_changes["latency_p95"] = p95_change
            
            if p95_change > baseline.latency_threshold:
                regressions.append(
                    f"P95 latency increased by {p95_change:.1f}% "
                    f"({baseline.latency_p95:.1f}ms → {current_metrics.get('latency_p95', 0):.1f}ms)"
                )
            elif p95_change < -baseline.latency_threshold:
                improvements.append(
                    f"P95 latency improved by {abs(p95_change):.1f}%"
                )
        
        # Check latency P99 (higher is worse)
        if baseline.latency_p99 > 0:
            p99_change = ((current_metrics.get("latency_p99", 0) - baseline.latency_p99) 
                         / baseline.latency_p99 * 100)
            percentage_changes["latency_p99"] = p99_change
            
            if p99_change > baseline.latency_threshold:
                regressions.append(
                    f"P99 latency increased by {p99_change:.1f}% "
                    f"({baseline.latency_p99:.1f}ms → {current_metrics.get('latency_p99', 0):.1f}ms)"
                )
        
        # Check error rate (higher is worse)
        error_rate_change = (current_metrics.get("error_rate", 0) - baseline.error_rate) * 100
        percentage_changes["error_rate"] = error_rate_change
        
        if error_rate_change > baseline.error_rate_threshold:
            regressions.append(
                f"Error rate increased by {error_rate_change:.1f}% "
                f"({baseline.error_rate*100:.1f}% → {current_metrics.get('error_rate', 0)*100:.1f}%)"
            )
        
        # Check CPU usage
        if baseline.cpu_usage_percent > 0:
            cpu_change = ((current_metrics.get("cpu_usage_percent", 0) - baseline.cpu_usage_percent) 
                         / baseline.cpu_usage_percent * 100)
            percentage_changes["cpu_usage"] = cpu_change
            
            if cpu_change > 30:  # 30% increase in CPU usage
                regressions.append(
                    f"CPU usage increased by {cpu_change:.1f}%"
                )
        
        # Check memory usage
        if baseline.memory_usage_mb > 0:
            memory_change = ((current_metrics.get("memory_usage_mb", 0) - baseline.memory_usage_mb) 
                            / baseline.memory_usage_mb * 100)
            percentage_changes["memory_usage"] = memory_change
            
            if memory_change > 30:  # 30% increase in memory usage
                regressions.append(
                    f"Memory usage increased by {memory_change:.1f}%"
                )
        
        result = RegressionResult(
            test_name=test_name,
            has_regression=len(regressions) > 0,
            regressions=regressions,
            improvements=improvements,
            baseline_metrics={
                "requests_per_second": baseline.requests_per_second,
                "latency_p95": baseline.latency_p95,
                "latency_p99": baseline.latency_p99,
                "error_rate": baseline.error_rate
            },
            current_metrics=current_metrics,
            percentage_changes=percentage_changes
        )
        
        return result


class PerformanceTrendAnalyzer:
    """Analyze performance trends over time."""
    
    def __init__(self, history_file: str = "performance_history.json"):
        self.history_file = Path(history_file)
        self.history: Dict[str, List[Dict[str, Any]]] = {}
        self.load_history()
    
    def load_history(self):
        """Load performance history from file."""
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.history = json.load(f)
    
    def save_history(self):
        """Save performance history to file."""
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_result(self, test_name: str, metrics: Dict[str, Any]):
        """Add performance result to history."""
        if test_name not in self.history:
            self.history[test_name] = []
        
        # Add timestamp if not present
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now().isoformat()
        
        self.history[test_name].append(metrics)
        self.save_history()
    
    def get_trend(self, test_name: str, metric_name: str) -> List[float]:
        """Get trend for a specific metric."""
        if test_name not in self.history:
            return []
        
        return [result.get(metric_name, 0) for result in self.history[test_name]]
    
    def analyze_trend(self, test_name: str, metric_name: str) -> Dict[str, Any]:
        """Analyze trend for a specific metric."""
        values = self.get_trend(test_name, metric_name)
        
        if len(values) < 2:
            return {
                "metric": metric_name,
                "trend": "insufficient_data",
                "values": values
            }
        
        # Calculate trend direction
        recent_avg = statistics.mean(values[-5:]) if len(values) >= 5 else statistics.mean(values)
        older_avg = statistics.mean(values[:5]) if len(values) >= 10 else statistics.mean(values[:-5]) if len(values) > 5 else values[0]
        
        change_percent = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        
        if abs(change_percent) < 5:
            trend = "stable"
        elif change_percent > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "metric": metric_name,
            "trend": trend,
            "change_percent": change_percent,
            "recent_average": recent_avg,
            "older_average": older_avg,
            "values": values
        }
    
    def generate_trend_report(self, test_name: str) -> str:
        """Generate trend analysis report."""
        if test_name not in self.history:
            return f"No history available for {test_name}"
        
        metrics_to_analyze = [
            "requests_per_second",
            "latency_p95",
            "latency_p99",
            "error_rate"
        ]
        
        lines = [f"\nPerformance Trend Analysis: {test_name}"]
        lines.append("=" * 60)
        lines.append(f"Data points: {len(self.history[test_name])}")
        lines.append("")
        
        for metric in metrics_to_analyze:
            analysis = self.analyze_trend(test_name, metric)
            
            if analysis["trend"] == "insufficient_data":
                continue
            
            trend_symbol = {
                "stable": "→",
                "increasing": "↑",
                "decreasing": "↓"
            }.get(analysis["trend"], "?")
            
            lines.append(f"{metric}:")
            lines.append(f"  Trend: {trend_symbol} {analysis['trend']}")
            lines.append(f"  Change: {analysis['change_percent']:.1f}%")
            lines.append(f"  Recent avg: {analysis['recent_average']:.2f}")
            lines.append(f"  Older avg: {analysis['older_average']:.2f}")
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def plot_trends(self, test_name: str, output_file: str = "performance_trends.png"):
        """Generate trend visualization plots."""
        if test_name not in self.history or len(self.history[test_name]) < 2:
            print(f"Insufficient data to plot trends for {test_name}")
            return
        
        metrics = ["requests_per_second", "latency_p95", "latency_p99", "error_rate"]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"Performance Trends: {test_name}")
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = self.get_trend(test_name, metric)
            
            if values:
                ax.plot(range(len(values)), values, marker='o')
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_xlabel('Test Run')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file)
        print(f"Trend plot saved to {output_file}")


class PerformanceReporter:
    """Generate comprehensive performance reports."""
    
    def __init__(self):
        self.detector = PerformanceRegressionDetector()
        self.analyzer = PerformanceTrendAnalyzer()
    
    def generate_report(
        self,
        test_results: List[Dict[str, Any]],
        output_file: str = "performance_report.html"
    ) -> str:
        """Generate comprehensive performance report."""
        html_parts = []
        
        # HTML header
        html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <title>Performance Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        h2 { color: #666; margin-top: 30px; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        .regression { background-color: #ffcccc; }
        .improvement { background-color: #ccffcc; }
        .stable { background-color: #f0f0f0; }
        .metric { margin: 10px 0; }
        .summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }
    </style>
</head>
<body>
""")
        
        # Report header
        html_parts.append(f"<h1>Performance Test Report</h1>")
        html_parts.append(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
        
        # Summary section
        html_parts.append("<div class='summary'>")
        html_parts.append("<h2>Summary</h2>")
        html_parts.append(f"<p>Total tests: {len(test_results)}</p>")
        html_parts.append("</div>")
        
        # Individual test results
        for result in test_results:
            test_name = result.get("test_name", "Unknown")
            
            html_parts.append(f"<h2>{test_name}</h2>")
            
            # Regression detection
            regression_result = self.detector.detect_regression(test_name, result)
            
            if regression_result.has_regression:
                html_parts.append("<div class='regression'>")
                html_parts.append("<h3>⚠️ Regressions Detected</h3>")
                html_parts.append("<ul>")
                for reg in regression_result.regressions:
                    html_parts.append(f"<li>{reg}</li>")
                html_parts.append("</ul>")
                html_parts.append("</div>")
            
            if regression_result.improvements:
                html_parts.append("<div class='improvement'>")
                html_parts.append("<h3>✓ Improvements</h3>")
                html_parts.append("<ul>")
                for imp in regression_result.improvements:
                    html_parts.append(f"<li>{imp}</li>")
                html_parts.append("</ul>")
                html_parts.append("</div>")
            
            # Metrics table
            html_parts.append("<table>")
            html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
            
            for key, value in result.items():
                if key not in ["test_name", "timestamp"]:
                    if isinstance(value, float):
                        html_parts.append(f"<tr><td>{key}</td><td>{value:.2f}</td></tr>")
                    else:
                        html_parts.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
            
            html_parts.append("</table>")
            
            # Add to history
            self.analyzer.add_result(test_name, result)
        
        # HTML footer
        html_parts.append("</body></html>")
        
        # Write report
        html_content = "\n".join(html_parts)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Performance report generated: {output_file}")
        return html_content


if __name__ == "__main__":
    # Example usage
    detector = PerformanceRegressionDetector()
    analyzer = PerformanceTrendAnalyzer()
    reporter = PerformanceReporter()
    
    # Example metrics
    example_metrics = {
        "test_name": "example_test",
        "requests_per_second": 100.0,
        "latency_median": 50.0,
        "latency_p95": 100.0,
        "latency_p99": 150.0,
        "error_rate": 0.01,
        "cpu_usage_percent": 45.0,
        "memory_usage_mb": 512.0
    }
    
    # Detect regression
    result = detector.detect_regression("example_test", example_metrics)
    print(result.get_summary())
    
    # Add to history
    analyzer.add_result("example_test", example_metrics)
    
    # Generate trend report
    print(analyzer.generate_trend_report("example_test"))
