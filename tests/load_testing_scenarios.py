"""
Load Testing Scenarios with Realistic Request Patterns

Comprehensive load testing scenarios for MLX distributed inference system
with various traffic patterns and workload simulations.
"""

import asyncio
import aiohttp
import time
import random
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LoadTestConfig:
    """Configuration for load testing scenarios."""
    
    base_url: str = "http://localhost:8000"
    duration_seconds: int = 60
    concurrent_users: int = 10
    requests_per_user: Optional[int] = None
    think_time_seconds: float = 1.0
    ramp_up_seconds: int = 10
    timeout_seconds: int = 30
    
    # Request patterns
    request_distribution: Dict[str, float] = field(default_factory=lambda: {
        "short": 0.6,   # 60% short requests (< 50 tokens)
        "medium": 0.3,  # 30% medium requests (50-200 tokens)
        "long": 0.1     # 10% long requests (> 200 tokens)
    })


@dataclass
class LoadTestResult:
    """Result from a load test execution."""
    
    scenario_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration_seconds: float
    requests_per_second: float
    
    latencies: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Calculate latency percentiles."""
        if not self.latencies:
            return {}
        
        sorted_latencies = sorted(self.latencies)
        return {
            "p50": sorted_latencies[int(len(sorted_latencies) * 0.50)],
            "p75": sorted_latencies[int(len(sorted_latencies) * 0.75)],
            "p90": sorted_latencies[int(len(sorted_latencies) * 0.90)],
            "p95": sorted_latencies[int(len(sorted_latencies) * 0.95)],
            "p99": sorted_latencies[int(len(sorted_latencies) * 0.99)],
        }
    
    def print_summary(self):
        """Print test result summary."""
        print(f"\n{'='*60}")
        print(f"Load Test Results: {self.scenario_name}")
        print(f"{'='*60}")
        print(f"Duration: {self.duration_seconds:.2f}s")
        print(f"Total Requests: {self.total_requests}")
        print(f"Successful: {self.successful_requests}")
        print(f"Failed: {self.failed_requests}")
        print(f"Success Rate: {(self.successful_requests/self.total_requests*100):.1f}%")
        print(f"Requests/sec: {self.requests_per_second:.2f}")
        
        if self.latencies:
            percentiles = self.get_latency_percentiles()
            print(f"\nLatency Percentiles (ms):")
            for p, value in percentiles.items():
                print(f"  {p}: {value:.2f}")
        
        if self.errors:
            print(f"\nTop Errors:")
            error_counts = {}
            for error in self.errors[:10]:
                error_counts[error] = error_counts.get(error, 0) + 1
            for error, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"  {error}: {count}")
        
        print(f"{'='*60}\n")


class RequestGenerator:
    """Generate realistic inference requests."""
    
    def __init__(self):
        self.prompts = {
            "short": [
                "Hello, how are you?",
                "What is the capital of France?",
                "Tell me a joke.",
                "What's the weather like?",
                "Good morning!"
            ],
            "medium": [
                "Can you explain the concept of machine learning in simple terms?",
                "Write a short poem about nature and the changing seasons.",
                "What are the main differences between Python and JavaScript?",
                "Describe the process of photosynthesis step by step.",
                "Give me three tips for improving productivity at work."
            ],
            "long": [
                "Write a detailed essay about the impact of artificial intelligence on modern society, "
                "including both positive and negative aspects, and provide examples from various industries.",
                "Explain the history of the internet from its inception to the present day, "
                "covering major milestones, key technologies, and influential figures.",
                "Describe a comprehensive business plan for a sustainable energy startup, "
                "including market analysis, financial projections, and growth strategy.",
                "Write a technical tutorial on building a web application using modern frameworks, "
                "covering frontend, backend, database design, and deployment strategies."
            ]
        }
    
    def generate_request(self, request_type: str = "short") -> Dict[str, Any]:
        """Generate a request payload."""
        prompts = self.prompts.get(request_type, self.prompts["short"])
        prompt = random.choice(prompts)
        
        # Determine max_tokens based on request type
        max_tokens_map = {
            "short": random.randint(20, 50),
            "medium": random.randint(50, 200),
            "long": random.randint(200, 500)
        }
        
        return {
            "model": "test-model",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens_map.get(request_type, 50),
            "temperature": random.uniform(0.5, 1.0),
            "stream": False
        }
    
    def generate_streaming_request(self, request_type: str = "short") -> Dict[str, Any]:
        """Generate a streaming request payload."""
        request = self.generate_request(request_type)
        request["stream"] = True
        return request


class LoadTestScenario:
    """Base class for load testing scenarios."""
    
    def __init__(self, config: LoadTestConfig):
        self.config = config
        self.request_generator = RequestGenerator()
        self.results_queue = Queue()
        self.stop_event = threading.Event()
    
    async def send_request(
        self,
        session: aiohttp.ClientSession,
        request_data: Dict[str, Any]
    ) -> Tuple[bool, float, Optional[str]]:
        """Send a single request and measure latency."""
        start_time = time.perf_counter()
        
        try:
            async with session.post(
                f"{self.config.base_url}/v1/chat/completions",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            ) as response:
                await response.text()
                latency = (time.perf_counter() - start_time) * 1000
                
                if response.status == 200:
                    return True, latency, None
                else:
                    return False, latency, f"HTTP {response.status}"
        
        except asyncio.TimeoutError:
            latency = (time.perf_counter() - start_time) * 1000
            return False, latency, "Timeout"
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000
            return False, latency, str(e)
    
    def select_request_type(self) -> str:
        """Select request type based on distribution."""
        rand = random.random()
        cumulative = 0.0
        
        for request_type, probability in self.config.request_distribution.items():
            cumulative += probability
            if rand <= cumulative:
                return request_type
        
        return "short"
    
    async def run_async(self) -> LoadTestResult:
        """Run the load test scenario asynchronously."""
        raise NotImplementedError("Subclasses must implement run_async")


class ConstantLoadScenario(LoadTestScenario):
    """Constant load scenario with fixed number of concurrent users."""
    
    async def user_session(
        self,
        user_id: int,
        session: aiohttp.ClientSession,
        results: List[Tuple[bool, float, Optional[str]]]
    ):
        """Simulate a single user session."""
        requests_sent = 0
        max_requests = self.config.requests_per_user or float('inf')
        
        while not self.stop_event.is_set() and requests_sent < max_requests:
            # Select request type
            request_type = self.select_request_type()
            request_data = self.request_generator.generate_request(request_type)
            
            # Send request
            success, latency, error = await self.send_request(session, request_data)
            results.append((success, latency, error))
            requests_sent += 1
            
            # Think time
            if self.config.think_time_seconds > 0:
                await asyncio.sleep(self.config.think_time_seconds)
    
    async def run_async(self) -> LoadTestResult:
        """Run constant load test."""
        logger.info(f"Starting constant load test: {self.config.concurrent_users} users")
        
        results = []
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            # Create user tasks
            tasks = [
                self.user_session(i, session, results)
                for i in range(self.config.concurrent_users)
            ]
            
            # Run for specified duration
            await asyncio.sleep(self.config.duration_seconds)
            self.stop_event.set()
            
            # Wait for all tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Process results
        successful = sum(1 for success, _, _ in results if success)
        failed = len(results) - successful
        latencies = [latency for _, latency, _ in results]
        errors = [error for _, _, error in results if error]
        
        return LoadTestResult(
            scenario_name="Constant Load",
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            duration_seconds=duration,
            requests_per_second=len(results) / duration if duration > 0 else 0,
            latencies=latencies,
            errors=errors
        )


class RampUpScenario(LoadTestScenario):
    """Ramp-up scenario with gradually increasing load."""
    
    async def run_async(self) -> LoadTestResult:
        """Run ramp-up load test."""
        logger.info(f"Starting ramp-up test: 0 → {self.config.concurrent_users} users")
        
        results = []
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            active_tasks = []
            users_per_step = max(1, self.config.concurrent_users // 10)
            ramp_step_duration = self.config.ramp_up_seconds / 10
            
            # Ramp up users
            for step in range(10):
                # Add new users
                for _ in range(users_per_step):
                    task = asyncio.create_task(
                        self.user_session(len(active_tasks), session, results)
                    )
                    active_tasks.append(task)
                
                await asyncio.sleep(ramp_step_duration)
            
            # Continue at full load
            remaining_time = self.config.duration_seconds - self.config.ramp_up_seconds
            if remaining_time > 0:
                await asyncio.sleep(remaining_time)
            
            # Stop all users
            self.stop_event.set()
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Process results
        successful = sum(1 for success, _, _ in results if success)
        failed = len(results) - successful
        latencies = [latency for _, latency, _ in results]
        errors = [error for _, _, error in results if error]
        
        return LoadTestResult(
            scenario_name="Ramp-Up Load",
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            duration_seconds=duration,
            requests_per_second=len(results) / duration if duration > 0 else 0,
            latencies=latencies,
            errors=errors
        )


class SpikeScenario(LoadTestScenario):
    """Spike scenario with sudden traffic bursts."""
    
    async def run_async(self) -> LoadTestResult:
        """Run spike load test."""
        logger.info(f"Starting spike test: sudden burst to {self.config.concurrent_users} users")
        
        results = []
        start_time = time.perf_counter()
        
        async with aiohttp.ClientSession() as session:
            # Phase 1: Normal load (20% of users)
            normal_users = max(1, self.config.concurrent_users // 5)
            tasks = [
                self.user_session(i, session, results)
                for i in range(normal_users)
            ]
            
            await asyncio.sleep(self.config.duration_seconds * 0.3)
            
            # Phase 2: Spike (all users)
            spike_tasks = [
                self.user_session(i, session, results)
                for i in range(normal_users, self.config.concurrent_users)
            ]
            tasks.extend(spike_tasks)
            
            await asyncio.sleep(self.config.duration_seconds * 0.4)
            
            # Phase 3: Back to normal
            self.stop_event.set()
            await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Process results
        successful = sum(1 for success, _, _ in results if success)
        failed = len(results) - successful
        latencies = [latency for _, latency, _ in results]
        errors = [error for _, _, error in results if error]
        
        return LoadTestResult(
            scenario_name="Spike Load",
            total_requests=len(results),
            successful_requests=successful,
            failed_requests=failed,
            duration_seconds=duration,
            requests_per_second=len(results) / duration if duration > 0 else 0,
            latencies=latencies,
            errors=errors
        )


class StressTestScenario(LoadTestScenario):
    """Stress test to find system breaking point."""
    
    async def run_async(self) -> LoadTestResult:
        """Run stress test with increasing load until failure."""
        logger.info("Starting stress test: finding system limits")
        
        all_results = []
        start_time = time.perf_counter()
        
        # Start with low load and increase
        current_users = 10
        max_users = self.config.concurrent_users
        
        async with aiohttp.ClientSession() as session:
            while current_users <= max_users:
                logger.info(f"Testing with {current_users} concurrent users")
                
                results = []
                self.stop_event.clear()
                
                # Run test with current load
                tasks = [
                    self.user_session(i, session, results)
                    for i in range(current_users)
                ]
                
                await asyncio.sleep(10)  # 10 second test at each level
                self.stop_event.set()
                await asyncio.gather(*tasks, return_exceptions=True)
                
                all_results.extend(results)
                
                # Check error rate
                if results:
                    error_rate = sum(1 for success, _, _ in results if not success) / len(results)
                    if error_rate > 0.2:  # More than 20% errors
                        logger.warning(f"High error rate at {current_users} users: {error_rate:.1%}")
                        break
                
                # Increase load
                current_users = int(current_users * 1.5)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Process results
        successful = sum(1 for success, _, _ in all_results if success)
        failed = len(all_results) - successful
        latencies = [latency for _, latency, _ in all_results]
        errors = [error for _, _, error in all_results if error]
        
        return LoadTestResult(
            scenario_name="Stress Test",
            total_requests=len(all_results),
            successful_requests=successful,
            failed_requests=failed,
            duration_seconds=duration,
            requests_per_second=len(all_results) / duration if duration > 0 else 0,
            latencies=latencies,
            errors=errors
        )


class LoadTestRunner:
    """Run multiple load testing scenarios."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[LoadTestResult] = []
    
    def run_scenario(self, scenario: LoadTestScenario) -> LoadTestResult:
        """Run a single load test scenario."""
        result = asyncio.run(scenario.run_async())
        self.results.append(result)
        result.print_summary()
        return result
    
    def run_all_scenarios(self):
        """Run all predefined load test scenarios."""
        scenarios = [
            # Constant load test
            ConstantLoadScenario(LoadTestConfig(
                base_url=self.base_url,
                duration_seconds=30,
                concurrent_users=10,
                think_time_seconds=0.5
            )),
            
            # Ramp-up test
            RampUpScenario(LoadTestConfig(
                base_url=self.base_url,
                duration_seconds=60,
                concurrent_users=50,
                ramp_up_seconds=20
            )),
            
            # Spike test
            SpikeScenario(LoadTestConfig(
                base_url=self.base_url,
                duration_seconds=45,
                concurrent_users=100
            )),
            
            # Stress test
            StressTestScenario(LoadTestConfig(
                base_url=self.base_url,
                duration_seconds=120,
                concurrent_users=200
            ))
        ]
        
        for scenario in scenarios:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running scenario: {scenario.__class__.__name__}")
            logger.info(f"{'='*60}\n")
            
            try:
                self.run_scenario(scenario)
            except Exception as e:
                logger.error(f"Scenario failed: {e}")
        
        self.print_overall_summary()
    
    def print_overall_summary(self):
        """Print summary of all test results."""
        print(f"\n{'='*60}")
        print("Overall Load Test Summary")
        print(f"{'='*60}")
        print(f"Total Scenarios: {len(self.results)}")
        print(f"Total Requests: {sum(r.total_requests for r in self.results)}")
        print(f"Total Successful: {sum(r.successful_requests for r in self.results)}")
        print(f"Total Failed: {sum(r.failed_requests for r in self.results)}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # Example usage
    runner = LoadTestRunner(base_url="http://localhost:8000")
    
    # Run a single scenario
    config = LoadTestConfig(
        duration_seconds=30,
        concurrent_users=10,
        think_time_seconds=1.0
    )
    scenario = ConstantLoadScenario(config)
    result = runner.run_scenario(scenario)
    
    # Or run all scenarios
    # runner.run_all_scenarios()
