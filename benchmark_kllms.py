"""
Performance Benchmarking Suite for KLLMS SDK
Tests performance characteristics, throughput, latency, and consensus quality.
"""

import time
import statistics
import concurrent.futures
import threading
import psutil
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
import dotenv

from k_llms import KLLMs
from openai import OpenAI

dotenv.load_dotenv(".env")


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""

    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    throughput_rps: float
    avg_consensus_quality: Optional[float] = None
    consensus_variance: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None


class SimpleTask(BaseModel):
    """Simple model for testing structured outputs."""

    task: str
    priority: int = Field(ge=1, le=5)
    completed: bool = False
    notes: Optional[str] = None


class PerformanceBenchmark:
    """Performance benchmarking suite for KLLMS SDK."""

    def __init__(self):
        self.kllms_client = KLLMs()
        self.openai_client = OpenAI()
        self.test_model = "gpt-4.1-nano"
        self.results: List[BenchmarkResult] = []

    def measure_system_resources(self) -> Dict[str, float]:
        """Measure current system resource usage."""
        process = psutil.Process()
        return {"memory_mb": process.memory_info().rss / 1024 / 1024, "cpu_percent": process.cpu_percent()}

    def single_request_latency_test(self, num_requests: int = 50) -> BenchmarkResult:
        """Test latency of single requests (no consensus)."""
        print(f"Running single request latency test ({num_requests} requests)...")

        response_times = []
        successful = 0
        failed = 0
        start_time = time.time()

        for i in range(num_requests):
            request_start = time.time()
            try:
                response = self.kllms_client.chat.completions.create(model=self.test_model, messages=[{"role": "user", "content": f"Say hello #{i + 1}"}], max_tokens=20)
                request_end = time.time()
                response_times.append(request_end - request_start)
                successful += 1
            except Exception as e:
                print(f"Request {i + 1} failed: {e}")
                failed += 1

        total_time = time.time() - start_time

        return BenchmarkResult(
            test_name="Single Request Latency",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            throughput_rps=successful / total_time if total_time > 0 else 0,
        )

    def consensus_latency_test(self, num_requests: int = 20, n: int = 3) -> BenchmarkResult:
        """Test latency and quality of consensus requests."""
        print(f"Running consensus latency test ({num_requests} requests, n={n})...")

        response_times = []
        consensus_qualities = []
        successful = 0
        failed = 0
        start_time = time.time()

        for i in range(num_requests):
            request_start = time.time()
            try:
                response = self.kllms_client.chat.completions.create(
                    model=self.test_model, messages=[{"role": "user", "content": f"What is 2+2? Request #{i + 1}"}], n=n, max_tokens=30, temperature=0.2
                )
                request_end = time.time()
                response_times.append(request_end - request_start)

                # Calculate consensus quality
                if response.likelihoods:
                    avg_likelihood = sum(response.likelihoods) / len(response.likelihoods)
                    consensus_qualities.append(avg_likelihood)

                successful += 1
            except Exception as e:
                print(f"Consensus request {i + 1} failed: {e}")
                failed += 1

        total_time = time.time() - start_time

        return BenchmarkResult(
            test_name=f"Consensus Latency (n={n})",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            throughput_rps=successful / total_time if total_time > 0 else 0,
            avg_consensus_quality=statistics.mean(consensus_qualities) if consensus_qualities else None,
            consensus_variance=statistics.stdev(consensus_qualities) if len(consensus_qualities) > 1 else None,
        )

    def structured_output_test(self, num_requests: int = 15) -> BenchmarkResult:
        """Test latency and accuracy of structured outputs."""
        print(f"Running structured output test ({num_requests} requests)...")

        response_times = []
        consensus_qualities = []
        successful = 0
        failed = 0
        start_time = time.time()

        for i in range(num_requests):
            request_start = time.time()
            try:
                response = self.kllms_client.chat.completions.parse(
                    model=self.test_model,
                    messages=[{"role": "user", "content": f"Create a task: 'Complete task #{i + 1}' with priority {(i % 5) + 1}"}],
                    response_format=SimpleTask,
                    n=2,
                    temperature=0.3,
                )
                request_end = time.time()
                response_times.append(request_end - request_start)

                # Validate structured output
                valid_outputs = 0
                for choice in response.choices:
                    if choice.message.parsed and isinstance(choice.message.parsed, SimpleTask):
                        valid_outputs += 1

                if response.likelihoods:
                    avg_likelihood = sum(response.likelihoods) / len(response.likelihoods)
                    consensus_qualities.append(avg_likelihood)

                successful += 1
            except Exception as e:
                print(f"Structured output request {i + 1} failed: {e}")
                failed += 1

        total_time = time.time() - start_time

        return BenchmarkResult(
            test_name="Structured Output",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            throughput_rps=successful / total_time if total_time > 0 else 0,
            avg_consensus_quality=statistics.mean(consensus_qualities) if consensus_qualities else None,
            consensus_variance=statistics.stdev(consensus_qualities) if len(consensus_qualities) > 1 else None,
        )

    def concurrent_request_test(self, num_requests: int = 20, max_workers: int = 5) -> BenchmarkResult:
        """Test performance under concurrent load."""
        print(f"Running concurrent request test ({num_requests} requests, {max_workers} workers)...")

        def make_request(request_id: int) -> Dict[str, Any]:
            """Make a single request and return timing info."""
            start_time = time.time()
            try:
                response = self.kllms_client.chat.completions.create(
                    model=self.test_model, messages=[{"role": "user", "content": f"Concurrent request #{request_id}"}], n=2, max_tokens=30
                )
                end_time = time.time()
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "consensus_quality": sum(response.likelihoods) / len(response.likelihoods) if response.likelihoods else None,
                }
            except Exception as e:
                return {"success": False, "response_time": time.time() - start_time, "error": str(e)}

        start_time = time.time()
        response_times = []
        consensus_qualities = []
        successful = 0
        failed = 0

        # Measure system resources before
        resources_before = self.measure_system_resources()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                response_times.append(result["response_time"])

                if result["success"]:
                    successful += 1
                    if result["consensus_quality"] is not None:
                        consensus_qualities.append(result["consensus_quality"])
                else:
                    failed += 1
                    print(f"Concurrent request failed: {result.get('error', 'Unknown error')}")

        total_time = time.time() - start_time

        # Measure system resources after
        resources_after = self.measure_system_resources()

        return BenchmarkResult(
            test_name=f"Concurrent Load (workers={max_workers})",
            total_requests=num_requests,
            successful_requests=successful,
            failed_requests=failed,
            total_time=total_time,
            avg_response_time=statistics.mean(response_times) if response_times else 0,
            min_response_time=min(response_times) if response_times else 0,
            max_response_time=max(response_times) if response_times else 0,
            throughput_rps=successful / total_time if total_time > 0 else 0,
            avg_consensus_quality=statistics.mean(consensus_qualities) if consensus_qualities else None,
            consensus_variance=statistics.stdev(consensus_qualities) if len(consensus_qualities) > 1 else None,
            memory_usage_mb=resources_after["memory_mb"] - resources_before["memory_mb"],
            cpu_usage_percent=resources_after["cpu_percent"],
        )

    def consensus_scaling_test(self) -> List[BenchmarkResult]:
        """Test how performance scales with different consensus values."""
        print("Running consensus scaling test...")

        consensus_values = [1, 2, 3, 5, 7, 10]
        results = []

        for n in consensus_values:
            print(f"  Testing n={n}...")

            response_times = []
            consensus_qualities = []
            successful = 0
            failed = 0
            num_requests = 10  # Fewer requests for scaling test

            start_time = time.time()

            for i in range(num_requests):
                request_start = time.time()
                try:
                    response = self.kllms_client.chat.completions.create(
                        model=self.test_model, messages=[{"role": "user", "content": "Count from 1 to 5"}], n=n, max_tokens=50, temperature=0.5
                    )
                    request_end = time.time()
                    response_times.append(request_end - request_start)

                    if response.likelihoods:
                        avg_likelihood = sum(response.likelihoods) / len(response.likelihoods)
                        variance = max(response.likelihoods) - min(response.likelihoods)
                        consensus_qualities.append(avg_likelihood)

                    successful += 1
                except Exception as e:
                    failed += 1
                    print(f"    Request failed: {e}")

            total_time = time.time() - start_time

            result = BenchmarkResult(
                test_name=f"Consensus Scaling (n={n})",
                total_requests=num_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time=total_time,
                avg_response_time=statistics.mean(response_times) if response_times else 0,
                min_response_time=min(response_times) if response_times else 0,
                max_response_time=max(response_times) if response_times else 0,
                throughput_rps=successful / total_time if total_time > 0 else 0,
                avg_consensus_quality=statistics.mean(consensus_qualities) if consensus_qualities else None,
                consensus_variance=statistics.stdev(consensus_qualities) if len(consensus_qualities) > 1 else None,
            )

            results.append(result)

        return results

    def temperature_impact_test(self) -> List[BenchmarkResult]:
        """Test how temperature affects consensus quality and performance."""
        print("Running temperature impact test...")

        temperatures = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0]
        results = []

        for temp in temperatures:
            print(f"  Testing temperature={temp}...")

            response_times = []
            consensus_qualities = []
            variances = []
            successful = 0
            failed = 0
            num_requests = 8

            start_time = time.time()

            for i in range(num_requests):
                request_start = time.time()
                try:
                    response = self.kllms_client.chat.completions.create(
                        model=self.test_model,
                        messages=[{"role": "user", "content": "Write one sentence about artificial intelligence."}],
                        n=4,
                        temperature=temp,
                        max_tokens=100,
                    )
                    request_end = time.time()
                    response_times.append(request_end - request_start)

                    if response.likelihoods:
                        avg_likelihood = sum(response.likelihoods) / len(response.likelihoods)
                        variance = max(response.likelihoods) - min(response.likelihoods)
                        consensus_qualities.append(avg_likelihood)
                        variances.append(variance)

                    successful += 1
                except Exception as e:
                    failed += 1
                    print(f"    Request failed: {e}")

            total_time = time.time() - start_time

            result = BenchmarkResult(
                test_name=f"Temperature Impact (temp={temp})",
                total_requests=num_requests,
                successful_requests=successful,
                failed_requests=failed,
                total_time=total_time,
                avg_response_time=statistics.mean(response_times) if response_times else 0,
                min_response_time=min(response_times) if response_times else 0,
                max_response_time=max(response_times) if response_times else 0,
                throughput_rps=successful / total_time if total_time > 0 else 0,
                avg_consensus_quality=statistics.mean(consensus_qualities) if consensus_qualities else None,
                consensus_variance=statistics.mean(variances) if variances else None,
            )

            results.append(result)

        return results

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks."""
        print("=" * 60)
        print("KLLMS SDK Performance Benchmark Suite")
        print("=" * 60)

        all_results = []

        # Individual tests
        all_results.append(self.single_request_latency_test())
        all_results.append(self.consensus_latency_test())
        all_results.append(self.structured_output_test())
        all_results.append(self.concurrent_request_test())

        # Scaling tests
        all_results.extend(self.consensus_scaling_test())
        all_results.extend(self.temperature_impact_test())

        self.results = all_results
        return all_results

    def print_results(self):
        """Print formatted benchmark results."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 80)

        for result in self.results:
            print(f"\n📊 {result.test_name}")
            print(f"   Requests: {result.successful_requests}/{result.total_requests} successful")
            print(f"   Total Time: {result.total_time:.2f}s")
            print(f"   Avg Response Time: {result.avg_response_time:.3f}s")
            print(f"   Min/Max Response Time: {result.min_response_time:.3f}s / {result.max_response_time:.3f}s")
            print(f"   Throughput: {result.throughput_rps:.2f} req/s")

            if result.avg_consensus_quality is not None:
                print(f"   Avg Consensus Quality: {result.avg_consensus_quality:.3f}")

            if result.consensus_variance is not None:
                print(f"   Consensus Variance: {result.consensus_variance:.3f}")

            if result.memory_usage_mb is not None:
                print(f"   Memory Usage: {result.memory_usage_mb:.1f} MB")

            if result.cpu_usage_percent is not None:
                print(f"   CPU Usage: {result.cpu_usage_percent:.1f}%")

        print("\n" + "=" * 80)
        print("PERFORMANCE INSIGHTS")
        print("=" * 80)

        # Find best and worst performing tests
        single_req_results = [r for r in self.results if "Single Request" in r.test_name]
        consensus_results = [r for r in self.results if "Consensus" in r.test_name and "Scaling" in r.test_name]

        if single_req_results:
            sr = single_req_results[0]
            print(f"🚀 Single Request Performance: {sr.avg_response_time:.3f}s avg, {sr.throughput_rps:.2f} req/s")

        if consensus_results:
            best_consensus = min(consensus_results, key=lambda x: x.avg_response_time)
            worst_consensus = max(consensus_results, key=lambda x: x.avg_response_time)
            print(f"🔄 Best Consensus Performance: {best_consensus.test_name} - {best_consensus.avg_response_time:.3f}s")
            print(f"⚠️  Worst Consensus Performance: {worst_consensus.test_name} - {worst_consensus.avg_response_time:.3f}s")

        # Temperature analysis
        temp_results = [r for r in self.results if "Temperature" in r.test_name]
        if temp_results:
            best_quality = max(temp_results, key=lambda x: x.avg_consensus_quality or 0)
            print(f"🌡️  Best Consensus Quality: {best_quality.test_name} - {best_quality.avg_consensus_quality:.3f}")

    def export_results(self, filename: str = None):
        """Export results to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kllms_benchmark_{timestamp}.json"

        export_data = {"timestamp": datetime.now().isoformat(), "benchmark_version": "1.0", "results": []}

        for result in self.results:
            export_data["results"].append(
                {
                    "test_name": result.test_name,
                    "total_requests": result.total_requests,
                    "successful_requests": result.successful_requests,
                    "failed_requests": result.failed_requests,
                    "total_time": result.total_time,
                    "avg_response_time": result.avg_response_time,
                    "min_response_time": result.min_response_time,
                    "max_response_time": result.max_response_time,
                    "throughput_rps": result.throughput_rps,
                    "avg_consensus_quality": result.avg_consensus_quality,
                    "consensus_variance": result.consensus_variance,
                    "memory_usage_mb": result.memory_usage_mb,
                    "cpu_usage_percent": result.cpu_usage_percent,
                }
            )

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\n📁 Results exported to: {filename}")


def main():
    """Run the complete benchmark suite."""
    benchmark = PerformanceBenchmark()

    try:
        benchmark.run_all_benchmarks()
        benchmark.print_results()
        benchmark.export_results()

        print("\n✅ Benchmark suite completed successfully!")

    except Exception as e:
        print(f"\n❌ Benchmark suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
