#!/usr/bin/env python3
"""
KLLMS SDK Test Runner
Simple script to run various test suites for the KLLMS SDK.
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path
import dotenv

# Load environment variables
dotenv.load_dotenv()


def run_example_tests():
    """Run the example/demo tests."""
    print("🧪 Running KLLMS SDK Example Tests...")
    print("=" * 50)

    try:
        result = subprocess.run([sys.executable, "example.py"], capture_output=False, text=True, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run example tests: {e}")
        return False


def run_comprehensive_tests():
    """Run the comprehensive unit tests."""
    print("🧪 Running KLLMS SDK Comprehensive Tests...")
    print("=" * 50)

    try:
        result = subprocess.run([sys.executable, "test_kllms_comprehensive.py"], capture_output=False, text=True, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run comprehensive tests: {e}")
        return False


def run_benchmark_tests():
    """Run the performance benchmark tests."""
    print("📊 Running KLLMS SDK Performance Benchmarks...")
    print("=" * 50)

    try:
        result = subprocess.run([sys.executable, "benchmark_kllms.py"], capture_output=False, text=True, cwd=Path(__file__).parent)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Failed to run benchmark tests: {e}")
        return False


def run_quick_test():
    """Run a quick smoke test to verify basic functionality."""
    print("⚡ Running Quick KLLMS SDK Smoke Test...")
    print("=" * 50)

    try:
        from k_llms import KLLMs

        # Quick test
        client = KLLMs()
        response = client.chat.completions.create(model="gpt-4.1-nano", messages=[{"role": "user", "content": "Say 'Hello, KLLMS is working!'"}], max_tokens=20)

        if response and response.choices and response.choices[0].message.content:
            print("✅ Quick test passed!")
            print(f"Response: {response.choices[0].message.content}")
            return True
        else:
            print("❌ Quick test failed - no valid response")
            return False

    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False


def run_consensus_test():
    """Run a quick consensus test."""
    print("🔄 Running Quick Consensus Test...")
    print("=" * 30)

    try:
        from k_llms import KLLMs

        client = KLLMs()
        response = client.chat.completions.create(model="gpt-4.1-nano", messages=[{"role": "user", "content": "What is 2+2?"}], n=3, temperature=0.1)

        if response and response.choices and len(response.choices) > 0:
            print("✅ Consensus test passed!")
            print(f"Consensus responses: {len(response.choices)}")
            print(f"Likelihoods: {response.likelihoods}")
            return True
        else:
            print("❌ Consensus test failed - no valid response")
            return False

    except Exception as e:
        print(f"❌ Consensus test failed:")
        print(f"   Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="KLLMS SDK Test Runner")
    parser.add_argument("test_type", choices=["all", "quick", "example", "comprehensive", "benchmark", "consensus"], help="Type of test to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    print("🚀 KLLMS SDK Test Runner")
    print("=" * 60)
    print(f"Test Type: {args.test_type}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = {}

    if args.test_type == "quick":
        results["quick"] = run_quick_test()
        results["consensus"] = run_consensus_test()

    elif args.test_type == "example":
        results["example"] = run_example_tests()

    elif args.test_type == "comprehensive":
        results["comprehensive"] = run_comprehensive_tests()

    elif args.test_type == "benchmark":
        results["benchmark"] = run_benchmark_tests()

    elif args.test_type == "consensus":
        results["consensus"] = run_consensus_test()

    elif args.test_type == "all":
        print("\n🎯 Running ALL test suites...")
        results["quick"] = run_quick_test()
        print("\n" + "-" * 60)
        results["consensus"] = run_consensus_test()
        print("\n" + "-" * 60)
        results["example"] = run_example_tests()
        print("\n" + "-" * 60)
        results["comprehensive"] = run_comprehensive_tests()
        print("\n" + "-" * 60)
        results["benchmark"] = run_benchmark_tests()

    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 60)

    total_tests = len(results)
    passed_tests = sum(1 for result in results.values() if result)
    failed_tests = total_tests - passed_tests

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.upper():<15} {status}")

    print("-" * 60)
    print(f"Total: {total_tests} | Passed: {passed_tests} | Failed: {failed_tests}")

    if failed_tests == 0:
        print("🎉 ALL TESTS PASSED!")
        sys.exit(0)
    else:
        print(f"⚠️  {failed_tests} test(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
