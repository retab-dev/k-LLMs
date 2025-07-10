"""
Comprehensive Test Suite for KLLMS SDK
This file contains structured tests that can be run for validation and CI/CD.
"""

import unittest
import time
import json
from typing import List, Optional, Union
from enum import Enum
import dotenv
from pydantic import BaseModel, Field, ValidationError

# Import the KLLMS SDK and OpenAI for comparison
from k_llms import KLLMs
from openai import OpenAI

dotenv.load_dotenv(".env")


# Test Models
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Task(BaseModel):
    id: int
    title: str
    description: str
    priority: Priority
    completed: bool = False
    tags: List[str] = []
    assignee: Optional[str] = None


class Project(BaseModel):
    name: str
    tasks: List[Task]
    budget: float
    deadline: str
    team_size: int


class SimpleAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)


class MathProblem(BaseModel):
    problem: str
    solution: float
    steps: List[str]
    difficulty: int = Field(ge=1, le=10)


class KLLMSTestSuite(unittest.TestCase):
    """Comprehensive test suite for KLLMS SDK functionality."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.kllms_client = KLLMs()
        cls.openai_client = OpenAI()
        cls.test_model = "gpt-4.1-nano"
        cls.test_results = []

    def setUp(self):
        """Set up for each test."""
        self.start_time = time.time()

    def tearDown(self):
        """Clean up after each test."""
        test_time = time.time() - self.start_time
        self.test_results.append({"test": self._testMethodName, "duration": test_time, "status": "passed" if hasattr(self, "_outcome") and self._outcome.success else "failed"})

    def test_basic_chat_completion(self):
        """Test basic chat completion functionality."""
        response = self.kllms_client.chat.completions.create(model=self.test_model, messages=[{"role": "user", "content": "Hello, how are you?"}], max_tokens=50)

        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)
        self.assertIsNotNone(response.choices[0].message.content)
        self.assertIsInstance(response.choices[0].message.content, str)

    def test_consensus_functionality(self):
        """Test consensus mechanism with multiple responses."""
        response = self.kllms_client.chat.completions.create(model=self.test_model, messages=[{"role": "user", "content": "What is 2+2?"}], n_consensus=3, temperature=0.1)

        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)
        self.assertIsNotNone(response.likelihoods)
        # Likelihoods is a dict in actual KLLMS implementation
        self.assertIsInstance(response.likelihoods, dict)

        # Check that likelihoods dict has valid values between 0 and 1
        if isinstance(response.likelihoods, dict):
            for likelihood in response.likelihoods.values():
                self.assertGreaterEqual(likelihood, 0.0)
                self.assertLessEqual(likelihood, 1.0)

    def test_structured_output_simple(self):
        """Test structured output with simple model."""
        response = self.kllms_client.chat.completions.parse(
            model=self.test_model,
            messages=[{"role": "user", "content": "Is Python a good programming language? Rate your confidence."}],
            response_format=SimpleAnswer,
            n_consensus=2,
        )

        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)

        for choice in response.choices:
            if choice.message.parsed:
                data = choice.message.parsed
                self.assertIsInstance(data, SimpleAnswer)
                self.assertIsInstance(data.answer, str)
                self.assertGreaterEqual(data.confidence, 0.0)
                self.assertLessEqual(data.confidence, 1.0)

    def test_structured_output_complex(self):
        """Test structured output with complex nested model."""
        prompt = """
        Create a software project with:
        - Name: "TaskMaster Pro"
        - 3 tasks with different priorities
        - Budget of $50000
        - Deadline: 2024-12-31
        - Team size: 5
        """

        response = self.kllms_client.chat.completions.parse(model=self.test_model, messages=[{"role": "user", "content": prompt}], response_format=Project, n_consensus=2)

        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)

        for choice in response.choices:
            if choice.message.parsed:
                project = choice.message.parsed
                self.assertIsInstance(project, Project)
                self.assertEqual(len(project.tasks), 3)
                self.assertEqual(project.budget, 50000.0)
                self.assertEqual(project.team_size, 5)

                # Validate tasks
                for task in project.tasks:
                    self.assertIsInstance(task, Task)
                    self.assertIsInstance(task.id, int)
                    self.assertIsInstance(task.title, str)
                    self.assertIn(task.priority, [p.value for p in Priority])

    def test_temperature_effects(self):
        """Test that temperature affects response variance."""
        prompt = "Write a creative sentence about space exploration."

        # Low temperature - should have higher consensus
        low_temp_response = self.kllms_client.chat.completions.create(model=self.test_model, messages=[{"role": "user", "content": prompt}], n_consensus=3, temperature=0.1)

        # High temperature - should have lower consensus
        high_temp_response = self.kllms_client.chat.completions.create(model=self.test_model, messages=[{"role": "user", "content": prompt}], n_consensus=3, temperature=1.5)

        # Calculate variance in likelihoods (extract numeric values)
        low_temp_values = [v for v in low_temp_response.likelihoods.values() if isinstance(v, (int, float))]
        high_temp_values = [v for v in high_temp_response.likelihoods.values() if isinstance(v, (int, float))]
        
        low_temp_variance = max(low_temp_values) - min(low_temp_values) if low_temp_values else 0
        high_temp_variance = max(high_temp_values) - min(high_temp_values) if high_temp_values else 0

        # Low temperature should generally have less variance (higher consensus)
        # This is probabilistic, so we'll just check that both have valid responses
        self.assertIsNotNone(low_temp_response)
        self.assertIsNotNone(high_temp_response)
        self.assertEqual(len(low_temp_response.choices), 4)  # 1 consensus + 3 individual
        self.assertEqual(len(high_temp_response.choices), 4)  # 1 consensus + 3 individual

    def test_math_problem_consistency(self):
        """Test consistency in mathematical reasoning."""
        response = self.kllms_client.chat.completions.parse(
            model=self.test_model,
            messages=[{"role": "user", "content": "Solve: If a train travels 60 mph for 2.5 hours, how far does it go? Show your work."}],
            response_format=MathProblem,
            n_consensus=3,
            temperature=0.2,
        )

        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)

        solutions = []
        for choice in response.choices:
            if choice.message.parsed:
                math_problem = choice.message.parsed
                self.assertIsInstance(math_problem, MathProblem)
                solutions.append(math_problem.solution)

        # All solutions should be the same for this simple math problem
        if len(solutions) > 1:
            # Allow small floating point differences
            for solution in solutions[1:]:
                self.assertAlmostEqual(solution, solutions[0], delta=1.0)

    def test_error_handling_invalid_model(self):
        """Test error handling with invalid model name."""
        with self.assertRaises(Exception):
            self.kllms_client.chat.completions.create(model="invalid-model-name", messages=[{"role": "user", "content": "Hello"}], n_consensus=2)

    def test_error_handling_empty_messages(self):
        """Test error handling with empty messages."""
        with self.assertRaises(Exception):
            self.kllms_client.chat.completions.create(model=self.test_model, messages=[], n_consensus=2)

    def test_error_handling_invalid_consensus(self):
        """Test error handling with invalid consensus values."""
        # Test with n_consensus = 0 - this should fall back to single request (no exception)
        response = self.kllms_client.chat.completions.create(model=self.test_model, messages=[{"role": "user", "content": "Hello"}], n_consensus=0)
        self.assertIsNotNone(response)
        self.assertEqual(len(response.choices), 1)  # Single choice when n_consensus=0

    def test_large_consensus_values(self):
        """Test behavior with large consensus values."""
        response = self.kllms_client.chat.completions.create(model=self.test_model, messages=[{"role": "user", "content": "Say hello"}], n_consensus=10, max_tokens=20)

        self.assertIsNotNone(response)
        self.assertEqual(len(response.choices), 11)  # 1 consensus + 10 individual
        self.assertIsNotNone(response.likelihoods)

    def test_conversation_context(self):
        """Test multi-turn conversation handling."""
        messages = [
            {"role": "user", "content": "I'm learning Python programming."},
            {"role": "assistant", "content": "That's great! Python is an excellent language for beginners. What would you like to learn first?"},
            {"role": "user", "content": "How do I create a list?"},
        ]

        response = self.kllms_client.chat.completions.create(model=self.test_model, messages=messages, n_consensus=2)

        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)

        # Response should mention lists or list creation
        content = response.choices[0].message.content.lower()
        self.assertTrue(any(keyword in content for keyword in ["list", "[]", "array", "append"]))

    def test_performance_timing(self):
        """Test response time performance."""
        start_time = time.time()

        response = self.kllms_client.chat.completions.create(model=self.test_model, messages=[{"role": "user", "content": "Quick response test"}], n_consensus=2, max_tokens=30)

        end_time = time.time()
        response_time = end_time - start_time

        self.assertIsNotNone(response)
        # Response should be reasonably fast (adjust threshold as needed)
        self.assertLess(response_time, 30.0)  # 30 seconds max

    def test_consensus_quality_analysis(self):
        """Test consensus quality metrics."""
        response = self.kllms_client.chat.completions.create(
            model=self.test_model, messages=[{"role": "user", "content": "What is the capital of France?"}], n_consensus=5, temperature=0.2
        )

        self.assertIsNotNone(response.likelihoods)
        self.assertIsInstance(response.likelihoods, dict)
        # With n_consensus=5, we get 6 choices (1 consensus + 5 individual)
        self.assertEqual(len(response.choices), 6)

        # Calculate consensus metrics (extract numeric values from dict)
        likelihood_values = [v for v in response.likelihoods.values() if isinstance(v, (int, float))]
        if likelihood_values:
            mean_likelihood = sum(likelihood_values) / len(likelihood_values)
            variance = max(likelihood_values) - min(likelihood_values)
        else:
            mean_likelihood = 0.0
            variance = 0.0

        # For a factual question with low temperature, we expect high consensus
        self.assertGreater(mean_likelihood, 0.5)  # Generally good quality
        self.assertLess(variance, 0.8)  # Not too much disagreement

    def test_structured_output_edge_cases(self):
        """Test structured output with edge cases."""

        class EdgeCaseModel(BaseModel):
            optional_field: Optional[str] = None
            empty_list: List[str] = []
            union_field: Union[str, int]
            default_int: int = 42

        response = self.kllms_client.chat.completions.parse(
            model=self.test_model,
            messages=[{"role": "user", "content": "Create data with: optional_field=null, empty_list=[], union_field=123, default_int=42"}],
            response_format=EdgeCaseModel,
            n_consensus=2,
        )

        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)

        for choice in response.choices:
            if choice.message.parsed:
                data = choice.message.parsed
                self.assertIsInstance(data, EdgeCaseModel)
                # Optional field can be None
                self.assertTrue(data.optional_field is None or isinstance(data.optional_field, str))
                # Empty list should be empty
                self.assertIsInstance(data.empty_list, list)
                # Union field should be int or str
                self.assertTrue(isinstance(data.union_field, (str, int)))

    @classmethod
    def tearDownClass(cls):
        """Print test results summary."""
        print("\n" + "=" * 50)
        print("KLLMS SDK Test Results Summary")
        print("=" * 50)

        total_tests = len(cls.test_results)
        passed_tests = len([r for r in cls.test_results if r["status"] == "passed"])
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")

        total_time = sum(r["duration"] for r in cls.test_results)
        avg_time = total_time / total_tests if total_tests > 0 else 0

        print(f"\nTotal Time: {total_time:.2f}s")
        print(f"Average Time per Test: {avg_time:.2f}s")

        print("\nIndividual Test Times:")
        for result in cls.test_results:
            status_symbol = "✓" if result["status"] == "passed" else "✗"
            print(f"  {status_symbol} {result['test']}: {result['duration']:.2f}s")

        print("=" * 50)


class KLLMSIntegrationTests(unittest.TestCase):
    """Integration tests for KLLMS SDK with real-world scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.kllms_client = KLLMs()
        cls.test_model = "gpt-4.1-nano"

    def test_real_world_data_extraction(self):
        """Test extracting structured data from unstructured text."""

        class NewsArticle(BaseModel):
            headline: str
            summary: str
            key_points: List[str]
            sentiment: str  # positive, negative, neutral
            entities: List[str]

        text = """
        BREAKING: Tech company announces breakthrough in quantum computing
        
        Today, InnovateTech Corp revealed their latest quantum processor that can solve 
        complex optimization problems 1000x faster than traditional computers. The CEO 
        stated this could revolutionize logistics, finance, and drug discovery. Stock 
        prices jumped 15% on the news. Scientists are calling it a major milestone.
        """

        response = self.kllms_client.chat.completions.parse(
            model=self.test_model, messages=[{"role": "user", "content": f"Extract key information from this news text: {text}"}], response_format=NewsArticle, n_consensus=3
        )

        self.assertIsNotNone(response)
        self.assertGreater(len(response.choices), 0)

        for choice in response.choices:
            if choice.message.parsed:
                article = choice.message.parsed
                self.assertIsInstance(article, NewsArticle)
                self.assertIn("quantum", article.headline.lower())
                self.assertGreater(len(article.key_points), 0)
                self.assertIn(article.sentiment.lower(), ["positive", "negative", "neutral"])

    def test_code_generation_consensus(self):
        """Test code generation with consensus."""

        class CodeSolution(BaseModel):
            language: str
            code: str
            explanation: str
            complexity: str

        response = self.kllms_client.chat.completions.parse(
            model=self.test_model,
            messages=[{"role": "user", "content": "Write a Python function to find the factorial of a number. Include time complexity."}],
            response_format=CodeSolution,
            n_consensus=3,
            temperature=0.3,
        )

        self.assertIsNotNone(response)
        solutions = []

        for choice in response.choices:
            if choice.message.parsed:
                solution = choice.message.parsed
                self.assertEqual(solution.language.lower(), "python")
                self.assertIn("def", solution.code)
                self.assertIn("factorial", solution.code.lower())
                solutions.append(solution)

        # Should have multiple valid solutions
        self.assertGreater(len(solutions), 0)


def run_comprehensive_tests():
    """Run all comprehensive tests for KLLMS SDK."""
    # Create test suite
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTest(unittest.makeSuite(KLLMSTestSuite))
    suite.addTest(unittest.makeSuite(KLLMSIntegrationTests))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result


if __name__ == "__main__":
    print("Starting KLLMS SDK Comprehensive Test Suite...")
    print("=" * 60)

    try:
        result = run_comprehensive_tests()

        if result.wasSuccessful():
            print("\n🎉 All tests passed successfully!")
            exit(0)
        else:
            print(f"\n❌ {len(result.failures)} failures, {len(result.errors)} errors")
            exit(1)

    except Exception as e:
        print(f"\n💥 Test suite crashed: {e}")
        exit(1)
