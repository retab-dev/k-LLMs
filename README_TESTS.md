# KLLMS SDK Test Suite

This directory contains comprehensive tests for the KLLMS SDK, including examples, unit tests, performance benchmarks, and a test runner utility.

## Test Files Overview

### 📋 Test Files

| File | Purpose | Type | Coverage |
|------|---------|------|----------|
| `example.py` | Comprehensive examples and integration tests | Demo/Integration | Complex nested models, error handling, real-world scenarios |
| `test_kllms_comprehensive.py` | Structured unit tests with assertions | Unit/Integration | API functionality, consensus, structured outputs |
| `benchmark_kllms.py` | Performance and load testing | Performance | Latency, throughput, consensus scaling |
| `run_tests.py` | Test runner utility | Utility | Orchestrates all test types |
| `README_TESTS.md` | This documentation | Documentation | Usage instructions and test descriptions |

## 🚀 Quick Start

### Run All Tests
```bash
python run_tests.py all
```

### Run Quick Smoke Test
```bash
python run_tests.py quick
```

### Run Specific Test Types
```bash
python run_tests.py example      # Demo/example tests
python run_tests.py comprehensive  # Unit tests with assertions
python run_tests.py benchmark    # Performance benchmarks
python run_tests.py consensus    # Quick consensus validation
```

## 📝 Test Descriptions

### 1. Example Tests (`example.py`)

**Purpose**: Comprehensive demonstration of KLLMS SDK capabilities with real-world scenarios.

**What it tests**:
- ✅ Basic OpenAI compatibility
- ✅ KLLMS consensus functionality
- ✅ Complex structured outputs with nested Pydantic models
- ✅ Error handling and edge cases
- ✅ Temperature and creativity variations
- ✅ Simple data types with consensus
- ✅ Multi-turn conversations
- ✅ Performance timing
- ✅ Consensus quality analysis
- ✅ Stress testing with rapid requests

**Key Features**:
- Tests 14 different scenarios
- Complex nested Pydantic models (Company, Department, Employee)
- Edge case handling (invalid inputs, extreme parameters)
- Consensus quality analysis
- Temperature impact testing

**Example Models**:
```python
class Employee(BaseModel):
    id: int
    name: str
    title: str
    salary: float
    skills: List[Skill]
    contact: ContactInfo
    address: Address
    # ... more fields
```

### 2. Comprehensive Unit Tests (`test_kllms_comprehensive.py`)

**Purpose**: Structured unit tests with proper assertions for continuous integration.

**What it tests**:
- ✅ Basic chat completion functionality
- ✅ Consensus mechanism validation
- ✅ Structured output parsing
- ✅ Temperature effects on consensus
- ✅ Mathematical reasoning consistency  
- ✅ Error handling (invalid models, empty messages)
- ✅ Edge cases in structured data
- ✅ Performance requirements
- ✅ Real-world data extraction
- ✅ Code generation with consensus

**Key Features**:
- Uses Python `unittest` framework
- Proper assertions and test isolation
- Automatic result reporting
- CI/CD compatible
- 15+ individual test methods

**Example Test**:
```python
def test_consensus_functionality(self):
    response = self.kllms_client.chat.completions.create(
        model=self.test_model,
        messages=[{"role": "user", "content": "What is 2+2?"}],
        n_consensus=3,
        temperature=0.1
    )
    
    self.assertEqual(len(response.choices), 3)
    self.assertEqual(len(response.likelihoods), 3)
    # ... more assertions
```

### 3. Performance Benchmarks (`benchmark_kllms.py`)

**Purpose**: Measure and analyze performance characteristics under various conditions.

**What it tests**:
- ⚡ Single request latency
- ⚡ Consensus request performance  
- ⚡ Structured output timing
- ⚡ Concurrent load handling
- ⚡ Consensus scaling (1-10 consensus values)
- ⚡ Temperature impact on performance
- ⚡ System resource usage
- ⚡ Throughput measurements

**Key Metrics**:
- Response time (avg/min/max)
- Throughput (requests per second)
- Consensus quality scores
- Memory and CPU usage
- Consensus variance analysis

**Example Output**:
```
📊 Single Request Latency
   Requests: 48/50 successful
   Total Time: 25.67s
   Avg Response Time: 0.535s
   Throughput: 1.87 req/s
   Avg Consensus Quality: 0.847
```

### 4. Test Runner (`run_tests.py`)

**Purpose**: Unified interface to run all test types with different configurations.

**Features**:
- Command-line interface
- Multiple test type options
- Result summarization
- Exit codes for CI/CD
- Verbose output option

**Usage Examples**:
```bash
# Quick validation
python run_tests.py quick

# Full test suite
python run_tests.py all

# Performance only
python run_tests.py benchmark

# With verbose output
python run_tests.py comprehensive --verbose
```

## 🧪 Test Coverage

### Core Functionality
- [x] Basic chat completions
- [x] Consensus mechanism
- [x] Structured output parsing
- [x] Multi-turn conversations
- [x] Temperature controls
- [x] Token limits
- [x] Model selection

### Advanced Features
- [x] Complex nested Pydantic models
- [x] Optional fields and defaults
- [x] Enum validations
- [x] List and Union types
- [x] Field constraints
- [x] Error handling

### Performance & Reliability
- [x] Latency measurements
- [x] Throughput testing
- [x] Concurrent request handling
- [x] Resource usage monitoring
- [x] Consensus quality analysis
- [x] Stress testing

### Error Scenarios
- [x] Invalid model names
- [x] Empty message arrays
- [x] Invalid consensus values
- [x] Network timeouts
- [x] Malformed responses
- [x] Validation errors

## 📊 Benchmark Results

Typical performance characteristics (may vary by environment):

| Test Type | Avg Response Time | Throughput | Consensus Quality |
|-----------|------------------|------------|------------------|
| Single Request | ~0.5s | ~2.0 req/s | N/A |
| Consensus (n=3) | ~1.2s | ~0.8 req/s | ~0.85 |
| Structured Output | ~1.5s | ~0.7 req/s | ~0.82 |
| Concurrent (5 workers) | ~2.1s | ~2.4 req/s | ~0.80 |

## 🔧 Setup Requirements

### Dependencies
```bash
pip install k_llms openai pydantic psutil python-dotenv
```

### Environment
- Set up `.env` file with OpenAI API key
- Ensure network access to OpenAI API
- Python 3.8+ recommended

### Model Availability
Tests use `gpt-4.1-nano` by default. Ensure your API key has access to this model.

## 🚨 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Ensure KLLMS SDK is properly installed
pip install -e .  # If developing locally
```

**2. API Key Issues**
```bash
# Check .env file
echo $OPENAI_API_KEY
```

**3. Model Access**
```
Error: Model 'gpt-4.1-nano' not found
# Update test files to use available model
```

**4. Timeout Errors**
```
# Increase timeout in test files
max_tokens=50  # Reduce token limits
```

### Performance Issues

If tests are running slowly:
1. Reduce number of requests in test files
2. Use smaller `n_consensus` values
3. Set lower `max_tokens` limits
4. Check network connection

## 📈 Interpreting Results

### Consensus Quality Scores
- **0.9-1.0**: Excellent consensus (high agreement)
- **0.8-0.9**: Good consensus (minor variations)
- **0.7-0.8**: Moderate consensus (some disagreement)
- **<0.7**: Poor consensus (high disagreement)

### Performance Metrics
- **Response Time**: Time from request to response
- **Throughput**: Requests processed per second
- **Consensus Variance**: Spread of likelihood scores
- **Success Rate**: Percentage of successful requests

## 🔄 Continuous Integration

For CI/CD integration:

```yaml
# Example GitHub Actions
- name: Run KLLMS Tests
  run: |
    cd backend/lib/k_llms
    python run_tests.py comprehensive
    
- name: Run Performance Benchmarks
  run: |
    cd backend/lib/k_llms
    python run_tests.py benchmark
```

Exit codes:
- `0`: All tests passed
- `1`: One or more tests failed

## 📚 Further Documentation

- [KLLMS SDK Documentation](../README.md)
- [Pydantic Models Guide](https://docs.pydantic.dev/)
- [OpenAI API Reference](https://platform.openai.com/docs/)

## 🤝 Contributing

When adding new tests:
1. Follow existing patterns
2. Add proper assertions
3. Include error handling
4. Update this README
5. Test in both success and failure scenarios

---

**Last Updated**: December 2024  
**Test Suite Version**: 1.0  
**Compatible KLLMS SDK**: All versions 