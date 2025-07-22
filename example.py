# Example usage of KLLMS OpenAI Wrapper - Complex Nested Models Test

from k_llms import KLLMs
from openai import OpenAI
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
from datetime import datetime
from enum import Enum
import dotenv
import json

dotenv.load_dotenv(".env")

# Initialize clients
kllms_client = KLLMs()
openai_client = OpenAI()


# Complex nested models for testing
class SkillLevel(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class Skill(BaseModel):
    name: str
    level: SkillLevel
    years_experience: float
    certifications: Optional[List[str]] = []


class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None


class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"


class Employee(BaseModel):
    id: int
    name: str
    title: str
    salary: float
    hire_date: str  # ISO format
    skills: List[Skill]
    contact: ContactInfo
    address: Address
    is_remote: bool = False
    manager_id: Optional[int] = None
    direct_reports: Optional[List[int]] = []


class Department(BaseModel):
    name: str
    budget: float
    employees: List[Employee]
    head_of_department: int  # employee ID
    location: Address
    projects: List[str]


class Company(BaseModel):
    name: str
    founded_year: int
    headquarters: Address
    departments: List[Department]
    total_employees: int
    annual_revenue: float
    stock_symbol: Optional[str] = None
    public_company: bool = False


# Test 1: Basic OpenAI request
print("=== Test 1: Basic OpenAI Request ===")
try:
    response = openai_client.chat.completions.create(model="gpt-4.1-nano", messages=[{"role": "user", "content": "Hello! What's the weather like?"}])
    print("OpenAI response:", response.choices[0].message.content)
except Exception as e:
    print(f"OpenAI request failed: {e}")

# Test 2: KLLMS consensus request
print("\n=== Test 2: KLLMS Consensus Request ===")
try:
    consensus_response = kllms_client.chat.completions.create(
        model="gpt-4.1-nano", messages=[{"role": "user", "content": "What is the most efficient sorting algorithm for large datasets?"}], n=3, temperature=1.0
    )
    print("KLLMS consensus response:", consensus_response.choices[0].message.content)
    print("KLLMS consensus response likelihoods:", consensus_response.likelihoods)

    for i in range(len(consensus_response.choices)):
        print(f"Choice {i}: {consensus_response.choices[i].message.content}")
except Exception as e:
    print(f"KLLMS consensus request failed: {e}")

# Test 3: Complex structured output with OpenAI parse
print("\n=== Test 3: Complex Structured Output (OpenAI) ===")
complex_prompt = """
Create a fictional tech company with the following details:
- Company name: "InnovaTech Solutions"
- Founded in 2018
- Headquarters in San Francisco, CA
- Has 2 departments: Engineering and Sales
- Engineering department has 3 employees with various programming skills
- Sales department has 2 employees with sales and marketing skills
- Each employee should have realistic contact info, addresses, and skill sets
- Include salary ranges appropriate for San Francisco tech scene
- Make some employees remote workers
- Set up manager-subordinate relationships
"""

try:
    parsed_result = openai_client.chat.completions.parse(model="gpt-4.1-nano", messages=[{"role": "user", "content": complex_prompt}], response_format=Company)
    print("Parsed result (OpenAI):")
    if parsed_result.choices[0].message.parsed:
        print(json.dumps(parsed_result.choices[0].message.parsed.model_dump(), indent=2))
    else:
        print("No parsed result available")
except Exception as e:
    print(f"OpenAI parse failed: {e}")

# Test 4: Complex structured output with KLLMS consensus
print("\n=== Test 4: Complex Structured Output (KLLMS Consensus) ===")
challenging_prompt = """
Create a fictional AI research company with these challenging requirements:
- Company name: "DeepMind Innovations" 
- Founded in 2020
- Headquarters in Seattle, WA
- Has 3 departments: Research, Engineering, and Business Development
- Research department: 4 employees with AI/ML expertise (PhD level)
- Engineering department: 5 employees with software engineering skills
- Business Development: 2 employees with business and legal skills
- Include edge cases: employees with 0.5 years experience, very high salaries, missing optional fields
- Some employees should have empty certification lists
- Include international addresses for remote workers
- Create a complex management hierarchy
- Use various skill levels from beginner to expert
"""

try:
    parsed_results = kllms_client.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": challenging_prompt}],
        response_format=Company,
        n=3,
        temperature=1.2,  # Higher temperature for more variation
    )
    print("Parsed results (KLLMS consensus):")
    if parsed_results.choices[0].message.parsed:
        print(json.dumps(parsed_results.choices[0].message.parsed.model_dump(), indent=2))
    else:
        print("No parsed result available")
    print(f"Consensus likelihoods: {parsed_results.likelihoods}")

    print("\n--- All Consensus Choices ---")
    for i, choice in enumerate(parsed_results.choices):
        print(f"\nChoice {i}:")
        try:
            if choice.message.parsed:
                choice_data = choice.message.parsed.model_dump()
                print(f"  Company: {choice_data['name']}")
                print(f"  Departments: {len(choice_data['departments'])}")
                print(f"  Total Employees: {choice_data['total_employees']}")
                print(f"  Annual Revenue: ${choice_data['annual_revenue']:,.2f}")
            else:
                print(f"  No parsed data available for choice {i}")
        except Exception as e:
            print(f"  Error parsing choice {i}: {e}")

except Exception as e:
    print(f"KLLMS consensus parse failed: {e}")

# Test 5: Failure-proofing with intentionally difficult prompt
print("\n=== Test 5: Failure-Proofing Test (Difficult Prompt) ===")
difficult_prompt = """
Create a company with these intentionally challenging constraints:
- Company name with special characters: "Ω-TechΣ & Co., Ltd."
- Founded in year 1995 (older company)
- Headquarters in Tokyo, Japan (non-US address)
- Employee with negative salary (bankruptcy scenario)
- Employee with 50+ years of experience in programming
- Skills that don't exist yet (future tech)
- Invalid email addresses and phone numbers
- Employees with conflicting manager relationships
- Department budget larger than company revenue
- Mix of valid and invalid data to test error handling
"""

try:
    difficult_results = kllms_client.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": difficult_prompt}],
        response_format=Company,
        n=5,  # More consensus for difficult case
        temperature=0.8,
    )
    print("Difficult prompt results:")
    if difficult_results.choices[0].message.parsed:
        print(json.dumps(difficult_results.choices[0].message.parsed.model_dump(), indent=2))
    else:
        print("No parsed result available")
    print(f"Consensus likelihoods: {difficult_results.likelihoods}")

    # Analyze consensus quality
    print("\n--- Consensus Analysis ---")
    for i, choice in enumerate(difficult_results.choices):
        try:
            if choice.message.parsed:
                choice_data = choice.message.parsed.model_dump()
                print(f"Choice {i}: {choice_data['name']} - {len(choice_data['departments'])} depts")
            else:
                print(f"Choice {i}: No parsed data available")
        except Exception as e:
            print(f"Choice {i}: Failed to parse - {e}")

except Exception as e:
    print(f"Difficult prompt test failed: {e}")

print("\n=== Test Complete ===")
print("Complex nested model testing finished. Check outputs for consensus quality and error handling.")

# Test 6: Error Handling and Edge Cases
print("\n=== Test 6: Error Handling and Edge Cases ===")

# Test invalid model
print("Testing invalid model...")
try:
    invalid_response = kllms_client.chat.completions.create(model="gpt-invalid-model", messages=[{"role": "user", "content": "Hello"}], n=2)
    print("Invalid model test passed (unexpected)")
except Exception as e:
    print(f"Invalid model test caught error (expected): {e}")

# Test empty messages
print("Testing empty messages...")
try:
    empty_response = kllms_client.chat.completions.create(model="gpt-4.1-nano", messages=[], n=2)
    print("Empty messages test passed (unexpected)")
except Exception as e:
    print(f"Empty messages test caught error (expected): {e}")

# Test extreme consensus values
print("Testing extreme consensus values...")
try:
    extreme_consensus = kllms_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Say hello"}],
        n=50,  # Very high consensus
    )
    print("Extreme consensus test passed")
    print(f"Consensus length: {len(extreme_consensus.choices)}")
except Exception as e:
    print(f"Extreme consensus test failed: {e}")

# Test 9: Simple Data Types with Consensus
print("\n=== Test 9: Simple Data Types with Consensus ===")


class SimpleResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    category: str


class NumberList(BaseModel):
    numbers: List[int]
    sum_total: int
    average: float


class BooleanDecision(BaseModel):
    decision: bool
    reasoning: str
    certainty_percentage: int = Field(ge=0, le=100)


# Test simple response format
print("Testing simple response format...")
try:
    simple_response = kllms_client.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Is Python good for machine learning? Provide your confidence and categorize the question."}],
        response_format=SimpleResponse,
        n=4,
    )
    print("Simple response consensus:")
    for i, choice in enumerate(simple_response.choices):
        if choice.message.parsed:
            data = choice.message.parsed.model_dump()
            print(f"  Choice {i}: {data['answer'][:50]}... (confidence: {data['confidence']})")
    print(f"Likelihoods: {simple_response.likelihoods}")
except Exception as e:
    print(f"Simple response test failed: {e}")

# Test number list format
print("Testing number list format...")
try:
    number_response = kllms_client.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Generate 5 random integers between 1-100, calculate their sum and average."}],
        response_format=NumberList,
        n=3,
    )
    print("Number list consensus:")
    for i, choice in enumerate(number_response.choices):
        if choice.message.parsed:
            data = choice.message.parsed.model_dump()
            print(f"  Choice {i}: numbers={data['numbers']}, sum={data['sum_total']}, avg={data['average']}")
    print(f"Likelihoods: {number_response.likelihoods}")
except Exception as e:
    print(f"Number list test failed: {e}")

# Test boolean decision format
print("Testing boolean decision format...")
try:
    bool_response = kllms_client.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "Should a startup prioritize growth over profitability in its first year?"}],
        response_format=BooleanDecision,
        n=5,
    )
    print("Boolean decision consensus:")
    for i, choice in enumerate(bool_response.choices):
        if choice.message.parsed:
            data = choice.message.parsed.model_dump()
            print(f"  Choice {i}: {data['decision']} ({data['certainty_percentage']}% certain)")
    print(f"Likelihoods: {bool_response.likelihoods}")
except Exception as e:
    print(f"Boolean decision test failed: {e}")

# Test 10: Conversation Context and Multi-turn
print("\n=== Test 10: Conversation Context and Multi-turn ===")

conversation_messages = [
    {"role": "user", "content": "I'm planning a trip to Japan. What should I know?"},
    {
        "role": "assistant",
        "content": "Japan is a fascinating destination! Here are key things to know: respect for customs, efficient public transport, tipping isn't customary, and many signs have English. What specific aspects interest you most?",
    },
    {"role": "user", "content": "Tell me about food etiquette and must-try dishes."},
]

try:
    conversation_response = kllms_client.chat.completions.create(model="gpt-4.1-nano", messages=conversation_messages, n=3, temperature=0.8)
    print("Multi-turn conversation response:")
    content = conversation_response.choices[0].message.content
    print(f"Response: {content or 'No content'}")
    if conversation_response.likelihoods:
        print(f"Consensus quality: {max(conversation_response.likelihoods):.3f}")
    else:
        print("Consensus quality: No likelihoods available")
except Exception as e:
    print(f"Conversation test failed: {e}")

# Test 11: Performance and Timing
print("\n=== Test 11: Performance and Timing ===")

import time


def time_request(description, request_func):
    print(f"Timing: {description}")
    start_time = time.time()
    try:
        result = request_func()
        end_time = time.time()
        print(f"  Success in {end_time - start_time:.2f} seconds")
        return result
    except Exception as e:
        end_time = time.time()
        print(f"  Failed in {end_time - start_time:.2f} seconds: {e}")
        return None


# Time single request
single_result = time_request(
    "Single OpenAI request", lambda: openai_client.chat.completions.create(model="gpt-4.1-nano", messages=[{"role": "user", "content": "Count from 1 to 10"}])
)

# Time consensus request
consensus_result = time_request(
    "KLLMS consensus request (n=3)", lambda: kllms_client.chat.completions.create(model="gpt-4.1-nano", messages=[{"role": "user", "content": "Count from 1 to 10"}], n=3)
)

# Time structured output
structured_result = time_request(
    "KLLMS structured consensus (n=3)",
    lambda: kllms_client.chat.completions.parse(model="gpt-4.1-nano", messages=[{"role": "user", "content": "Generate a simple employee record"}], response_format=Employee, n=3),
)

# Test 12: Edge Cases in Structured Data
print("\n=== Test 12: Edge Cases in Structured Data ===")


class EdgeCaseModel(BaseModel):
    optional_field: Optional[str] = None
    list_field: List[str] = []
    union_field: Union[str, int, float]
    enum_field: SkillLevel
    nested_optional: Optional[ContactInfo] = None


edge_prompt = """
Create data with edge cases:
- Leave optional_field as null
- Keep list_field empty
- Use a number for union_field
- Set enum_field to 'expert'
- Leave nested_optional as null
"""

try:
    edge_response = kllms_client.chat.completions.parse(model="gpt-4.1-nano", messages=[{"role": "user", "content": edge_prompt}], response_format=EdgeCaseModel, n=3)
    print("Edge case model consensus:")
    for i, choice in enumerate(edge_response.choices):
        if choice.message.parsed:
            data = choice.message.parsed.model_dump()
            print(f"  Choice {i}: {data}")
    print(f"Likelihoods: {edge_response.likelihoods}")
except Exception as e:
    print(f"Edge case test failed: {e}")

# Test 13: Consensus Quality Analysis
print("\n=== Test 13: Consensus Quality Analysis ===")


def analyze_consensus_quality(responses, description):
    print(f"Analyzing consensus quality for: {description}")
    if not responses or not hasattr(responses, "likelihoods"):
        print("  No valid responses to analyze")
        return

    likelihoods = responses.likelihoods
    print(f"  Number of responses: {len(likelihoods)}")
    print(f"  Mean likelihood: {sum(likelihoods) / len(likelihoods):.3f}")
    print(f"  Max likelihood: {max(likelihoods):.3f}")
    print(f"  Min likelihood: {min(likelihoods):.3f}")
    print(f"  Variance: {max(likelihoods) - min(likelihoods):.3f}")

    # Check if there's strong consensus (low variance)
    if max(likelihoods) - min(likelihoods) < 0.1:
        print("  Strong consensus detected!")
    elif max(likelihoods) - min(likelihoods) > 0.5:
        print("  High disagreement detected!")
    else:
        print("  Moderate consensus detected")


# Test different consensus scenarios
scenarios = [("Low temperature (should have high consensus)", 0.1), ("High temperature (should have low consensus)", 1.8), ("Medium temperature", 0.7)]

for desc, temp in scenarios:
    try:
        test_response = kllms_client.chat.completions.create(model="gpt-4.1-nano", messages=[{"role": "user", "content": "What is the capital of France?"}], n=5, temperature=temp)
        analyze_consensus_quality(test_response, desc)
    except Exception as e:
        print(f"Consensus analysis failed for {desc}: {e}")

# Test 14: Stress Testing
print("\n=== Test 14: Stress Testing ===")

print("Testing rapid consecutive requests...")
rapid_results = []
for i in range(5):
    try:
        result = kllms_client.chat.completions.create(model="gpt-4.1-nano", messages=[{"role": "user", "content": f"Quick response #{i + 1}"}], n=2, max_tokens=50)
        rapid_results.append(result)
        print(f"  Request {i + 1}: Success")
    except Exception as e:
        print(f"  Request {i + 1}: Failed - {e}")

print(f"Rapid test completed: {len(rapid_results)}/5 successful")

print("\n=== Extended Testing Complete ===")
print("All additional KLLMS SDK tests finished. Check outputs for comprehensive coverage.")
