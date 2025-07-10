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
    print(f"Total choices: {len(extreme_consensus.choices)} (1 consensus + 50 individual)")
    print(f"Consensus choice content: {extreme_consensus.choices[0].message.content}")
    print(f"Individual choices: {len(extreme_consensus.choices) - 1}")
except Exception as e:
    print(f"Extreme consensus test failed: {e}")
