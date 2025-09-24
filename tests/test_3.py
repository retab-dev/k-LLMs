# Example usage of KLLMS OpenAI Wrapper - Complex Nested Models Test

from k_llms import KLLMs
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union
from datetime import datetime
from enum import Enum
import dotenv
import json

dotenv.load_dotenv(".env")

# Initialize clients
kllms_client = KLLMs()


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


# Test 3: Complex structured output with Retab parse
print("\n=== Test 3: Complex Structured Output (Retab) ===")
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
    parsed_result = kllms_client.chat.completions.parse(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": complex_prompt}], 
        response_format=Company, 
        n=2
        )
        
    print("Parsed result (Retab):")
    if parsed_result.choices[0].message.parsed:
        print(json.dumps(parsed_result.choices[0].message.parsed.model_dump(), indent=2))
        print("\n\nKLLMS consensus response likelihoods:", parsed_result.likelihoods)

    for i in range(len(parsed_result.choices)):
        assert parsed_result.choices[i].message.parsed
        print(f"Choice {i}: {parsed_result.choices[i].message.content}")

    if not parsed_result.choices[0].message.parsed:
        print("No parsed result available")
except Exception as e:
    print(f"OpenAI parse failed: {e}")
