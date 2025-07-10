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
