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
