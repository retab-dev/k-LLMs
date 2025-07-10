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


# Test 2: KLLMS consensus request
print("\n=== Test 2: KLLMS Consensus Request ===")
try:
    consensus_response = kllms_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": "What is the most efficient sorting algorithm for large datasets? Make the explanation very very short"}],
        n=3,
        temperature=1.0,
    )
    print("KLLMS consensus response:", consensus_response.choices[0].message.content)
    print("KLLMS consensus response likelihoods:", consensus_response.likelihoods)

    for i in range(len(consensus_response.choices)):
        print(f"Choice {i}: {consensus_response.choices[i].message.content}")
except Exception as e:
    print(f"KLLMS consensus request failed: {e}")
