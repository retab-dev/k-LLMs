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
