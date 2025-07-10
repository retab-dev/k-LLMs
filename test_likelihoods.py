# Test script to demonstrate likelihood functionality

from k_llms import KLLMs
from pydantic import BaseModel
import dotenv

dotenv.load_dotenv(".env")

# Initialize client
kllms_client = KLLMs()

# Test with a simple question that might have varying responses
print("=== Testing Simple Consensus with Likelihoods ===")
consensus_response = kllms_client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "What is the capital of France?"}], 
    n_consensus=3
)

print("Consolidated response:", consensus_response.choices[0].message.content)
print("Original responses:", [choice.message.content for choice in consensus_response.choices[1:]])
print("Likelihoods:", consensus_response.likelihoods)
print()

# Test with structured output
print("=== Testing Structured Output with Parse ===")

class PersonInfo(BaseModel):
    name: str
    age: int
    city: str

# Test with parse consensus
parsed_response = kllms_client.chat.completions.parse(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "Create a person: Alice who is 25 years old and lives in Paris"}], 
    response_format=PersonInfo, 
    n_consensus=3
)

print("Consolidated parsed result:", parsed_response.choices[0].message.parsed)
print("Original parsed results:", [choice.message.parsed for choice in parsed_response.choices[1:]])
print("Total choices:", len(parsed_response.choices))
print("Likelihoods:", parsed_response.likelihoods)
print()

# Test with a single request (should show 100% confidence)
print("=== Testing Single Request (Should show 100% confidence) ===")
single_response = kllms_client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "What is 1+1?"}], 
    n_consensus=1
)

print("Single response:", single_response.choices[0].message.content)
print("Total choices:", len(single_response.choices))
print("Likelihoods:", single_response.likelihoods)
print()

print("=== All tests completed successfully! ===")