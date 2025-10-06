# Example usage of KLLMS OpenAI Wrapper - Complex Nested Models Test

from k_llms import KLLMs
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
import dotenv
import json

dotenv.load_dotenv(".env.local")

# Initialize clients
kllms_client = KLLMs()


# Test 3: Complex structured output
print("\n=== Test 3: Complex Structured Output ===")
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

json_schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "city": {"type": "string"}
    },
    "required": ["name", "age", "city"],
    "additionalProperties": False,
}

try:
    response_format = ResponseFormatJSONSchema(
        type="json_schema",
        json_schema=JSONSchema(name="schema", schema=json_schema, strict=True),
    )
    parsed_result = kllms_client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[{"role": "user", "content": complex_prompt}], 
        response_format=response_format,
        n=2
        )
        
    print("Result:")
    if parsed_result.choices[0].message.content:
        print(parsed_result.choices[0].message.content)
        print("\n\nKLLMS consensus response likelihoods:", parsed_result.likelihoods)

    for i in range(len(parsed_result.choices)):
        content = parsed_result.choices[i].message.content or "{}"
        print(f"Choice {i}: {json.dumps(json.loads(content), indent=2)}")

    if not parsed_result.choices[0].message.content:
        print("No result available")
except Exception as e:
    print(f"OpenAI parse failed: {e}")
