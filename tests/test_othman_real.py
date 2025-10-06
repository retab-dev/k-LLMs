# Example usage of KLLMS OpenAI Wrapper - Complex Nested Models Test

from k_llms import KLLMs
from openai.types.shared_params.response_format_json_schema import JSONSchema, ResponseFormatJSONSchema
import dotenv
import json
from retab import Retab
from retab.types.schemas import Schema
from retab.utils.json_schema import filter_auxiliary_fields
from typing import Literal

dotenv.load_dotenv(".env.local")


folder_name: Literal["lellikelly", "oilandgas", "coppermines"] = "oilandgas"

document_path = f"data/{folder_name}/document.pdf"
json_schema_path = f"data/{folder_name}/schema.json"

# Initialize clients
kllms_client = KLLMs()


# Test 3: Complex structured output
print("\n=== Test 3: Complex Structured Output ===")
system_prompt = """
You are a helpful assistant that can extract information from a document.
You will be given a document and you will need to extract the information from the document.
You will need to extract the information from the document and return it in a JSON format.
"""

retab_client = Retab()

messages_obj = retab_client.documents.create_messages(
    document=document_path
)

#print(messages_obj.messages)

schema_dict = json.loads(open(json_schema_path).read())

#print("Schema dict: ", schema_dict)

schema_obj = Schema(
    json_schema=schema_dict,
)

response_format = ResponseFormatJSONSchema(
    type="json_schema",
    json_schema=JSONSchema(name="schema", schema=schema_obj.inference_json_schema, strict=True),
)

try:
    
    parsed_result = kllms_client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {
                "role": "developer", "content": system_prompt,
            },
            
        ]+ messages_obj.messages, # type: ignore
        response_format=response_format,
        n=4,
        reasoning_effort="minimal"
    )
        
    print("Result:")
    if parsed_result.choices[0].message.content:
        #print(parsed_result.choices[0].message.content)
        print("\n\nKLLMS consensus response likelihoods:", json.dumps(parsed_result.likelihoods, indent=2))

    for i in range(len(parsed_result.choices)):
        content = parsed_result.choices[i].message.content or "{}"
        # Filter out reasoning fields if this is what we want
        json_content = filter_auxiliary_fields(json.loads(content))
        print(f"Choice {i}: {json.dumps(json_content, indent=2)}")

    if not parsed_result.choices[0].message.content:
        print("No result available")
except Exception as e:
    print(f"OpenAI parse failed: {e}")
