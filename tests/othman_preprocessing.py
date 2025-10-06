from retab import Retab
import dotenv
from openai import OpenAI
from typing import Any
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
    JSONSchema,
)

from retab.types.schemas import Schema

dotenv.load_dotenv(".env.local")

schema: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number"},
        "city": {"type": "string"},
    },
    "additionalProperties": False,
}

schema_obj = Schema(
    json_schema=schema,
)


json_schema: JSONSchema = {
    "name": "UserData",
    "schema": schema_obj.inference_json_schema,
    "strict": True,
}

response_format: ResponseFormatJSONSchema = {
    "type": "json_schema",
    "json_schema": json_schema,
}

print(schema_obj.inference_json_schema)

retab_client = Retab()

messages_obj = retab_client.documents.create_messages(
    document="data/document.pdf"
)

print(messages_obj.messages)

openai_client = OpenAI()

response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages_obj.messages, # type: ignore
    response_format=response_format,
)

print(response.choices[0].message.content)

