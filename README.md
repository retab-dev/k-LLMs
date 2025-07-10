# k-LLMS

Built with 🩷 at [retab](https://retab.com)

k-llms is a wrapper around the OpenAI client that adds consensus functionality through the `n` parameter.

## Features

- Drop-in replacement for OpenAI client
- Uses the `n` parameter to generate multiple completions efficiently
- Automatic result consolidation using majority voting
- Likelihood computations
- Support for both sync and async operations
- Compatible with all OpenAI chat completion parameters
- Support for structured outputs with `parse()`

## Installation

```python
# The wrapper uses the official OpenAI client
pip install openai
pip install k-llms
```

## Usage

### Basic Usage

```python
from k_llms import KLLMs
from openai import OpenAI

# Initialize the client (uses OPENAI_API_KEY env var by default)
kllms_client = KLLMs()

openai_client = OpenAI()

# Make a single request (normal OpenAI behavior)
response = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Make multiple requests with consensus
consensus_response = kllms_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    n=3  # Generates 3 completions and consolidates
)
```

### Structured Outputs with Parse

```python
from pydantic import BaseModel

class UserInfo(BaseModel):
    name: str
    age: int

# Single parse request
result = openai_client.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "John is 30 years old"}],
    response_format=UserInfo
)

# Multiple parse requests with consensus
result = kllms_client.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "John is 30 years old"}],
    response_format=UserInfo,
    n=3
)

# Access consolidated result
consensus_user = result.choices[0].message.parsed  # Consolidated UserInfo object
original_users = [choice.message.parsed for choice in result.choices[1:]]  # Original results
```

### Async Usage

```python
from k_llms import AsyncKLLMs
from openai import AsyncOpenAI
import asyncio

async def main():
    kllms_client = AsyncKLLMs()
    openai_client = AsyncOpenAI()
    
    response = await kllms_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
        n=3
    )
    print(response.choices[0].message.content)

asyncio.run(main())
```

## How Consensus Works

When `n > 1`:

1. For chat completions: Uses OpenAI's native `n` parameter to generate multiple completions in a single API call
2. For responses API: Makes parallel requests (as the Responses API doesn't support the `n` parameter)
3. For both `completions.create()` and `parse()`: Results are consolidated using majority voting
   - For simple values: Most common value wins
   - For JSON/dict responses: Field-by-field majority voting
   - For lists: Element-by-element consolidation
4. All responses return a choices array where:
   - `choices[0]`: Consolidated/consensus result
   - `choices[1...n]`: Individual original results from each API call

## API Compatibility

The wrapper maintains full compatibility with the OpenAI client API. All parameters supported by the official client work seamlessly, including:

- `temperature`, `top_p`, `max_tokens`
- `response_format`, `tools`, `tool_choice`
- `stream` (automatically disabled - all responses are non-streaming)
- All other OpenAI parameters

## Limitations

- Streaming is not supported (all requests return `KLLMsChatCompletion` objects)