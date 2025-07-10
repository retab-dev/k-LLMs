"""
Example usage of the k_llms library with the Responses API functionality.

This example demonstrates how to use the k_llms library with OpenAI's Responses API
to make requests with consensus functionality.

Note: The OpenAI Responses API is different from Chat Completions and requires
specific setup and model access.
"""

import asyncio
import dotenv
from k_llms import KLLMs, AsyncKLLMs

# Load environment variables
dotenv.load_dotenv(".env")


def sync_example():
    """Example using the synchronous KLLMs client with Responses API."""
    print("Initializing KLLMs client...")
    client = KLLMs()
    
    try:
        # Single request using responses
        print("\nMaking single response request...")
        single_response = client.responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": "What are the benefits of renewable energy?"}],
            n=1
        )
        print(f"Single response type: {type(single_response)}")
        print(f"Response output: {single_response.output}")
        
    except Exception as e:
        print(f"Single response request failed: {e}")
        print("Note: Responses API requires special access and setup")
    
    try:
        # Multiple parallel requests with consensus
        print("\nMaking multiple response requests with consensus...")
        consensus_response = client.responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": "What are the benefits of renewable energy?"}],
            n=3
        )
        print(f"Consensus response type: {type(consensus_response)}")
        print(f"Number of individual responses: {len(consensus_response.individual_responses) if consensus_response.individual_responses else 0}")
        print(f"Consolidated output: {consensus_response.output}")
        print(f"Likelihoods: {consensus_response.likelihoods}")
        
    except Exception as e:
        print(f"Consensus response request failed: {e}")
        print("Note: Responses API requires special access and setup")

    try:
        # Example with parse method for structured output
        print("\nUsing parse method with responses...")
        parsed_response = client.responses.parse(
            model="gpt-4o",
            input=[{"role": "user", "content": "Generate a summary of renewable energy benefits"}],
            n=2
        )
        print(f"Parsed response type: {type(parsed_response)}")
        print(f"Parsed output: {parsed_response.output}")
        
    except Exception as e:
        print(f"Parse response request failed: {e}")
        print("Note: Responses API requires special access and setup")


async def async_example():
    """Example using the asynchronous AsyncKLLMs client with Responses API."""
    print("Initializing AsyncKLLMs client...")
    client = AsyncKLLMs()
    
    try:
        # Single async request
        print("\nMaking async single response request...")
        single_response = await client.responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
            n=1
        )
        print(f"Async single response type: {type(single_response)}")
        print(f"Response output: {single_response.output}")
        
    except Exception as e:
        print(f"Async single response request failed: {e}")
        print("Note: Responses API requires special access and setup")

    try:
        # Multiple parallel async requests with consensus
        print("\nMaking parallel async response requests with consensus...")
        consensus_response = await client.responses.create(
            model="gpt-4o",
            input=[{"role": "user", "content": "Explain quantum computing in simple terms"}],
            n=3
        )
        print(f"Async consensus response type: {type(consensus_response)}")
        print(f"Number of individual responses: {len(consensus_response.individual_responses) if consensus_response.individual_responses else 0}")
        print(f"Consolidated output: {consensus_response.output}")
        print(f"Likelihoods: {consensus_response.likelihoods}")
        
    except Exception as e:
        print(f"Async consensus response request failed: {e}")
        print("Note: Responses API requires special access and setup")


def chat_completions_example():
    """Example using Chat Completions (which is more commonly available)."""
    print("Initializing KLLMs client for Chat Completions...")
    client = KLLMs()
    
    try:
        # Chat completions with consensus - this should work with standard OpenAI access
        print("\nMaking chat completions request with consensus...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using a more commonly available model
            messages=[{"role": "user", "content": "What are 3 benefits of renewable energy?"}],
            n=3,
            max_tokens=150
        )
        
        print(f"Response type: {type(response)}")
        print(f"Number of choices: {len(response.choices)}")
        print(f"Consolidated response: {response.choices[0].message.content}")
        print(f"Likelihoods: {response.likelihoods}")
        
        # Show individual responses
        print("\nIndividual responses:")
        for i, choice in enumerate(response.choices[1:], 1):
            print(f"  Response {i}: {choice.message.content[:100]}...")
            
    except Exception as e:
        print(f"Chat completions request failed: {e}")


if __name__ == "__main__":
    print("k_llms Responses API Example")
    print("=" * 50)
    
    # First try Chat Completions (more likely to work)
    print("CHAT COMPLETIONS EXAMPLE (Recommended):")
    try:
        chat_completions_example()
    except Exception as e:
        print(f"Chat completions example error: {e}")
        print("Please ensure OPENAI_API_KEY is set in your environment")

    print("\n" + "=" * 50)

    # Try Responses API (requires special access)
    print("RESPONSES API EXAMPLE (Requires Special Access):")
    print("Note: The Responses API requires special access from OpenAI")
    
    # Run synchronous example
    print("\nSYNCHRONOUS RESPONSES EXAMPLE:")
    try:
        sync_example()
    except Exception as e:
        print(f"Sync responses example error: {e}")

    print("\n" + "-" * 30)

    # Run asynchronous example
    print("ASYNCHRONOUS RESPONSES EXAMPLE:")
    try:
        asyncio.run(async_example())
    except Exception as e:
        print(f"Async responses example error: {e}")

    print("\nExample completed!")
    print("\nNOTE: If you're getting errors with the Responses API, try using")
    print("the Chat Completions API instead, which is more widely available.")