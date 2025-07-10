"""
Example usage of the OpenAI Responses API wrapper with parallel requests and consolidation.

This example demonstrates how to use the Responses and AsyncResponses classes
to make parallel requests to the OpenAI Responses API and consolidate the results.
"""

import asyncio
from openai import OpenAI, AsyncOpenAI
from k_llms.resources.responses import Responses, AsyncResponses


def sync_example():
    """Example using the synchronous Responses wrapper."""
    # Initialize OpenAI client
    client = OpenAI()

    # Create responses wrapper
    responses_wrapper = Responses(client)

    # Single request (n_consensus=1)
    print("Making single request...")
    single_response = responses_wrapper.create(model="gpt-4o", input="What are the benefits of renewable energy?", n_consensus=1)
    print(f"Single response type: {type(single_response)}")

    # Multiple parallel requests (n_consensus=3)
    print("\nMaking parallel requests...")
    multiple_responses = responses_wrapper.create(model="gpt-4o", input="What are the benefits of renewable energy?", n_consensus=3)
    print(f"Got {len(multiple_responses)} responses")

    # Consolidate the responses
    print("\nConsolidating responses...")
    consolidated = responses_wrapper.consolidate_responses(multiple_responses, consolidation_method="enhanced_consensus")

    print(f"Consolidated: {consolidated['consolidated']}")
    print(f"Original count: {consolidated['original_count']}")
    print(f"Consolidated content preview: {str(consolidated['consolidated_content'])[:200]}...")

    # Example with parse method for structured output
    print("\n" + "=" * 50)
    print("Using parse method...")

    parsed_responses = responses_wrapper.parse(
        model="gpt-4o",
        input="Generate a summary of renewable energy benefits",
        n_consensus=2,
        # Add any response format parameters here
    )

    if len(parsed_responses) > 1:
        parsed_consolidated = responses_wrapper.consolidate_responses(parsed_responses, consolidation_method="enhanced_consensus")
        print(f"Parsed and consolidated {len(parsed_responses)} responses")


async def async_example():
    """Example using the asynchronous AsyncResponses wrapper."""
    # Initialize async OpenAI client
    client = AsyncOpenAI()

    # Create async responses wrapper
    responses_wrapper = AsyncResponses(client)

    # Single async request
    print("Making async single request...")
    single_response = await responses_wrapper.create(model="gpt-4o", input="Explain quantum computing in simple terms", n_consensus=1)
    print(f"Async single response type: {type(single_response)}")

    # Multiple parallel async requests
    print("\nMaking parallel async requests...")
    multiple_responses = await responses_wrapper.create(model="gpt-4o", input="Explain quantum computing in simple terms", n_consensus=3)
    print(f"Got {len(multiple_responses)} async responses")

    # Consolidate the async responses
    print("\nConsolidating async responses...")
    consolidated = await responses_wrapper.consolidate_responses(multiple_responses, consolidation_method="enhanced_consensus")

    print(f"Async consolidated: {consolidated['consolidated']}")
    print(f"Original count: {consolidated['original_count']}")
    print(f"Consolidated content preview: {str(consolidated['consolidated_content'])[:200]}...")


if __name__ == "__main__":
    print("OpenAI Responses API Wrapper Example")
    print("=" * 50)

    # Run synchronous example
    print("SYNCHRONOUS EXAMPLE:")
    try:
        sync_example()
    except Exception as e:
        print(f"Sync example error: {e}")

    print("\n" + "=" * 50)

    # Run asynchronous example
    print("ASYNCHRONOUS EXAMPLE:")
    try:
        asyncio.run(async_example())
    except Exception as e:
        print(f"Async example error: {e}")

    print("\nExample completed!")
