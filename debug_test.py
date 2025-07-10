#!/usr/bin/env python3
"""
Debug test for KLLMS consensus functionality
"""

import dotenv
import traceback

# Load environment variables
dotenv.load_dotenv()

print("🔍 Debug Test for KLLMS Consensus")
print("=" * 50)

try:
    print("1. Importing KLLMs...")
    from k_llms import KLLMs

    print("2. Creating client...")
    client = KLLMs()

    print("3. Making consensus request...")
    response = client.chat.completions.create(model="gpt-4.1-nano", messages=[{"role": "user", "content": "What is 2+2?"}], n=3, temperature=0.1)

    print("4. Checking response...")
    print(f"   Response type: {type(response)}")
    print(f"   Has choices: {hasattr(response, 'choices')}")

    if hasattr(response, "choices"):
        print(f"   Number of choices: {len(response.choices)}")

    print(f"   Has likelihoods: {hasattr(response, 'likelihoods')}")

    if hasattr(response, "likelihoods"):
        print(f"   Likelihoods: {response.likelihoods}")
        print(f"   Likelihoods type: {type(response.likelihoods)}")

    if response and response.choices and len(response.choices) == 3:
        print("✅ Consensus test passed!")
        print(f"Consensus responses: {len(response.choices)}")
        print(f"Likelihoods: {response.likelihoods}")
    else:
        print("❌ Consensus test failed - unexpected response structure")

except Exception as e:
    print(f"❌ Error occurred: {e}")
    print("Full traceback:")
    traceback.print_exc()
