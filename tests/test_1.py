# Example usage of KLLMS OpenAI Wrapper - Complex Nested Models Test

from k_llms import KLLMs
import dotenv

dotenv.load_dotenv(".env")

# Initialize clients
kllms_client = KLLMs()

# Test 1: KLLMS consensus request without response_format
print("\n=== Test 1: KLLMS Consensus Request ===")
consensus_response = kllms_client.chat.completions.create(
    model="gpt-4.1-nano", 
    messages=[{"role": "user", "content": "What is the most efficient sorting algorithm for large datasets?"}], 
    n=3, 
    temperature=1.0
)

print("KLLMS consensus response:", consensus_response.choices[0].message.content)

print("KLLMS consensus response likelihoods:", consensus_response.likelihoods)

for i in range(len(consensus_response.choices)):
    print(f"Choice {i}: {consensus_response.choices[i].message.content}")

print("Full output:", consensus_response.model_dump_json(indent=2))