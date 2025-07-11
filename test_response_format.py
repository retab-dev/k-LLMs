#!/usr/bin/env python3

import json
from pydantic import BaseModel
from k_llms import KLLMs


# Define test models
class UserInfo(BaseModel):
    name: str
    age: int
    city: str


class TaskInfo(BaseModel):
    title: str
    priority: int
    completed: bool
    description: str


def test_response_format_parsing():
    """Test that response format parsing works correctly with consensus."""

    print("=== Testing Response Format Parsing with Consensus ===")

    client = KLLMs()

    # Test 1: UserInfo model
    print("\n1. Testing UserInfo model...")
    try:
        response = client.chat.completions.parse(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Create user info for: Alice, 25 years old, lives in Paris"}], response_format=UserInfo, n=3, temperature=0.1
        )

        print(f"Total choices: {len(response.choices)}")
        print(f"Likelihoods: {response.likelihoods}")

        # Check consensus choice (index 0)
        consensus_choice = response.choices[0]
        print(f"Consensus choice parsed type: {type(consensus_choice.message.parsed)}")
        print(f"Consensus choice content: {consensus_choice.message.content}")

        if consensus_choice.message.parsed:
            parsed_user = consensus_choice.message.parsed
            print(f"Parsed consensus user: {parsed_user}")
            print(f"User name: {parsed_user.name}")
            print(f"User age: {parsed_user.age}")
            print(f"User city: {parsed_user.city}")
        else:
            print("ERROR: Consensus choice has no parsed content!")

        # Check individual choices
        for i, choice in enumerate(response.choices[1:], 1):
            print(f"Individual choice {i} parsed type: {type(choice.message.parsed)}")
            if choice.message.parsed:
                print(f"Individual choice {i}: {choice.message.parsed}")

    except Exception as e:
        print(f"Test 1 failed: {e}")

    # Test 2: TaskInfo model
    print("\n2. Testing TaskInfo model...")
    try:
        response = client.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Create a task: 'Review code', priority 2, not completed, description 'Review the new feature implementation'"}],
            response_format=TaskInfo,
            n=2,
            temperature=0.2,
        )

        print(f"Total choices: {len(response.choices)}")
        print(f"Likelihoods: {response.likelihoods}")

        # Check consensus choice
        consensus_choice = response.choices[0]
        print(f"Consensus choice parsed type: {type(consensus_choice.message.parsed)}")

        if consensus_choice.message.parsed:
            parsed_task = consensus_choice.message.parsed
            print(f"Parsed consensus task: {parsed_task}")
            print(f"Task title: {parsed_task.title}")
            print(f"Task priority: {parsed_task.priority}")
            print(f"Task completed: {parsed_task.completed}")
            print(f"Task description: {parsed_task.description}")
        else:
            print("ERROR: Consensus choice has no parsed content!")

    except Exception as e:
        print(f"Test 2 failed: {e}")

    # Test 3: Single request (should still work)
    print("\n3. Testing single request...")
    try:
        response = client.chat.completions.parse(
            model="gpt-4o-mini", messages=[{"role": "user", "content": "Create user info for: Bob, 30, New York"}], response_format=UserInfo, n=1
        )

        print(f"Total choices: {len(response.choices)}")
        print(f"Likelihoods: {response.likelihoods}")

        if response.choices[0].message.parsed:
            parsed_user = response.choices[0].message.parsed
            print(f"Single request parsed user: {parsed_user}")
        else:
            print("ERROR: Single choice has no parsed content!")

    except Exception as e:
        print(f"Test 3 failed: {e}")


if __name__ == "__main__":
    test_response_format_parsing()
    print("\n=== All tests completed ===")
