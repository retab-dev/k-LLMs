from typing import Any, List, Optional, Union
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import ChatCompletion, ParsedChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.parsed_chat_completion import ParsedChoice
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage
from openai.lib._parsing import ResponseFormatT
import json
from ..types.completions import KLLMsChatCompletion
from ..types.parsed import KLLMsParsedChatCompletion
from .consensus_utils import (
    consensus_values,
    ConsensusSettings,
    SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    async_consensus_values,
    recursive_list_alignments,
    async_recursive_list_alignments,
)
from pydantic import BaseModel


def _safe_parse_content(content: str) -> dict[str, Any]:
    """
    Safely parse content as JSON. If parsing fails, wrap the content in {"text": content}.

    Args:
        content: The content string to parse

    Returns:
        Parsed JSON object or {"text": content} if parsing fails
    """
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return {"text": content}


def _format_consensus_content(consensus_content: dict[str, Any]) -> str:
    """
    Format consensus content for the final message. If it's a simple text wrapper,
    return just the text. Otherwise, return JSON string.

    Args:
        consensus_content: The consensus content dictionary

    Returns:
        Formatted content string
    """
    if consensus_content is None:
        return ""

    # Check if this is a simple text wrapper from free-form input
    if isinstance(consensus_content, dict) and len(consensus_content) == 1 and "text" in consensus_content and isinstance(consensus_content["text"], str):
        return consensus_content["text"]

    # Otherwise, return as JSON string
    return json.dumps(consensus_content)


def consolidate_chat_completions(
    completions: Union[List[ChatCompletion], ChatCompletion],
    get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    client: OpenAI,
    consensus_settings: ConsensusSettings = ConsensusSettings(),
) -> KLLMsChatCompletion:
    """
    Consolidate multiple ChatCompletion objects or a single ChatCompletion with multiple choices into a single KLLMsChatCompletion with consensus.

    Args:
        completions: List of ChatCompletion objects or single ChatCompletion with multiple choices to consolidate
        get_openai_embeddings_from_text: Function to get embeddings for text similarity
        consensus_settings: Settings for consensus algorithm. If None, uses default settings.

    Returns:
        KLLMsChatCompletion with consolidated response and likelihood scores
    """
    # Handle single completion case
    if isinstance(completions, ChatCompletion):
        completion = completions
        assert len(completion.choices) > 0, "Cannot consolidate empty list of choices"

        if len(completion.choices) == 1:
            # Single choice - just wrap it
            return KLLMsChatCompletion.model_validate(completion.model_dump())

        # Multiple choices - get consensus content
        choice_contents: list[dict[str, Any]] = []
        for choice in completion.choices:
            if choice.message.content:
                choice_contents.append(_safe_parse_content(choice.message.content))

        # Align lists/objects across choices once before consensus
        if len(choice_contents) >= 2:
            aligned_seq, _ = recursive_list_alignments(
                choice_contents,
                consensus_settings.string_similarity_method,
                get_openai_embeddings_from_text,
                client,
                consensus_settings.min_support_ratio,
            )
            choice_contents = [(d if isinstance(d, dict) else {}) for d in aligned_seq]

        consensus_content, likelihoods = consensus_values(
            choice_contents,
            consensus_settings,
            get_openai_embeddings_from_text,
            client=client,
        )

        # Create consolidated message
        # Convert consensus content to JSON string for message content
        content_str = _format_consensus_content(consensus_content)
        consolidated_message = ChatCompletionMessage(
            role="assistant",
            content=content_str,
            function_call=completion.choices[0].message.function_call if completion.choices else None,
            tool_calls=completion.choices[0].message.tool_calls if completion.choices else None,
            refusal=completion.choices[0].message.refusal if completion.choices else None,
        )

        # Create consolidated choice (consensus result)
        consolidated_choice = Choice(
            finish_reason=completion.choices[0].finish_reason if completion.choices else "stop",
            index=0,
            message=consolidated_message,
            logprobs=completion.choices[0].logprobs if completion.choices else None,
        )

        # Keep original individual choices with updated indices
        individual_choices = []
        for i, choice in enumerate(completion.choices):
            individual_choice = Choice(finish_reason=choice.finish_reason, index=i + 1, message=choice.message, logprobs=choice.logprobs)
            individual_choices.append(individual_choice)

        # Combine consensus choice with individual choices
        all_choices = [consolidated_choice] + individual_choices

        # Use original completion usage
        consolidated_usage = completion.usage

        return KLLMsChatCompletion.model_validate({**completion.model_dump(), "choices": all_choices, "likelihoods": likelihoods, "usage": consolidated_usage})

    # Handle list of completions case
    else:
        completion_list = completions
        assert len(completion_list) > 0, "Cannot consolidate empty list of completions"

        if len(completion_list) == 1:
            # Single completion - just wrap it
            return KLLMsChatCompletion.model_validate(completion_list[0].model_dump())

        # Multiple completions - get consensus content
        completion_contents: list[dict[str, Any]] = []
        for completion in completion_list:
            if completion.choices and completion.choices[0].message.content:
                completion_contents.append(_safe_parse_content(completion.choices[0].message.content))

        # Align lists/objects across completions once before consensus
        if len(completion_contents) >= 2:
            aligned_seq, _ = recursive_list_alignments(
                completion_contents,
                consensus_settings.string_similarity_method,
                get_openai_embeddings_from_text,
                client,
                consensus_settings.min_support_ratio,
            )
            completion_contents = [(d if isinstance(d, dict) else {}) for d in aligned_seq]

        consensus_content, likelihoods = consensus_values(
            completion_contents,
            consensus_settings,
            get_openai_embeddings_from_text,
            client=client,
        )

        # Use the first completion as the base
        base_completion = completion_list[0]

        # Create consolidated message
        # Convert consensus content to JSON string for message content
        content_str = _format_consensus_content(consensus_content)
        consolidated_message = ChatCompletionMessage(
            role="assistant",
            content=content_str,
            function_call=base_completion.choices[0].message.function_call if base_completion.choices else None,
            tool_calls=base_completion.choices[0].message.tool_calls if base_completion.choices else None,
            refusal=base_completion.choices[0].message.refusal if base_completion.choices else None,
        )

        # Create consolidated choice (consensus result)
        consolidated_choice = Choice(
            finish_reason=base_completion.choices[0].finish_reason if base_completion.choices else "stop",
            index=0,
            message=consolidated_message,
            logprobs=base_completion.choices[0].logprobs if base_completion.choices else None,
        )

        # Keep original individual choices with updated indices
        individual_choices = []
        for i, completion in enumerate(completion_list):
            if completion.choices:
                individual_choice = Choice(
                    finish_reason=completion.choices[0].finish_reason, index=i + 1, message=completion.choices[0].message, logprobs=completion.choices[0].logprobs
                )
                individual_choices.append(individual_choice)

        # Combine consensus choice with individual choices
        all_choices = [consolidated_choice] + individual_choices

        # Use base completion usage
        consolidated_usage = base_completion.usage

        return KLLMsChatCompletion.model_validate({**base_completion.model_dump(), "choices": all_choices, "likelihoods": likelihoods, "usage": consolidated_usage})


async def async_consolidate_chat_completions(
    completion: ChatCompletion,
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    client: AsyncOpenAI,
    consensus_settings: ConsensusSettings = ConsensusSettings(),
) -> KLLMsChatCompletion:
    """
    Async version of consolidate_chat_completions.
    Consolidate multiple choices in a ChatCompletion object into a single KLLMsChatCompletion with consensus.

    Args:
        completions: ChatCompletion object with multiple choices to consolidate
        consensus_settings: Settings for consensus algorithm. If None, uses default settings.
        async_get_openai_embeddings_from_text: Async function to get embeddings for text similarity

    Returns:
        KLLMsChatCompletion with consolidated response and likelihood scores
    """

    assert len(completion.choices) > 0, "Cannot consolidate empty list of choices"

    if len(completion.choices) == 1:
        # Single choice - just wrap it
        return KLLMsChatCompletion.model_validate(completion.model_dump())

    # Check if we have multiple choices (from n parameter)
    else:  # len(completion.choices) > 1:
        # Get consensus content

        # Extract all choice contents for consensus
        async_choice_contents: list[dict[str, Any]] = []
        for choice in completion.choices:
            if choice.message.content:
                async_choice_contents.append(_safe_parse_content(choice.message.content))

        # Align lists/objects across choices once before consensus
        if len(async_choice_contents) >= 2:
            aligned_seq, _ = await async_recursive_list_alignments(
                async_choice_contents,
                consensus_settings.string_similarity_method,
                async_get_openai_embeddings_from_text,
                client,
                consensus_settings.min_support_ratio,
            )
            async_choice_contents = [(d if isinstance(d, dict) else {}) for d in aligned_seq]

        consensus_content, likelihoods = await async_consensus_values(
            async_choice_contents,
            consensus_settings,
            async_get_openai_embeddings_from_text,
            client=client,
        )

        # Create consolidated message
        # Convert consensus content to JSON string for message content
        content_str = _format_consensus_content(consensus_content)
        consolidated_message = ChatCompletionMessage(
            role="assistant",
            content=content_str,
            function_call=completion.choices[0].message.function_call if completion.choices else None,
            tool_calls=completion.choices[0].message.tool_calls if completion.choices else None,
            refusal=completion.choices[0].message.refusal if completion.choices else None,
        )

        # Create consolidated choice (consensus result)
        consolidated_choice = Choice(
            finish_reason=completion.choices[0].finish_reason if completion.choices else "stop",
            index=0,
            message=consolidated_message,
            logprobs=completion.choices[0].logprobs if completion.choices else None,
        )

        # Keep original individual choices with updated indices
        individual_choices = []
        for i, choice in enumerate(completion.choices):
            individual_choice = Choice(finish_reason=choice.finish_reason, index=i + 1, message=choice.message, logprobs=choice.logprobs)
            individual_choices.append(individual_choice)

        # Combine consensus choice with individual choices
        all_choices = [consolidated_choice] + individual_choices

        # Use original completion usage
        consolidated_usage = completion.usage

        return KLLMsChatCompletion.model_validate({**completion.model_dump(), "choices": all_choices, "likelihoods": likelihoods, "usage": consolidated_usage})


def consolidate_parsed_chat_completions(
    completion: ParsedChatCompletion,
    get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    client: OpenAI,
    consensus_settings: ConsensusSettings = ConsensusSettings(),
    response_format: Optional[type[ResponseFormatT]] = None,
) -> KLLMsParsedChatCompletion:
    """
    Consolidate multiple choices in a ParsedChatCompletion object into a single KLLMsParsedChatCompletion with consensus.

    Args:
        completion: ParsedChatCompletion object with multiple choices to consolidate
        get_openai_embeddings_from_text: Function to get embeddings for text similarity
        consensus_settings: Settings for consensus algorithm. If None, uses default settings.
        response_format: The response format type for parsing the consensus content

    Returns:
        KLLMsParsedChatCompletion with consolidated response and likelihood scores
    """

    assert len(completion.choices) > 0, "Cannot consolidate empty list of choices"

    if len(completion.choices) == 1:
        # Single choice - just wrap it
        return KLLMsParsedChatCompletion.model_validate(completion.model_dump())

    # Multiple choices - get consensus content
    parsed_choice_contents: list[dict[str, Any]] = []
    for choice in completion.choices:
        if choice.message.content:
            parsed_choice_contents.append(_safe_parse_content(choice.message.content))

    # Align lists/objects across choices once before consensus
    if len(parsed_choice_contents) >= 2:
        aligned_seq, _ = recursive_list_alignments(
            parsed_choice_contents,
            consensus_settings.string_similarity_method,
            get_openai_embeddings_from_text,
            client,
            consensus_settings.min_support_ratio,
        )
        parsed_choice_contents = [(d if isinstance(d, dict) else {}) for d in aligned_seq]

    consensus_content, likelihoods = consensus_values(
        parsed_choice_contents,
        consensus_settings,
        get_openai_embeddings_from_text,
        client=client,
    )

    # Parse the consensus content if response_format is a BaseModel
    parsed_consensus = None
    if response_format and consensus_content is not None:
        try:
            # Check if the response_format is a BaseModel subclass
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                parsed_consensus = response_format.model_validate(consensus_content)
        except Exception:
            # If parsing fails, keep parsed as None
            parsed_consensus = None

    # Create consolidated message
    # For parsed completions, content should be a JSON string
    content_str = _format_consensus_content(consensus_content)
    consolidated_message = ParsedChatCompletionMessage(
        role="assistant",
        content=content_str,
        function_call=completion.choices[0].message.function_call if completion.choices else None,
        tool_calls=completion.choices[0].message.tool_calls if completion.choices else None,
        refusal=completion.choices[0].message.refusal if completion.choices else None,
        parsed=parsed_consensus,
    )

    # Create consolidated choice (consensus result)
    consolidated_choice = ParsedChoice(
        finish_reason=completion.choices[0].finish_reason if completion.choices else "stop",
        index=0,
        message=consolidated_message,
        logprobs=completion.choices[0].logprobs if completion.choices else None,
    )

    # Keep original individual choices with updated indices
    individual_choices = []
    for i, choice in enumerate(completion.choices):
        individual_choice = ParsedChoice(finish_reason=choice.finish_reason, index=i + 1, message=choice.message, logprobs=choice.logprobs)
        individual_choices.append(individual_choice)

    # Combine consensus choice with individual choices
    all_choices = [consolidated_choice] + individual_choices

    # Use original completion usage
    consolidated_usage = completion.usage

    return KLLMsParsedChatCompletion.model_validate({**completion.model_dump(), "choices": all_choices, "likelihoods": likelihoods, "usage": consolidated_usage})


async def async_consolidate_parsed_chat_completions(
    completion: ParsedChatCompletion,
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    client: AsyncOpenAI,
    consensus_settings: ConsensusSettings = ConsensusSettings(),
    response_format: Optional[type[ResponseFormatT]] = None,
) -> KLLMsParsedChatCompletion:
    """
    Async version of consolidate_parsed_chat_completions.
    Consolidate multiple choices in a ParsedChatCompletion object into a single KLLMsParsedChatCompletion with consensus.

    Args:
        completion: ParsedChatCompletion object with multiple choices to consolidate
        async_get_openai_embeddings_from_text: Async function to get embeddings for text similarity
        consensus_settings: Settings for consensus algorithm. If None, uses default settings.
        response_format: The response format type for parsing the consensus content

    Returns:
        KLLMsParsedChatCompletion with consolidated response and likelihood scores
    """

    assert len(completion.choices) > 0, "Cannot consolidate empty list of choices"

    if len(completion.choices) == 1:
        # Single choice - just wrap it
        return KLLMsParsedChatCompletion.model_validate(completion.model_dump())

    # Multiple choices - get consensus content
    async_parsed_choice_contents: list[dict[str, Any]] = []
    for choice in completion.choices:
        if choice.message.content:
            async_parsed_choice_contents.append(_safe_parse_content(choice.message.content))

    # Align lists/objects across choices once before consensus
    if len(async_parsed_choice_contents) >= 2:
        aligned_seq, _ = await async_recursive_list_alignments(
            async_parsed_choice_contents,
            consensus_settings.string_similarity_method,
            async_get_openai_embeddings_from_text,
            client,
            consensus_settings.min_support_ratio,
        )
        async_parsed_choice_contents = [(d if isinstance(d, dict) else {}) for d in aligned_seq]

    consensus_content, likelihoods = await async_consensus_values(
        async_parsed_choice_contents,
        consensus_settings,
        async_get_openai_embeddings_from_text,
        client=client,
    )

    # Parse the consensus content if response_format is a BaseModel
    parsed_consensus = None
    if response_format and consensus_content is not None:
        try:
            # Check if the response_format is a BaseModel subclass
            if isinstance(response_format, type) and issubclass(response_format, BaseModel):
                parsed_consensus = response_format.model_validate(consensus_content)
        except Exception:
            # If parsing fails, keep parsed as None
            parsed_consensus = None

    # Create consolidated message
    # For parsed completions, content should be a JSON string
    content_str = _format_consensus_content(consensus_content)
    consolidated_message = ParsedChatCompletionMessage(
        role="assistant",
        content=content_str,
        function_call=completion.choices[0].message.function_call if completion.choices else None,
        tool_calls=completion.choices[0].message.tool_calls if completion.choices else None,
        refusal=completion.choices[0].message.refusal if completion.choices else None,
        parsed=parsed_consensus,
    )

    # Create consolidated choice (consensus result)
    consolidated_choice = ParsedChoice(
        finish_reason=completion.choices[0].finish_reason if completion.choices else "stop",
        index=0,
        message=consolidated_message,
        logprobs=completion.choices[0].logprobs if completion.choices else None,
    )

    # Keep original individual choices with updated indices
    individual_choices = []
    for i, choice in enumerate(completion.choices):
        individual_choice = ParsedChoice(finish_reason=choice.finish_reason, index=i + 1, message=choice.message, logprobs=choice.logprobs)
        individual_choices.append(individual_choice)

    # Combine consensus choice with individual choices
    all_choices = [consolidated_choice] + individual_choices

    return KLLMsParsedChatCompletion.model_validate({**completion.model_dump(), "choices": all_choices, "likelihoods": likelihoods})
