from typing import Any, List, Optional, Union, Sequence
from openai.types.chat import ChatCompletion, ParsedChatCompletion, ChatCompletionMessage
from openai.types.completion_usage import CompletionUsage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.parsed_chat_completion import ParsedChoice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.parsed_chat_completion import ParsedChatCompletionMessage
from openai.types.responses import Response
from openai.types.responses.response_usage import ResponseUsage, InputTokensDetails, OutputTokensDetails

from ..types.completions import KLLMsChatCompletion
from ..types.parsed import KLLMsParsedChatCompletion
from ..types.responses import KLLMsResponse
from .consensus_utils import (
    consensus_values,
    ConsensusSettings,
    SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    intermediary_consensus_cleanup,
    async_consensus_values,
)


def dummy_get_openai_embeddings_from_text(strings: List[str]) -> List[List[float]]:
    """Dummy function for embeddings when not available"""
    return [[0.0] * 384 for _ in strings]


async def async_dummy_get_openai_embeddings_from_text(strings: List[str]) -> List[List[float]]:
    """Async dummy function for embeddings when not available"""
    return [[0.0] * 384 for _ in strings]


def consolidateResponseUsage(responses: List[Response]) -> Optional[ResponseUsage]:
    """
    Consolidate ResponseUsage objects from multiple responses.

    Args:
        responses: List of Response objects with usage information

    Returns:
        Consolidated ResponseUsage object or None if no usage data
    """
    usages = [r.usage for r in responses if r.usage is not None]
    if not usages:
        return None

    # Sum up usage statistics
    total_input_tokens = sum(u.input_tokens for u in usages)
    total_output_tokens = sum(u.output_tokens for u in usages)
    total_tokens = sum(u.total_tokens for u in usages)

    # Sum up detailed token counts
    total_cached_tokens = sum(u.input_tokens_details.cached_tokens for u in usages)
    total_reasoning_tokens = sum(u.output_tokens_details.reasoning_tokens for u in usages)

    return ResponseUsage(
        input_tokens=total_input_tokens,
        input_tokens_details=InputTokensDetails(cached_tokens=total_cached_tokens),
        output_tokens=total_output_tokens,
        output_tokens_details=OutputTokensDetails(reasoning_tokens=total_reasoning_tokens),
        total_tokens=total_tokens,
    )


def consolidateCompletionUsage(completions: Sequence[Union[ChatCompletion, ParsedChatCompletion[Any]]]) -> Optional[CompletionUsage]:
    """
    Consolidate CompletionUsage objects from multiple completions.

    Args:
        completions: List of completion objects with usage information

    Returns:
        Consolidated CompletionUsage object or None if no usage data
    """
    usages = [c.usage for c in completions if c.usage is not None]
    if not usages:
        return None

    # Sum up usage statistics
    total_prompt_tokens = sum(u.prompt_tokens or 0 for u in usages)
    total_completion_tokens = sum(u.completion_tokens or 0 for u in usages)
    total_tokens = sum(u.total_tokens or 0 for u in usages)

    return CompletionUsage(prompt_tokens=total_prompt_tokens, completion_tokens=total_completion_tokens, total_tokens=total_tokens)


def consolidate_chat_completions(
    completions: List[ChatCompletion],
    sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    consensus_settings: Optional[ConsensusSettings] = None,
) -> KLLMsChatCompletion:
    """
    Consolidate multiple ChatCompletion objects into a single KLLMsChatCompletion with consensus.

    Args:
        completions: List of ChatCompletion objects to consolidate
        consensus_settings: Settings for consensus algorithm. If None, uses default settings.
        sync_get_openai_embeddings_from_text: Function to get embeddings for text similarity

    Returns:
        KLLMsChatCompletion with consolidated response and likelihood scores
    """
    if not completions:
        raise ValueError("Cannot consolidate empty list of completions")

    if len(completions) == 1:
        # Single completion - just wrap it
        completion = completions[0]
        return KLLMsChatCompletion(
            id=completion.id,
            choices=completion.choices,
            created=completion.created,
            model=completion.model,
            object=completion.object,
            usage=completion.usage,
            system_fingerprint=completion.system_fingerprint,
            likelihoods=None,
        )

    # Use default settings if not provided
    if consensus_settings is None:
        consensus_settings = ConsensusSettings()

    # Use dummy embeddings if not provided
    if sync_get_openai_embeddings_from_text is None:
        sync_get_openai_embeddings_from_text = dummy_get_openai_embeddings_from_text

        # Extract all choice contents for consensus
    all_contents = []
    for completion in completions:
        if completion.choices and completion.choices[0].message.content:
            all_contents.append(completion.choices[0].message.content)
        else:
            all_contents.append(None)

    # Get consensus content
    consensus_content, confidence = consensus_values(all_contents, consensus_settings, sync_get_openai_embeddings_from_text, is_last_chunk=True)

    # Initialize defaults in case consensus_values doesn't return expected values
    if consensus_content is None:
        consensus_content = ""
    if confidence is None:
        confidence = 0.0

    # Create consolidated usage
    consolidated_usage = consolidateCompletionUsage(completions)

    # Use the first completion as template
    template = completions[0]

    # Create consolidated message
    consolidated_message = ChatCompletionMessage(
        role="assistant",
        content=consensus_content if consensus_content is not None else "",
        function_call=template.choices[0].message.function_call if template.choices else None,
        tool_calls=template.choices[0].message.tool_calls if template.choices else None,
        refusal=template.choices[0].message.refusal if template.choices else None,
    )

    # Create consolidated choice (consensus result)
    consolidated_choice = Choice(
        finish_reason=template.choices[0].finish_reason if template.choices else "stop",
        index=0,
        message=consolidated_message,
        logprobs=template.choices[0].logprobs if template.choices else None,
    )

    # Create individual model choices
    individual_choices = []
    for i, completion in enumerate(completions):
        if completion.choices:
            individual_choice = Choice(
                finish_reason=completion.choices[0].finish_reason, index=i + 1, message=completion.choices[0].message, logprobs=completion.choices[0].logprobs
            )
            individual_choices.append(individual_choice)

    # Combine consensus choice with individual choices
    all_choices = [consolidated_choice] + individual_choices

    # Create likelihood information
    likelihoods = {"content": confidence} if isinstance(confidence, (int, float)) else None

    return KLLMsChatCompletion(
        id=template.id,
        choices=all_choices,
        created=template.created,
        model=template.model,
        object=template.object,
        usage=consolidated_usage,
        system_fingerprint=template.system_fingerprint,
        likelihoods=likelihoods,
    )


def consolidate_parsed_chat_completions(
    completions: List[ParsedChatCompletion],
    sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    consensus_settings: Optional[ConsensusSettings] = None,
) -> KLLMsParsedChatCompletion:
    """
    Consolidate multiple ParsedChatCompletion objects into a single KLLMsParsedChatCompletion with consensus.

    Args:
        completions: List of ParsedChatCompletion objects to consolidate
        consensus_settings: Settings for consensus algorithm. If None, uses default settings.
        sync_get_openai_embeddings_from_text: Function to get embeddings for text similarity

    Returns:
        KLLMsParsedChatCompletion with consolidated response and likelihood scores
    """
    if not completions:
        raise ValueError("Cannot consolidate empty list of parsed completions")

    if len(completions) == 1:
        # Single completion - just wrap it with full confidence
        completion = completions[0]
        
        # For single completions, create a confidence structure matching the parsed object
        if completion.choices and completion.choices[0].message.parsed:
            # Convert parsed object to dict to understand its structure
            if hasattr(completion.choices[0].message.parsed, "model_dump"):
                parsed_dict = completion.choices[0].message.parsed.model_dump()
            elif hasattr(completion.choices[0].message.parsed, "dict"):
                parsed_dict = completion.choices[0].message.parsed.dict()
            else:
                parsed_dict = completion.choices[0].message.parsed.__dict__
            
            # Create a confidence structure with 1.0 confidence for all fields
            def create_confidence_structure(obj):
                if isinstance(obj, dict):
                    return {key: create_confidence_structure(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [create_confidence_structure(item) for item in obj]
                else:
                    return 1.0
            
            likelihoods = create_confidence_structure(parsed_dict)
        else:
            likelihoods = {"_consensus_score": 1.0}
        
        return KLLMsParsedChatCompletion(
            id=completion.id,
            choices=completion.choices,
            created=completion.created,
            model=completion.model,
            object=completion.object,
            usage=completion.usage,
            system_fingerprint=completion.system_fingerprint,
            likelihoods=likelihoods,
        )

    # Use default settings if not provided
    if consensus_settings is None:
        consensus_settings = ConsensusSettings()

    # Use dummy embeddings if not provided
    if sync_get_openai_embeddings_from_text is None:
        sync_get_openai_embeddings_from_text = dummy_get_openai_embeddings_from_text

    # Extract all parsed objects for consensus
    all_parsed_objects = []
    for completion in completions:
        if completion.choices and completion.choices[0].message.parsed:
            # Convert parsed object to dict for consensus processing
            if hasattr(completion.choices[0].message.parsed, "model_dump"):
                parsed_dict = completion.choices[0].message.parsed.model_dump()
            elif hasattr(completion.choices[0].message.parsed, "dict"):
                parsed_dict = completion.choices[0].message.parsed.dict()
            else:
                # Fallback for other types
                parsed_dict = completion.choices[0].message.parsed.__dict__
            all_parsed_objects.append(parsed_dict)
        else:
            all_parsed_objects.append(None)

    # Get consensus parsed object
    consensus_parsed_dict, confidence = consensus_values(all_parsed_objects, consensus_settings, sync_get_openai_embeddings_from_text, is_last_chunk=True)

    # Clean up the consensus result
    consensus_parsed_dict = intermediary_consensus_cleanup(consensus_parsed_dict)

    # Create consolidated usage
    consolidated_usage = consolidateCompletionUsage(completions)

    # Use the first completion as template
    template = completions[0]

    # Get the original parsed object type from the first completion
    original_parsed_type = type(template.choices[0].message.parsed) if template.choices and template.choices[0].message.parsed else None

    # Reconstruct the parsed object with consensus data
    if original_parsed_type and consensus_parsed_dict:
        try:
            # Try to reconstruct the original type
            if hasattr(original_parsed_type, "model_validate"):
                consensus_parsed = original_parsed_type.model_validate(consensus_parsed_dict)
            elif hasattr(original_parsed_type, "parse_obj"):
                consensus_parsed = original_parsed_type.parse_obj(consensus_parsed_dict)
            else:
                # Fallback: create a wrapper that mimics Pydantic behavior
                class DictWrapper:
                    def __init__(self, data):
                        self._data = data
                        # Copy attributes to make it behave like an object
                        for key, value in data.items():
                            setattr(self, key, value)
                    
                    def model_dump(self):
                        return self._data
                    
                    def dict(self):
                        return self._data
                
                consensus_parsed = DictWrapper(consensus_parsed_dict)
        except Exception as e:
            # If reconstruction fails, create a wrapper that still provides the expected interface
            class DictWrapper:
                def __init__(self, data):
                    self._data = data
                    # Copy attributes to make it behave like an object
                    for key, value in data.items():
                        setattr(self, key, value)
                
                def model_dump(self):
                    return self._data
                
                def dict(self):
                    return self._data
            
            consensus_parsed = DictWrapper(consensus_parsed_dict)
    else:
        # Handle case where we don't have original type or consensus data
        if consensus_parsed_dict:
            class DictWrapper:
                def __init__(self, data):
                    self._data = data
                    # Copy attributes to make it behave like an object
                    for key, value in data.items():
                        setattr(self, key, value)
                
                def model_dump(self):
                    return self._data
                
                def dict(self):
                    return self._data
            
            consensus_parsed = DictWrapper(consensus_parsed_dict)
        else:
            consensus_parsed = None

    # Create consolidated message
    consolidated_message = ParsedChatCompletionMessage(
        role="assistant",
        content=template.choices[0].message.content if template.choices else None,
        function_call=template.choices[0].message.function_call if template.choices else None,
        tool_calls=template.choices[0].message.tool_calls if template.choices else None,
        refusal=template.choices[0].message.refusal if template.choices else None,
        parsed=consensus_parsed,
    )

    # Create consolidated choice (consensus result)
    consolidated_choice = ParsedChoice(
        finish_reason=template.choices[0].finish_reason if template.choices else "stop",
        index=0,
        message=consolidated_message,
        logprobs=template.choices[0].logprobs if template.choices else None,
    )

    # Create individual model choices
    individual_choices = []
    for i, completion in enumerate(completions):
        if completion.choices:
            individual_choice = ParsedChoice(
                finish_reason=completion.choices[0].finish_reason, index=i + 1, message=completion.choices[0].message, logprobs=completion.choices[0].logprobs
            )
            individual_choices.append(individual_choice)

    # Combine consensus choice with individual choices
    all_choices = [consolidated_choice] + individual_choices

    # Create likelihood information from confidence
    likelihoods = confidence if isinstance(confidence, dict) else {"_consensus_score": confidence}

    return KLLMsParsedChatCompletion(
        id=template.id,
        choices=all_choices,
        created=template.created,
        model=template.model,
        object=template.object,
        usage=consolidated_usage,
        system_fingerprint=template.system_fingerprint,
        likelihoods=likelihoods,
    )


def consolidate_responses(
    responses: List[Response],
    sync_get_openai_embeddings_from_text: SYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    consensus_settings: Optional[ConsensusSettings] = None,
) -> KLLMsResponse:
    """
    Consolidate multiple Response objects into a single KLLMsResponse with consensus.

    Args:
        responses: List of Response objects to consolidate
        consensus_settings: Settings for consensus algorithm. If None, uses default settings.
        sync_get_openai_embeddings_from_text: Function to get embeddings for text similarity

    Returns:
        KLLMsResponse with consolidated response and likelihood scores
    """
    if not responses:
        raise ValueError("Cannot consolidate empty list of responses")

    if len(responses) == 1:
        # Single response - just wrap it
        response = responses[0]
        return KLLMsResponse(
            id=response.id,
            created_at=response.created_at,
            error=response.error,
            incomplete_details=response.incomplete_details,
            instructions=response.instructions,
            metadata=response.metadata,
            model=response.model,
            object=response.object,
            output=response.output,
            parallel_tool_calls=response.parallel_tool_calls,
            temperature=response.temperature,
            tool_choice=response.tool_choice,
            tools=response.tools,
            top_p=response.top_p,
            background=response.background,
            max_output_tokens=response.max_output_tokens,
            max_tool_calls=response.max_tool_calls,
            previous_response_id=response.previous_response_id,
            prompt=response.prompt,
            reasoning=response.reasoning,
            service_tier=response.service_tier,
            status=response.status,
            text=response.text,
            top_logprobs=response.top_logprobs,
            truncation=response.truncation,
            usage=response.usage,
            user=response.user,
            likelihoods=None,
            individual_responses=None,
        )

    # Use default settings if not provided
    if consensus_settings is None:
        consensus_settings = ConsensusSettings()

    # Use dummy embeddings if not provided
    if sync_get_openai_embeddings_from_text is None:
        sync_get_openai_embeddings_from_text = dummy_get_openai_embeddings_from_text

    # Extract outputs for consensus - handle different output types
    all_outputs = []
    for response in responses:
        if response.output:
            # Try to extract structured data from output
            if isinstance(response.output, list):
                # Handle list of output items
                output_data = {}
                for i, item in enumerate(response.output):
                    if hasattr(item, "content"):
                        # Extract content from each item
                        for j, content_item in enumerate(getattr(item, "content", [])):
                            if hasattr(content_item, "text"):
                                output_data[f"item_{i}_content_{j}"] = getattr(content_item, "text", "")
                            elif hasattr(content_item, "data"):
                                output_data[f"item_{i}_data_{j}"] = getattr(content_item, "data", "")
                    elif hasattr(item, "text"):
                        output_data[f"item_{i}_text"] = getattr(item, "text", "")
                    elif hasattr(item, "data"):
                        output_data[f"item_{i}_data"] = getattr(item, "data", "")
                all_outputs.append(output_data if output_data else None)
            else:
                # Handle single output item
                if hasattr(response.output, "model_dump"):
                    all_outputs.append(response.output.model_dump())
                elif hasattr(response.output, "dict"):
                    all_outputs.append(response.output.dict())
                else:
                    all_outputs.append(response.output.__dict__ if hasattr(response.output, "__dict__") else str(response.output))
        else:
            all_outputs.append(None)

    # Get consensus output
    consensus_output, confidence = consensus_values(all_outputs, consensus_settings, sync_get_openai_embeddings_from_text, is_last_chunk=True)

    # Clean up the consensus result
    consensus_output = intermediary_consensus_cleanup(consensus_output)

    # Create consolidated usage if available
    consolidated_usage = consolidateResponseUsage(responses)

    # Use the first response as template
    template = responses[0]

    # Create likelihood information from confidence
    likelihoods = confidence if isinstance(confidence, dict) else {"_consensus_score": confidence}

    # Create the consolidated response
    # Note: This creates a single response with the consensus output
    # Individual responses can be accessed separately if needed
    return KLLMsResponse(
        id=template.id,
        created_at=template.created_at,
        error=template.error,
        incomplete_details=template.incomplete_details,
        instructions=template.instructions,
        metadata=template.metadata,
        model=template.model,
        object=template.object,
        output=consensus_output,
        parallel_tool_calls=template.parallel_tool_calls,
        temperature=template.temperature,
        tool_choice=template.tool_choice,
        tools=template.tools,
        top_p=template.top_p,
        background=template.background,
        max_output_tokens=template.max_output_tokens,
        max_tool_calls=template.max_tool_calls,
        previous_response_id=template.previous_response_id,
        prompt=template.prompt,
        reasoning=template.reasoning,
        service_tier=template.service_tier,
        status=template.status,
        text=template.text,
        top_logprobs=template.top_logprobs,
        truncation=template.truncation,
        usage=consolidated_usage,
        user=template.user,
        likelihoods=likelihoods,
        individual_responses=responses,
    )


# ============================================================================
# ASYNC VERSIONS OF CONSOLIDATION FUNCTIONS
# ============================================================================


async def async_consolidate_chat_completions(
    completions: List[ChatCompletion],
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    consensus_settings: Optional[ConsensusSettings] = None,
) -> KLLMsChatCompletion:
    """
    Async version of consolidate_chat_completions.
    Consolidate multiple ChatCompletion objects into a single KLLMsChatCompletion with consensus.

    Args:
        completions: List of ChatCompletion objects to consolidate
        consensus_settings: Settings for consensus algorithm. If None, uses default settings.
        async_get_openai_embeddings_from_text: Async function to get embeddings for text similarity

    Returns:
        KLLMsChatCompletion with consolidated response and likelihood scores
    """

    if not completions:
        raise ValueError("Cannot consolidate empty list of completions")

    if len(completions) == 1:
        # Single completion - just wrap it
        completion = completions[0]
        return KLLMsChatCompletion(
            id=completion.id,
            choices=completion.choices,
            created=completion.created,
            model=completion.model,
            object=completion.object,
            usage=completion.usage,
            system_fingerprint=completion.system_fingerprint,
            likelihoods=None,
        )

    # Use default settings if not provided
    if consensus_settings is None:
        consensus_settings = ConsensusSettings()

    # Use dummy embeddings if not provided
    if async_get_openai_embeddings_from_text is None:
        async_get_openai_embeddings_from_text = async_dummy_get_openai_embeddings_from_text

    # Extract all choice contents for consensus
    all_contents = []
    for completion in completions:
        if completion.choices and completion.choices[0].message.content:
            all_contents.append(completion.choices[0].message.content)
        else:
            all_contents.append(None)

    # Get consensus content
    consensus_content, confidence = await async_consensus_values(all_contents, consensus_settings, async_get_openai_embeddings_from_text, is_last_chunk=True)

    # Initialize defaults in case consensus_values doesn't return expected values
    if consensus_content is None:
        consensus_content = ""
    if confidence is None:
        confidence = 0.0

    # Create consolidated usage
    consolidated_usage = consolidateCompletionUsage(completions)

    # Use the first completion as template
    template = completions[0]

    # Create consolidated message
    consolidated_message = ChatCompletionMessage(
        role="assistant",
        content=consensus_content if consensus_content is not None else "",
        function_call=template.choices[0].message.function_call if template.choices else None,
        tool_calls=template.choices[0].message.tool_calls if template.choices else None,
        refusal=template.choices[0].message.refusal if template.choices else None,
    )

    # Create consolidated choice (consensus result)
    consolidated_choice = Choice(
        finish_reason=template.choices[0].finish_reason if template.choices else "stop",
        index=0,
        message=consolidated_message,
        logprobs=template.choices[0].logprobs if template.choices else None,
    )

    # Create individual model choices
    individual_choices = []
    for i, completion in enumerate(completions):
        if completion.choices:
            individual_choice = Choice(
                finish_reason=completion.choices[0].finish_reason, index=i + 1, message=completion.choices[0].message, logprobs=completion.choices[0].logprobs
            )
            individual_choices.append(individual_choice)

    # Combine consensus choice with individual choices
    all_choices = [consolidated_choice] + individual_choices

    # Create likelihood information
    likelihoods = {"content": confidence} if isinstance(confidence, (int, float)) else None

    return KLLMsChatCompletion(
        id=template.id,
        choices=all_choices,
        created=template.created,
        model=template.model,
        object=template.object,
        usage=consolidated_usage,
        system_fingerprint=template.system_fingerprint,
        likelihoods=likelihoods,
    )


async def async_consolidate_parsed_chat_completions(
    completions: List[ParsedChatCompletion],
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    consensus_settings: Optional[ConsensusSettings] = None,
) -> KLLMsParsedChatCompletion:
    """
    Async version of consolidate_parsed_chat_completions.
    Consolidate multiple ParsedChatCompletion objects into a single KLLMsParsedChatCompletion with consensus.

    Args:
        completions: List of ParsedChatCompletion objects to consolidate
        consensus_settings: Settings for consensus algorithm. If None, uses default settings.
        async_get_openai_embeddings_from_text: Async function to get embeddings for text similarity

    Returns:
        KLLMsParsedChatCompletion with consolidated response and likelihood scores
    """
    # Import async consensus functions here to avoid circular imports

    if not completions:
        raise ValueError("Cannot consolidate empty list of parsed completions")

    if len(completions) == 1:
        # Single completion - just wrap it with full confidence
        completion = completions[0]
        
        # For single completions, create a confidence structure matching the parsed object
        if completion.choices and completion.choices[0].message.parsed:
            # Convert parsed object to dict to understand its structure
            if hasattr(completion.choices[0].message.parsed, "model_dump"):
                parsed_dict = completion.choices[0].message.parsed.model_dump()
            elif hasattr(completion.choices[0].message.parsed, "dict"):
                parsed_dict = completion.choices[0].message.parsed.dict()
            else:
                parsed_dict = completion.choices[0].message.parsed.__dict__
            
            # Create a confidence structure with 1.0 confidence for all fields
            def create_confidence_structure(obj):
                if isinstance(obj, dict):
                    return {key: create_confidence_structure(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [create_confidence_structure(item) for item in obj]
                else:
                    return 1.0
            
            likelihoods = create_confidence_structure(parsed_dict)
        else:
            likelihoods = {"_consensus_score": 1.0}
        
        return KLLMsParsedChatCompletion(
            id=completion.id,
            choices=completion.choices,
            created=completion.created,
            model=completion.model,
            object=completion.object,
            usage=completion.usage,
            system_fingerprint=completion.system_fingerprint,
            likelihoods=likelihoods,
        )

    # Use default settings if not provided
    if consensus_settings is None:
        consensus_settings = ConsensusSettings()

    # Use dummy embeddings if not provided
    if async_get_openai_embeddings_from_text is None:
        async_get_openai_embeddings_from_text = async_dummy_get_openai_embeddings_from_text

    # Extract all parsed objects for consensus
    all_parsed_objects = []
    for completion in completions:
        if completion.choices and completion.choices[0].message.parsed:
            # Convert parsed object to dict for consensus processing
            if hasattr(completion.choices[0].message.parsed, "model_dump"):
                parsed_dict = completion.choices[0].message.parsed.model_dump()
            elif hasattr(completion.choices[0].message.parsed, "dict"):
                parsed_dict = completion.choices[0].message.parsed.dict()
            else:
                # Fallback for other types
                parsed_dict = completion.choices[0].message.parsed.__dict__
            all_parsed_objects.append(parsed_dict)
        else:
            all_parsed_objects.append(None)

    # Get consensus parsed object
    consensus_parsed_dict, confidence = await async_consensus_values(all_parsed_objects, consensus_settings, async_get_openai_embeddings_from_text, is_last_chunk=True)

    # Clean up the consensus result
    consensus_parsed_dict = intermediary_consensus_cleanup(consensus_parsed_dict)

    # Create consolidated usage
    consolidated_usage = consolidateCompletionUsage(completions)

    # Use the first completion as template
    template = completions[0]

    # Get the original parsed object type from the first completion
    original_parsed_type = type(template.choices[0].message.parsed) if template.choices and template.choices[0].message.parsed else None

    # Reconstruct the parsed object with consensus data
    if original_parsed_type and consensus_parsed_dict:
        try:
            # Try to reconstruct the original type
            if hasattr(original_parsed_type, "model_validate"):
                consensus_parsed = original_parsed_type.model_validate(consensus_parsed_dict)
            elif hasattr(original_parsed_type, "parse_obj"):
                consensus_parsed = original_parsed_type.parse_obj(consensus_parsed_dict)
            else:
                # Fallback: create a wrapper that mimics Pydantic behavior
                class DictWrapper:
                    def __init__(self, data):
                        self._data = data
                        # Copy attributes to make it behave like an object
                        for key, value in data.items():
                            setattr(self, key, value)
                    
                    def model_dump(self):
                        return self._data
                    
                    def dict(self):
                        return self._data
                
                consensus_parsed = DictWrapper(consensus_parsed_dict)
        except Exception as e:
            # If reconstruction fails, create a wrapper that still provides the expected interface
            class DictWrapper:
                def __init__(self, data):
                    self._data = data
                    # Copy attributes to make it behave like an object
                    for key, value in data.items():
                        setattr(self, key, value)
                
                def model_dump(self):
                    return self._data
                
                def dict(self):
                    return self._data
            
            consensus_parsed = DictWrapper(consensus_parsed_dict)
    else:
        # Handle case where we don't have original type or consensus data
        if consensus_parsed_dict:
            class DictWrapper:
                def __init__(self, data):
                    self._data = data
                    # Copy attributes to make it behave like an object
                    for key, value in data.items():
                        setattr(self, key, value)
                
                def model_dump(self):
                    return self._data
                
                def dict(self):
                    return self._data
            
            consensus_parsed = DictWrapper(consensus_parsed_dict)
        else:
            consensus_parsed = None

    # Create consolidated message
    consolidated_message = ParsedChatCompletionMessage(
        role="assistant",
        content=template.choices[0].message.content if template.choices else None,
        function_call=template.choices[0].message.function_call if template.choices else None,
        tool_calls=template.choices[0].message.tool_calls if template.choices else None,
        refusal=template.choices[0].message.refusal if template.choices else None,
        parsed=consensus_parsed,
    )

    # Create consolidated choice (consensus result)
    consolidated_choice = ParsedChoice(
        finish_reason=template.choices[0].finish_reason if template.choices else "stop",
        index=0,
        message=consolidated_message,
        logprobs=template.choices[0].logprobs if template.choices else None,
    )

    # Create individual model choices
    individual_choices = []
    for i, completion in enumerate(completions):
        if completion.choices:
            individual_choice = ParsedChoice(
                finish_reason=completion.choices[0].finish_reason, index=i + 1, message=completion.choices[0].message, logprobs=completion.choices[0].logprobs
            )
            individual_choices.append(individual_choice)

    # Combine consensus choice with individual choices
    all_choices = [consolidated_choice] + individual_choices

    # Create likelihood information from confidence
    likelihoods = confidence if isinstance(confidence, dict) else {"_consensus_score": confidence}

    return KLLMsParsedChatCompletion(
        id=template.id,
        choices=all_choices,
        created=template.created,
        model=template.model,
        object=template.object,
        usage=consolidated_usage,
        system_fingerprint=template.system_fingerprint,
        likelihoods=likelihoods,
    )


async def async_consolidate_responses(
    responses: List[Response],
    async_get_openai_embeddings_from_text: ASYNC_GET_OPENAI_EMBEDDINGS_FROM_TEXT_TYPE,
    consensus_settings: Optional[ConsensusSettings] = None,
) -> KLLMsResponse:
    """
    Async version of consolidate_responses.
    Consolidate multiple Response objects into a single KLLMsResponse with consensus.

    Args:
        responses: List of Response objects to consolidate
        consensus_settings: Settings for consensus algorithm. If None, uses default settings.
        async_get_openai_embeddings_from_text: Async function to get embeddings for text similarity

    Returns:
        KLLMsResponse with consolidated response and likelihood scores
    """

    if not responses:
        raise ValueError("Cannot consolidate empty list of responses")

    if len(responses) == 1:
        # Single response - just wrap it
        response = responses[0]
        return KLLMsResponse(
            id=response.id,
            created_at=response.created_at,
            error=response.error,
            incomplete_details=response.incomplete_details,
            instructions=response.instructions,
            metadata=response.metadata,
            model=response.model,
            object=response.object,
            output=response.output,
            parallel_tool_calls=response.parallel_tool_calls,
            temperature=response.temperature,
            tool_choice=response.tool_choice,
            tools=response.tools,
            top_p=response.top_p,
            background=response.background,
            max_output_tokens=response.max_output_tokens,
            max_tool_calls=response.max_tool_calls,
            previous_response_id=response.previous_response_id,
            prompt=response.prompt,
            reasoning=response.reasoning,
            service_tier=response.service_tier,
            status=response.status,
            text=response.text,
            top_logprobs=response.top_logprobs,
            truncation=response.truncation,
            usage=response.usage,
            user=response.user,
            likelihoods=None,
            individual_responses=None,
        )

    # Use default settings if not provided
    if consensus_settings is None:
        consensus_settings = ConsensusSettings()

    # Use dummy embeddings if not provided
    if async_get_openai_embeddings_from_text is None:
        async_get_openai_embeddings_from_text = async_dummy_get_openai_embeddings_from_text

    # Extract outputs for consensus - handle different output types
    all_outputs = []
    for response in responses:
        if response.output:
            # Try to extract structured data from output
            if isinstance(response.output, list):
                # Handle list of output items
                output_data = {}
                for i, item in enumerate(response.output):
                    if hasattr(item, "content"):
                        # Extract content from each item
                        for j, content_item in enumerate(getattr(item, "content", [])):
                            if hasattr(content_item, "text"):
                                output_data[f"item_{i}_content_{j}"] = getattr(content_item, "text", "")
                            elif hasattr(content_item, "data"):
                                output_data[f"item_{i}_data_{j}"] = getattr(content_item, "data", "")
                    elif hasattr(item, "text"):
                        output_data[f"item_{i}_text"] = getattr(item, "text", "")
                    elif hasattr(item, "data"):
                        output_data[f"item_{i}_data"] = getattr(item, "data", "")
                all_outputs.append(output_data if output_data else None)
            else:
                # Handle single output item
                if hasattr(response.output, "model_dump"):
                    all_outputs.append(response.output.model_dump())
                elif hasattr(response.output, "dict"):
                    all_outputs.append(response.output.dict())
                else:
                    all_outputs.append(response.output.__dict__ if hasattr(response.output, "__dict__") else str(response.output))
        else:
            all_outputs.append(None)

    # Get consensus output
    consensus_output, confidence = await async_consensus_values(all_outputs, consensus_settings, async_get_openai_embeddings_from_text, is_last_chunk=True)

    # Clean up the consensus result
    consensus_output = intermediary_consensus_cleanup(consensus_output)

    # Create consolidated usage if available
    consolidated_usage = consolidateResponseUsage(responses)

    # Use the first response as template
    template = responses[0]

    # Create likelihood information from confidence
    likelihoods = confidence if isinstance(confidence, dict) else {"_consensus_score": confidence}

    # Create the consolidated response
    # Note: This creates a single response with the consensus output
    # Individual responses can be accessed separately if needed
    return KLLMsResponse(
        id=template.id,
        created_at=template.created_at,
        error=template.error,
        incomplete_details=template.incomplete_details,
        instructions=template.instructions,
        metadata=template.metadata,
        model=template.model,
        object=template.object,
        output=consensus_output,
        parallel_tool_calls=template.parallel_tool_calls,
        temperature=template.temperature,
        tool_choice=template.tool_choice,
        tools=template.tools,
        top_p=template.top_p,
        background=template.background,
        max_output_tokens=template.max_output_tokens,
        max_tool_calls=template.max_tool_calls,
        previous_response_id=template.previous_response_id,
        prompt=template.prompt,
        reasoning=template.reasoning,
        service_tier=template.service_tier,
        status=template.status,
        text=template.text,
        top_logprobs=template.top_logprobs,
        truncation=template.truncation,
        usage=consolidated_usage,
        user=template.user,
        likelihoods=likelihoods,
        individual_responses=responses,
    )
