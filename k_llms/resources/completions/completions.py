import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.lib._parsing import ResponseFormatT

from ...utils.consolidation import consolidate_chat_completions, consolidate_parsed_chat_completions
from ...types.completions import KLLMsChatCompletion
from ...types.parsed import KLLMsParsedChatCompletion

if TYPE_CHECKING:
    from ...client import KLLMs, AsyncKLLMs


class Completions:
    def __init__(self, wrapper: "KLLMs"):
        self._wrapper = wrapper

    def create(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        response_format: Optional[type[ResponseFormatT]] = None,
        **kwargs: Any,
    ) -> KLLMsChatCompletion:
        # Always force stream=False since we don't support streaming
        kwargs.pop("stream", None)

        # if not response_format:
        #    raise ValueError("response_format is required")

        # Build the call parameters
        call_params = {"messages": messages, "model": model, "stream": False}

        # Add explicit parameters if provided
        if temperature is not None:
            call_params["temperature"] = temperature
        if max_tokens is not None:
            call_params["max_tokens"] = max_tokens
        if top_p is not None:
            call_params["top_p"] = top_p
        if frequency_penalty is not None:
            call_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            call_params["presence_penalty"] = presence_penalty
        if stop is not None:
            call_params["stop"] = stop
        if seed is not None:
            call_params["seed"] = seed

        # Add any additional kwargs
        call_params.update(kwargs)

        # Create a wrapper function that matches the expected signature
        def embeddings_wrapper(texts: List[str]) -> List[List[float]]:
            return self._wrapper.get_embeddings(texts, "text-embedding-3-small", 2048, False)

        if n and n > 1:
            # Use OpenAI's native n parameter to generate multiple completions in one request
            call_params["n"] = n
            completion = self._wrapper.client.chat.completions.create(**call_params)
            # The completion will have multiple choices, consolidate them
            return consolidate_chat_completions(
                completion,
                embeddings_wrapper,
                api_key=self._wrapper.client.api_key,
                base_url=self._wrapper.client.base_url,
            )
        else:
            # Single request - wrap in KLLMsChatCompletion
            completion = self._wrapper.client.chat.completions.create(**call_params)
            return consolidate_chat_completions(
                completion,
                embeddings_wrapper,
                api_key=self._wrapper.client.api_key,
                base_url=self._wrapper.client.base_url,
            )

    def parse(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        response_format: type[ResponseFormatT],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> KLLMsParsedChatCompletion:
        # Build the call parameters
        call_params = {"messages": messages, "model": model, "response_format": response_format}

        # Add explicit parameters if provided
        if temperature is not None:
            call_params["temperature"] = temperature
        if max_tokens is not None:
            call_params["max_tokens"] = max_tokens
        if top_p is not None:
            call_params["top_p"] = top_p
        if frequency_penalty is not None:
            call_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            call_params["presence_penalty"] = presence_penalty
        if stop is not None:
            call_params["stop"] = stop
        if seed is not None:
            call_params["seed"] = seed

        # Add any additional kwargs
        call_params.update(kwargs)

        # Create a wrapper function that matches the expected signature
        def embeddings_wrapper(texts: List[str]) -> List[List[float]]:
            return self._wrapper.get_embeddings(texts, "text-embedding-3-small", 2048, False)

        if n and n > 1:
            # Use OpenAI's native n parameter to generate multiple completions in one request
            call_params["n"] = n
            completion = self._wrapper.client.beta.chat.completions.parse(**call_params)
            # The completion will have multiple choices, consolidate them
            return consolidate_parsed_chat_completions(
                completion,
                embeddings_wrapper,
                response_format=response_format,
                api_key=self._wrapper.client.api_key,
                base_url=self._wrapper.client.base_url,
            )
        else:
            # Single request - wrap in KLLMsParsedChatCompletion
            completion = self._wrapper.client.beta.chat.completions.parse(**call_params)
            return consolidate_parsed_chat_completions(
                completion,
                embeddings_wrapper,
                response_format=response_format,
                api_key=self._wrapper.client.api_key,
                base_url=self._wrapper.client.base_url,
            )


class AsyncCompletions:
    def __init__(self, wrapper: "AsyncKLLMs"):
        self._wrapper = wrapper

    async def create(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        response_format: Optional[type[ResponseFormatT]] = None,
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> KLLMsChatCompletion:
        # Always force stream=False since we don't support streaming
        kwargs.pop("stream", None)

        # if not response_format:
        #    raise ValueError("response_format is required")

        # Build the call parameters
        call_params = {"messages": messages, "model": model, "stream": False}

        # Add explicit parameters if provided
        if temperature is not None:
            call_params["temperature"] = temperature
        if max_tokens is not None:
            call_params["max_tokens"] = max_tokens
        if top_p is not None:
            call_params["top_p"] = top_p
        if frequency_penalty is not None:
            call_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            call_params["presence_penalty"] = presence_penalty
        if stop is not None:
            call_params["stop"] = stop
        if seed is not None:
            call_params["seed"] = seed

        # Add any additional kwargs
        call_params.update(kwargs)

        # Import async consolidation functions
        from ...utils.consolidation import async_consolidate_chat_completions

        # Create a wrapper function that matches the expected async signature
        async def embeddings_wrapper(texts: List[str]) -> List[List[float]]:
            return await self._wrapper.get_embeddings(texts, "text-embedding-3-small", 2048, False)

        if n and n > 1:
            # Use OpenAI's native n parameter to generate multiple completions in one request
            call_params["n"] = n
            completion = await self._wrapper.client.chat.completions.create(**call_params)
            # The completion will have multiple choices, consolidate them
            return await async_consolidate_chat_completions(
                completion,
                embeddings_wrapper,
                api_key=self._wrapper.client.api_key,
                base_url=self._wrapper.client.base_url,
            )
        else:
            # Single request - wrap in KLLMsChatCompletion
            completion = await self._wrapper.client.chat.completions.create(**call_params)
            return await async_consolidate_chat_completions(
                completion,
                embeddings_wrapper,
                api_key=self._wrapper.client.api_key,
                base_url=self._wrapper.client.base_url,
            )

    async def parse(
        self,
        *,
        messages: List[ChatCompletionMessageParam],
        model: str,
        response_format: type[ResponseFormatT],
        n: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> KLLMsParsedChatCompletion:
        # Build the call parameters
        call_params = {"messages": messages, "model": model, "response_format": response_format}

        # Add explicit parameters if provided
        if temperature is not None:
            call_params["temperature"] = temperature
        if max_tokens is not None:
            call_params["max_tokens"] = max_tokens
        if top_p is not None:
            call_params["top_p"] = top_p
        if frequency_penalty is not None:
            call_params["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            call_params["presence_penalty"] = presence_penalty
        if stop is not None:
            call_params["stop"] = stop
        if seed is not None:
            call_params["seed"] = seed

        # Add any additional kwargs
        call_params.update(kwargs)

        # Import async consolidation functions
        from ...utils.consolidation import async_consolidate_parsed_chat_completions

        # Create a wrapper function that matches the expected async signature
        async def embeddings_wrapper(texts: List[str]) -> List[List[float]]:
            return await self._wrapper.get_embeddings(texts, "text-embedding-3-small", 2048, False)

        if n and n > 1:
            # Use OpenAI's native n parameter to generate multiple completions in one request
            call_params["n"] = n
            completion = await self._wrapper.client.beta.chat.completions.parse(**call_params)
            # The completion will have multiple choices, consolidate them
            return await async_consolidate_parsed_chat_completions(
                completion,
                embeddings_wrapper,
                response_format=response_format,
                api_key=self._wrapper.client.api_key,
                base_url=self._wrapper.client.base_url,
            )
        else:
            # Single request - wrap in KLLMsParsedChatCompletion
            completion = await self._wrapper.client.beta.chat.completions.parse(**call_params)
            return await async_consolidate_parsed_chat_completions(
                completion,
                embeddings_wrapper,
                response_format=response_format,
                api_key=self._wrapper.client.api_key,
                base_url=self._wrapper.client.base_url,
            )
