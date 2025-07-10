import asyncio
import logging
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

from openai import OpenAI, AsyncOpenAI
from openai.types.responses import Response

from ...utils.consolidation import consolidate_responses
from ...types.responses import KLLMsResponse

if TYPE_CHECKING:
    from ...client import KLLMs, AsyncKLLMs

logger = logging.getLogger(__name__)


class Responses:
    """
    Wrapper around OpenAI's Responses API that supports parallel requests and consolidation.

    This class provides functionality similar to the completions wrapper, but for the new
    stateful Responses API designed for multi-turn conversations.
    """

    def __init__(self, wrapper: "KLLMs"):
        self._wrapper = wrapper
        self._responses = wrapper.client.responses

    def create(
        self,
        *,
        model: str,
        input: Any,
        n_consensus: Optional[int] = None,
        **kwargs,
    ) -> KLLMsResponse:
        """
        Create a response using the OpenAI Responses API with optional parallel requests.

        Args:
            model: The model to use for the response
            input: Input for the conversation (format depends on API implementation)
            n_consensus: Number of parallel requests to make (default: 1)
            **kwargs: Additional parameters to pass to the API

        Returns:
            KLLMsResponse with consolidated response and likelihood scores
        """
        # Build the call parameters
        call_params = {"model": model, "input": input}
        call_params.update(kwargs)

        # Create a wrapper function that matches the expected signature
        def embeddings_wrapper(texts: List[str]) -> List[List[float]]:
            return self._wrapper.get_embeddings(texts, "text-embedding-3-small", 2048, False)

        if n_consensus and n_consensus >= 1:
            # Make n_consensus parallel requests
            responses = []
            for _ in range(n_consensus):
                response = self._responses.create(**call_params)
                responses.append(response)

            # Return consolidated result
            return consolidate_responses(responses, embeddings_wrapper)
        else:
            # Single request - wrap in KLLMsResponse
            response = self._responses.create(**call_params)
            return consolidate_responses([response], embeddings_wrapper)

    def parse(
        self,
        *,
        model: str,
        input: Any,
        n_consensus: Optional[int] = None,
        **kwargs,
    ) -> KLLMsResponse:
        """
        Parse a response using the OpenAI Responses API with structured output.

        Args:
            model: The model to use for the response
            input: Input for the conversation (format depends on API implementation)
            n_consensus: Number of parallel requests to make (default: 1)
            **kwargs: Additional parameters to pass to the API

        Returns:
            KLLMsResponse with consolidated response and likelihood scores
        """
        # Build the call parameters
        call_params = {"model": model, "input": input}
        call_params.update(kwargs)

        # Create a wrapper function that matches the expected signature
        def embeddings_wrapper(texts: List[str]) -> List[List[float]]:
            return self._wrapper.get_embeddings(texts, "text-embedding-3-small", 2048, False)

        if n_consensus and n_consensus >= 1:
            # Make n_consensus parallel requests
            responses = []
            for _ in range(n_consensus):
                response = self._responses.parse(**call_params)
                responses.append(response)

            # Return consolidated result
            # Cast ParsedResponse to Response for consolidation
            return consolidate_responses([Response(**resp.model_dump()) for resp in responses], embeddings_wrapper)
        else:
            # Single request - wrap in KLLMsResponse
            response = self._responses.parse(**call_params)
            # Cast ParsedResponse to Response for consolidation
            return consolidate_responses([Response(**response.model_dump())], embeddings_wrapper)


class AsyncResponses:
    """
    Async wrapper around OpenAI's Responses API that supports parallel requests and consolidation.
    """

    def __init__(self, wrapper: "AsyncKLLMs"):
        self._wrapper = wrapper
        self._responses = wrapper.client.responses

    async def create(
        self,
        *,
        model: str,
        input: Any,
        n_consensus: Optional[int] = None,
        **kwargs,
    ) -> KLLMsResponse:
        """
        Create a response using the OpenAI Responses API with optional parallel requests (async).

        Args:
            model: The model to use for the response
            input: Input for the conversation (format depends on API implementation)
            n_consensus: Number of parallel requests to make (default: 1)
            **kwargs: Additional parameters to pass to the API

        Returns:
            KLLMsResponse with consolidated response and likelihood scores
        """
        # Build the call parameters
        call_params = {"model": model, "input": input}
        call_params.update(kwargs)

        # Import async consolidation functions
        from ...utils.consolidation import async_consolidate_responses

        # Create a wrapper function that matches the expected async signature
        async def embeddings_wrapper(texts: List[str]) -> List[List[float]]:
            return await self._wrapper.get_embeddings(texts, "text-embedding-3-small", 2048, False)

        if n_consensus and n_consensus >= 1:
            # For multiple requests, use true parallel execution
            async def make_request():
                return await self._responses.create(**call_params)

            # Create tasks for parallel execution
            tasks = [make_request() for _ in range(n_consensus)]

            # Execute all tasks concurrently
            responses = await asyncio.gather(*tasks)

            # Return consolidated result
            return await async_consolidate_responses(responses, embeddings_wrapper)
        else:
            # Single request - wrap in KLLMsResponse
            response = await self._responses.create(**call_params)
            return await async_consolidate_responses([response], embeddings_wrapper)

    async def parse(
        self,
        *,
        model: str,
        input: Any,
        n_consensus: Optional[int] = None,
        **kwargs,
    ) -> KLLMsResponse:
        """
        Parse a response using the OpenAI Responses API with structured output (async).

        Args:
            model: The model to use for the response
            input: Input for the conversation (format depends on API implementation)
            n_consensus: Number of parallel requests to make (default: 1)
            **kwargs: Additional parameters to pass to the API

        Returns:
            KLLMsResponse with consolidated response and likelihood scores
        """
        # Build the call parameters
        call_params = {"model": model, "input": input}
        call_params.update(kwargs)

        # Import async consolidation functions
        from ...utils.consolidation import async_consolidate_responses

        # Create a wrapper function that matches the expected async signature
        async def embeddings_wrapper(texts: List[str]) -> List[List[float]]:
            return await self._wrapper.get_embeddings(texts, "text-embedding-3-small", 2048, False)

        if n_consensus and n_consensus >= 1:
            # For multiple requests, use parallel execution
            async def make_request():
                return await self._responses.parse(**call_params)

            # Create tasks for parallel execution
            tasks = [make_request() for _ in range(n_consensus)]

            # Execute all tasks concurrently
            responses = await asyncio.gather(*tasks)

            # Return consolidated result
            # Cast ParsedResponse to Response for consolidation
            return await async_consolidate_responses([Response(**resp.model_dump()) for resp in responses], embeddings_wrapper)
        else:
            # Single request - wrap in KLLMsResponse
            response = await self._responses.parse(**call_params)
            # Cast ParsedResponse to Response for consolidation
            return await async_consolidate_responses([Response(**response.model_dump())], embeddings_wrapper)
