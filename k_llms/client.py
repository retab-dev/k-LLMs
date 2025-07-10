import os
from typing import Any, List, Optional, Union, overload, Awaitable, Callable
import asyncio

from openai import AsyncOpenAI, OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.lib._parsing import ResponseFormatT
import tiktoken
import tqdm

from .resources.completions import Completions, AsyncCompletions
from .resources.responses import Responses, AsyncResponses
from .types.parsed import KLLMsParsedChatCompletion

# Constants for embedding models
MAX_TOKENS_PER_MODEL = {"text-embedding-3-small": 8191, "text-embedding-3-large": 8191}
PRICING = {"text-embedding-3-small": 0.020, "text-embedding-3-large": 0.13}


class BaseOpenAIWrapper:
    def __init__(
        self,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 2,
        **kwargs: Any,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.organization = organization
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self._extra_kwargs = kwargs


class KLLMs(BaseOpenAIWrapper):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = OpenAI(
            api_key=self.api_key, organization=self.organization, base_url=self.base_url, timeout=self.timeout, max_retries=self.max_retries, **self._extra_kwargs
        )
        self.chat = Chat(self)
        self.responses = Responses(self)
        self.get_embeddings: Callable[[list[str], str, int, bool], list[list[float]]] = lambda texts, model, batch_size, verbose: get_embeddings(
            self._client, texts, model, batch_size, verbose
        )

    @property
    def client(self) -> OpenAI:
        return self._client


class AsyncKLLMs(BaseOpenAIWrapper):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._client = AsyncOpenAI(
            api_key=self.api_key, organization=self.organization, base_url=self.base_url, timeout=self.timeout, max_retries=self.max_retries, **self._extra_kwargs
        )
        self.chat = AsyncChat(self)
        self.responses = AsyncResponses(self)
        self.get_embeddings: Callable[[list[str], str, int, bool], Awaitable[list[list[float]]]] = lambda texts, model, batch_size, verbose: async_get_embeddings(
            self._client, texts, model, batch_size, verbose
        )

    @property
    def client(self) -> AsyncOpenAI:
        return self._client


class Chat:
    def __init__(self, wrapper: KLLMs):
        self._wrapper = wrapper
        self.completions = Completions(wrapper)


class AsyncChat:
    def __init__(self, wrapper: AsyncKLLMs):
        self._wrapper = wrapper
        self.completions = AsyncCompletions(wrapper)


def get_embeddings(
    openai_client: OpenAI,
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 2048,
    verbose: bool = False,
) -> list[list[float]]:
    """
    Get embeddings for a list of texts using the OpenAI embeddings API.

    Args:
        texts: List of texts to embed
        model: Embedding model to use
        batch_size: Batch size for processing
        verbose: Whether to print progress and pricing info

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if model not in MAX_TOKENS_PER_MODEL:
        raise ValueError(f"Model {model} not supported. Available models: {list(MAX_TOKENS_PER_MODEL.keys())}")

    # Get encoding for token limit checking
    enc = tiktoken.encoding_for_model(model)
    max_tokens = MAX_TOKENS_PER_MODEL[model]

    # Preprocess texts to crop the maximum number of tokens
    processed_texts = [enc.decode(enc.encode(text)[:max_tokens]) for text in texts]

    embeddings: list[list[float]] = []
    total_price = 0.0

    trange: tqdm.tqdm[int] | range
    if verbose:
        trange = tqdm.trange(0, len(processed_texts), batch_size)
    else:
        trange = range(0, len(processed_texts), batch_size)

    for idx in trange:
        batch = processed_texts[idx : idx + batch_size]
        response = openai_client.embeddings.create(input=batch, model=model)
        total_price += response.usage.prompt_tokens * PRICING[model] / 1000000.0
        embeddings.extend([x.embedding for x in response.data])

    if verbose:
        print(f"TOTAL PRICE: ${total_price:.6f}")

    return embeddings


async def async_get_embeddings(
    openai_client: AsyncOpenAI,
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 2048,
    verbose: bool = False,
) -> list[list[float]]:
    """
    Get embeddings for a list of texts using the OpenAI embeddings API (async).

    Args:
        texts: List of texts to embed
        model: Embedding model to use
        batch_size: Batch size for processing
        verbose: Whether to print progress and pricing info

    Returns:
        List of embeddings (each embedding is a list of floats)
    """
    if model not in MAX_TOKENS_PER_MODEL:
        raise ValueError(f"Model {model} not supported. Available models: {list(MAX_TOKENS_PER_MODEL.keys())}")

    max_tokens = MAX_TOKENS_PER_MODEL[model]

    def selective_crop(texts: list[str], model: str) -> list[str]:
        """Crop texts that are likely to exceed token limits."""
        enc = tiktoken.encoding_for_model(model)
        return [enc.decode(enc.encode(text)[:max_tokens]) if len(text) * 3 > max_tokens else text for text in texts]

    def encode_and_crop(texts: list[str], model: str) -> list[str]:
        """Encode and crop all texts."""
        enc = tiktoken.encoding_for_model(model)
        return [enc.decode(enc.encode(text)[:max_tokens]) for text in texts]

    # Offload selective cropping to a thread
    processed_texts = await asyncio.to_thread(selective_crop, texts, model)

    embeddings: list[list[float]] = []
    total_price = 0.0

    trange: tqdm.tqdm[int] | range
    if verbose:
        trange = tqdm.trange(0, len(processed_texts), batch_size)
    else:
        trange = range(0, len(processed_texts), batch_size)

    try:
        for idx in trange:
            batch = processed_texts[idx : idx + batch_size]
            response = await openai_client.embeddings.create(input=batch, model=model)
            total_price += response.usage.prompt_tokens * PRICING[model] / 1000000.0
            embeddings.extend([x.embedding for x in response.data])
    except Exception as e:
        # Fallback: crop all strings and retry
        if verbose:
            print(f"Embedding request failed with error: {e}. Retrying with all strings cropped.")
        processed_texts = await asyncio.to_thread(encode_and_crop, texts, model)
        embeddings = []
        if verbose:
            trange = tqdm.trange(0, len(processed_texts), batch_size)
        else:
            trange = range(0, len(processed_texts), batch_size)
        for idx in trange:
            batch = processed_texts[idx : idx + batch_size]
            response = await openai_client.embeddings.create(input=batch, model=model)
            total_price += response.usage.prompt_tokens * PRICING[model] / 1000000.0
            embeddings.extend([x.embedding for x in response.data])

    if verbose:
        print(f"TOTAL PRICE: ${total_price:.6f}")

    return embeddings
