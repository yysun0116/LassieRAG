from typing import Any, Dict, List, Optional

import httpx
from llama_index.core.bridge.pydantic import Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.openai.utils import create_retry_decorator
from openai import AsyncOpenAI, OpenAI

embedding_retry_decorator = create_retry_decorator(
    max_retries=6,
    random_exponential=True,
    stop_after_delay_seconds=60,
    min_seconds=1,
    max_seconds=20,
)


@embedding_retry_decorator
def get_embedding(
    client: OpenAI, text: str, engine: str, prompt: str, **kwargs: Any
) -> List[float]:
    """Get embedding.

    NOTE: Copied from OpenAI's embedding utils:
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies
    like matplotlib, plotly, scipy, sklearn.

    """
    text = f"{prompt} {text.replace("\n", " ")}" if prompt else text.replace("\n", " ")

    return client.embeddings.create(input=[text], model=engine, **kwargs).data[0].embedding


@embedding_retry_decorator
async def aget_embedding(
    aclient: AsyncOpenAI, text: str, engine: str, prompt: str, **kwargs: Any
) -> List[float]:
    """Asynchronously get embedding.

    NOTE: Copied from OpenAI's embedding utils:
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies
    like matplotlib, plotly, scipy, sklearn.

    """
    text = f"{prompt} {text.replace("\n", " ")}" if prompt else text.replace("\n", " ")

    return (await aclient.embeddings.create(input=[text], model=engine, **kwargs)).data[0].embedding


@embedding_retry_decorator
def get_embeddings(
    client: OpenAI, list_of_text: List[str], engine: str, prompt: str, **kwargs: Any
) -> List[List[float]]:
    """Get embeddings.

    NOTE: Copied from OpenAI's embedding utils:
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies
    like matplotlib, plotly, scipy, sklearn.

    """
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    list_of_text = [
        f"{prompt} {text.replace("\n", " ")}" if prompt else text.replace("\n", " ")
        for text in list_of_text
    ]

    data = client.embeddings.create(input=list_of_text, model=engine, **kwargs).data
    return [d.embedding for d in data]


@embedding_retry_decorator
async def aget_embeddings(
    aclient: AsyncOpenAI,
    list_of_text: List[str],
    engine: str,
    prompt: str,
    **kwargs: Any,
) -> List[List[float]]:
    """Asynchronously get embeddings.

    NOTE: Copied from OpenAI's embedding utils:
    https://github.com/openai/openai-python/blob/main/openai/embeddings_utils.py

    Copied here to avoid importing unnecessary dependencies
    like matplotlib, plotly, scipy, sklearn.

    """
    assert len(list_of_text) <= 2048, "The batch size should not be larger than 2048."

    list_of_text = [
        f"{prompt} {text.replace("\n", " ")}" if prompt else text.replace("\n", " ")
        for text in list_of_text
    ]

    data = (await aclient.embeddings.create(input=list_of_text, model=engine, **kwargs)).data
    return [d.embedding for d in data]


class OpenAILikeEmbedding(OpenAIEmbedding):
    query_instruction: Optional[str] = Field(
        default=None, description="Instruction to prepend to query text"
    )
    text_instruction: Optional[str] = Field(
        default=None, description="Instruction to prepend to text"
    )

    def __init__(
        self,
        model_name: str,
        embed_batch_size: int = 100,
        dimensions: Optional[int] = None,
        additional_kwargs: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        api_version: Optional[str] = None,
        max_retries: int = 10,
        timeout: float = 60.0,
        query_instruction: Optional[str] = None,
        text_instruction: Optional[str] = None,
        reuse_client: bool = True,
        callback_manager: Optional[CallbackManager] = None,
        default_headers: Optional[Dict[str, str]] = None,
        http_client: Optional[httpx.Client] = None,
        async_http_client: Optional[httpx.AsyncClient] = None,
        num_workers: Optional[int] = None,
        **kwargs: Any,
    ) -> None:

        super().__init__(
            embed_batch_size=embed_batch_size,
            dimensions=dimensions,
            callback_manager=callback_manager,
            model_name=model_name,
            additional_kwargs=additional_kwargs,
            api_key=api_key,
            api_base=api_base,
            api_version=api_version,
            max_retries=max_retries,
            reuse_client=reuse_client,
            timeout=timeout,
            default_headers=default_headers,
            http_client=http_client,
            async_http_client=async_http_client,
            num_workers=num_workers,
            **kwargs,
        )

    @classmethod
    def class_name(cls) -> str:
        return "OpenAILikeEmbedding"

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        client = self._get_client()
        return get_embedding(
            client,
            query,
            engine=self._query_engine,
            prompt=self.query_instruction,
            **self.additional_kwargs,
        )

    async def _aget_query_embedding(self, query: str) -> List[float]:
        """The asynchronous version of _get_query_embedding."""
        aclient = self._get_aclient()
        return await aget_embedding(
            aclient,
            query,
            engine=self._query_engine,
            prompt=self.query_instruction,
            **self.additional_kwargs,
        )

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        client = self._get_client()
        return get_embedding(
            client,
            text,
            engine=self._text_engine,
            prompt=self.text_instruction,
            **self.additional_kwargs,
        )

    async def _aget_text_embedding(self, text: str) -> List[float]:
        """Asynchronously get text embedding."""
        aclient = self._get_aclient()
        return await aget_embedding(
            aclient,
            text,
            engine=self._text_engine,
            prompt=self.text_instruction,
            **self.additional_kwargs,
        )

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get text embeddings.

        By default, this is a wrapper around _get_text_embedding.
        Can be overridden for batch queries.

        """
        client = self._get_client()
        return get_embeddings(
            client,
            texts,
            engine=self._text_engine,
            prompt=self.text_instruction,
            **self.additional_kwargs,
        )

    async def _aget_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously get text embeddings."""
        aclient = self._get_aclient()
        return await aget_embeddings(
            aclient,
            texts,
            engine=self._text_engine,
            prompt=self.text_instruction,
            **self.additional_kwargs,
        )
