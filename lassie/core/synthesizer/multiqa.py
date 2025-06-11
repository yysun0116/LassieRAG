from typing import Any, Optional, Sequence

import llama_index.core.instrumentation as instrument
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts.base import BasePromptTemplate
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.types import RESPONSE_TEXT_TYPE

from lassie.core.prompts.selectors import MULTI_QA_PROMPT_SEL
from lassie.utils import get_logger

dispatcher = instrument.get_dispatcher(__name__)
logger = get_logger(__name__)


class MultiQA(BaseSynthesizer):
    def __init__(
        self,
        llm: Optional[LLM] = None,
        callback_manager: Optional[CallbackManager] = None,
        prompt_helper: Optional[PromptHelper] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        streaming: bool = False,
    ) -> None:
        super().__init__(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            streaming=streaming,
        )
        self._text_qa_template = text_qa_template or MULTI_QA_PROMPT_SEL

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {
            "text_qa_template": self._text_qa_template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""
        if "text_qa_template" in prompts:
            self._text_qa_template = prompts["text_qa_template"]

    def _format_chunk_texts(self, query_str: str, chunk_texts: Sequence[str]):
        formatted_chunk = "\n".join(
            [f"Document [{idx+1}]: {chunk}" for idx, chunk in enumerate(chunk_texts)]
        )

        # check if the retrieved document size is available
        formatted_chunk_size = self._prompt_helper._token_counter.get_string_tokens(formatted_chunk)
        text_qa_template = self._text_qa_template.partial_format(query_str=query_str)
        available_chunk_size = self._prompt_helper._get_available_chunk_size(
            text_qa_template, padding=5, llm=self._llm
        )
        if formatted_chunk_size > available_chunk_size:
            logger.warning(
                "The retrieved document size exceeds the available context length, so some information may be truncated..."
            )
        return formatted_chunk

    async def aget_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Apply the same prompt to text chunks and return async responses."""
        formatted_chunk = self._format_chunk_texts(query_str=query_str, chunk_texts=text_chunks)

        if not self._streaming:
            return await self._llm.apredict(
                self._text_qa_template,
                query_str=query_str,
                context_str=formatted_chunk,
                **response_kwargs,
            )
        else:
            return self._llm.stream(
                self._text_qa_template,
                query_str=query_str,
                context_str=formatted_chunk,
                **response_kwargs,
            )

    def get_response(
        self,
        query_str: str,
        text_chunks: Sequence[str],
        **response_kwargs: Any,
    ) -> RESPONSE_TEXT_TYPE:
        """Apply the same prompt to text chunks and return responses."""
        formatted_chunk = self._format_chunk_texts(query_str=query_str, chunk_texts=text_chunks)

        if not self._streaming:
            return self._llm.predict(
                self._text_qa_template,
                query_str=query_str,
                context_str=formatted_chunk,
                **response_kwargs,
            )
        else:
            return self._llm.stream(
                self._text_qa_template,
                query_str=query_str,
                context_str=formatted_chunk,
                **response_kwargs,
            )
