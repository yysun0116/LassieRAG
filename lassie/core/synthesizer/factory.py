from typing import Optional

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.settings import Settings

from lassie.core.prompts.selectors import MULTI_QA_PROMPT_SEL
from lassie.core.synthesizer.multiqa import MultiQA


def get_custom_synthesizer(
    llm: Optional[LLM] = None,
    prompt_helper: Optional[PromptHelper] = None,
    text_qa_template: Optional[BasePromptTemplate] = None,
    response_mode: str = "multiqa",
    callback_manager: Optional[CallbackManager] = None,
    streaming: bool = False,
    **kwargs,
) -> BaseSynthesizer:
    """Get a response synthesizer."""
    text_qa_template = text_qa_template or MULTI_QA_PROMPT_SEL

    callback_manager = callback_manager or Settings.callback_manager
    llm = llm or Settings.llm
    prompt_helper = (
        prompt_helper
        or Settings._prompt_helper
        or PromptHelper.from_llm_metadata(
            llm.metadata,
        )
    )
    if response_mode == "multiqa":
        return MultiQA(
            llm=llm,
            callback_manager=callback_manager,
            prompt_helper=prompt_helper,
            text_qa_template=text_qa_template,
            streaming=streaming,
        )

    else:
        raise ValueError(f"Unknown mode: {response_mode}")
