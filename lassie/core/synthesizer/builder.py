from typing import Callable, Optional, Type

from llama_index.core import get_response_synthesizer
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.indices.prompt_helper import PromptHelper
from llama_index.core.llms import LLM
from llama_index.core.prompts import BasePromptTemplate
from llama_index.core.response_synthesizers.base import BaseSynthesizer
from llama_index.core.response_synthesizers.type import ResponseMode
from llama_index.core.types import BasePydanticProgram

from lassie.core.model_loader import ModelLoader
from lassie.core.synthesizer.factory import get_custom_synthesizer


class SynthesizerBuilder(BaseModel):
    loaded_models: ModelLoader = Field(
        default=None, description="The model loader of LLM used in synthesizer"
    )

    @classmethod
    def from_config(cls, loaded_models: ModelLoader, config: dict = {}) -> BaseSynthesizer:
        instance = cls(loaded_models=loaded_models)
        return instance.build(**config)

    def build(
        self,
        prompt_helper: Optional[PromptHelper] = None,
        text_qa_template: Optional[BasePromptTemplate] = None,
        refine_template: Optional[BasePromptTemplate] = None,
        summary_template: Optional[BasePromptTemplate] = None,
        simple_template: Optional[BasePromptTemplate] = None,
        response_mode: str = "multiqa",
        callback_manager: Optional[CallbackManager] = None,
        use_async: bool = False,
        streaming: bool = False,
        structured_answer_filtering: bool = False,
        output_cls: Optional[Type[BaseModel]] = None,
        program_factory: Optional[Callable[[BasePromptTemplate], BasePydanticProgram]] = None,
        verbose: bool = False,
    ) -> BaseSynthesizer:
        if response_mode in [
            getattr(ResponseMode, attr) for attr in dir(ResponseMode) if attr.isupper()
        ]:
            synthesizer = get_response_synthesizer(
                llm=self.loaded_models._llm,
                prompt_helper=prompt_helper,
                text_qa_template=text_qa_template,
                refine_template=refine_template,
                summary_template=summary_template,
                simple_template=simple_template,
                response_mode=response_mode,
                callback_manager=callback_manager,
                use_async=use_async,
                streaming=streaming,
                structured_answer_filtering=structured_answer_filtering,
                output_cls=output_cls,
                program_factory=program_factory,
                verbose=verbose,
            )
        else:
            synthesizer = get_custom_synthesizer(
                llm=self.loaded_models._llm,
                prompt_helper=prompt_helper,
                text_qa_template=text_qa_template,
                response_mode=response_mode,
                callback_manager=callback_manager,
                streaming=streaming,
            )
        return synthesizer
