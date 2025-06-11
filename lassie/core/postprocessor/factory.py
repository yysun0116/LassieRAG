from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.postprocessor import (
    MetadataReplacementPostProcessor,
    SentenceTransformerRerank,
)
from llama_index.core.postprocessor.types import BaseNodePostprocessor

from lassie.core.model_loader import ModelLoader

POSTPROCESS_OPTIONS = {
    "metadata_replace": MetadataReplacementPostProcessor,
    "rerank": SentenceTransformerRerank,
}


class PostProcessorBuilder(BaseModel):
    loaded_models: ModelLoader = Field(
        default=None, description="The models used in postprocessing"
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_config(
        cls, loaded_models: ModelLoader = None, config: dict = None
    ) -> List[BaseNodePostprocessor]:
        instance = cls(loaded_models=loaded_models)
        if config is None or config == {}:
            return []
        else:
            postprocess_fns = []
            for processor_type in config:
                postprocessor_i = instance.build_processor(
                    processor_type=processor_type, **config[processor_type]
                )
                postprocess_fns.append(postprocessor_i)
            return postprocess_fns

    def build_processor(self, processor_type: str, **processor_kwargs) -> BaseNodePostprocessor:
        if processor_type in POSTPROCESS_OPTIONS:
            return POSTPROCESS_OPTIONS[processor_type](**processor_kwargs)
        else:
            raise ValueError(
                f"The postprocessor {processor_type} has not been implemented, please try one of the following: {POSTPROCESS_OPTIONS.keys()}"
            )
