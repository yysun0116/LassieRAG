from typing import List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.node_parser.interface import NodeParser

from lassie.core.model_loader import ModelLoader
from lassie.core.node_transformer.chunking import Chunker

PREPROCESS_OPTIONS = {
    "chunking": Chunker,
}


class PreProcessorBuilder(BaseModel):
    loaded_models: ModelLoader = Field(default=None, description="The models used in preprocessing")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_config(
        cls, loaded_models: ModelLoader = None, config: dict = None
    ) -> List[NodeParser]:
        instance = cls(loaded_models=loaded_models)
        if config is None or config == {}:
            return []
        else:
            preprocess_fns = []
            for processor_type in config:
                preprocessor_i = instance.build_processor(
                    processor_type=processor_type, **config[processor_type]
                )
                preprocess_fns.append(preprocessor_i)
            return preprocess_fns

    def build_processor(self, processor_type: str, **processor_kwargs) -> NodeParser:
        if processor_type in PREPROCESS_OPTIONS:
            preprocessor = PREPROCESS_OPTIONS[processor_type](loaded_models=self.loaded_models)
            return preprocessor.create(**processor_kwargs)
        else:
            raise ValueError(
                f"The postprocessor {processor_type} has not been implemented, please try one of the following: {", ".join(PREPROCESS_OPTIONS.keys())}"
            )
