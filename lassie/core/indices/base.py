from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.indices.base import BaseIndex

from lassie.core.model_loader.factory import ModelLoader


class BaseIndexLoader(BaseModel):
    loaded_models: ModelLoader = Field(default=None, description="Model loader of embedding model")
    index_config: dict = Field(default={}, description="The config used for initializing index")

    @classmethod
    def from_config(cls):
        raise NotImplementedError("from_config() function should be implemented!")

    def run(self, **kwargs) -> BaseIndex:
        raise NotImplementedError("run() function should be implemented!")
