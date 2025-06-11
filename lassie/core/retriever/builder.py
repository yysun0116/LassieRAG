import inspect
from copy import deepcopy

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.indices.base import BaseIndex
from llama_index.core.retrievers import VectorIndexRetriever

from lassie.core.model_loader import ModelLoader
from lassie.utils import get_logger

logger = get_logger(__name__)

RETRIEVER_OPTIONS = {
    "vector_index": VectorIndexRetriever,
}


class RetrieverBuilder(BaseModel):
    loaded_models: ModelLoader = Field(
        default=None, description="The model loader of LLM and embedding model used in retriever"
    )
    retriever_type: str = Field(
        default="vector_index",
        description="The type of Index that used for building up the retriever",
    )
    _retriever: BaseRetriever

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.retriever_type in RETRIEVER_OPTIONS:
            logger.info(f"The {self.retriever_type} retriever will be initialized")
            self._retriever = RETRIEVER_OPTIONS[self.retriever_type]
        else:
            raise ValueError(
                f"The {self.retriever_type} retriever has not implemented, please try one of the following types of {", ".join(RETRIEVER_OPTIONS)}..."
            )

    @classmethod
    def from_config(
        cls, loaded_models: ModelLoader, index: BaseIndex, config: dict
    ) -> BaseRetriever:
        if config is None or config == {}:
            ValueError("The config should be provided")

        config_copy = deepcopy(config)
        retriever_type = config_copy.pop("retriever_type", "vector_index")
        instance = cls(
            loaded_models=loaded_models,
            retriever_type=retriever_type,
        )

        return instance.build(index=index, **config_copy)

    def build(self, index: BaseIndex, **retriever_config) -> BaseRetriever:
        if "embed_model" in [
            param.name for param in inspect.signature(self._retriever.__init__).parameters.values()
        ]:
            return self._retriever(
                index=index, embed_model=self.loaded_models._embed_model, **retriever_config
            )
        else:
            return self._retriever(index=index, **retriever_config)
