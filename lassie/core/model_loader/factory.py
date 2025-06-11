from copy import deepcopy

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.utils import Tokenizer

from lassie.core.model_loader.base import BaseModelLoader
from lassie.core.model_loader.hf_loader import HFModelLoader
from lassie.core.model_loader.openai_like_loader import OpenAILikeModelLoader
from lassie.utils import get_logger

logger = get_logger(__name__)

MODEL_LOADERS = {
    "huggingface": HFModelLoader,
    "openailike": OpenAILikeModelLoader,
}


class ModelLoader(BaseModel):
    llm_source: str = Field(default="huggingface", description="The source for loading the LLM")
    embed_model_source: str = Field(
        default="huggingface", description="The source for loading the model"
    )
    _llm_loader: BaseModelLoader
    _embed_model_loader: BaseModelLoader
    _llm: LLM
    _embed_model: BaseEmbedding
    _llm_tokenizer: Tokenizer
    _embed_model_tokenizer: Tokenizer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.llm_source in MODEL_LOADERS and self.embed_model_source in MODEL_LOADERS:
            if self.llm_source == self.embed_model_source:
                self._llm_loader = self._embed_model_loader = MODEL_LOADERS[self.llm_source]()
            else:
                self._llm_loader = MODEL_LOADERS[self.llm_source]()
                self._embed_model_loader = MODEL_LOADERS[self.embed_model_source]()
        else:
            raise ValueError(
                f"The source <{self.llm_source}> or <{self.embed_model_source}> for loading the models for loading the models has not been implemented. Please try one of the following: <{", ".join(MODEL_LOADERS.keys())}>"
            )

    @classmethod
    def from_config(cls, config: dict = None):
        if config is None:
            raise ValueError("The config should be provided")

        config_copy = deepcopy(config)
        llm_config = config_copy.get("LLM", {})
        llm_source = llm_config.pop("source", "huggingface")

        embed_model_config = config_copy.get("Embedding_model", {})
        embed_model_source = embed_model_config.pop("source", "huggingface")

        instance = cls(llm_source=llm_source, embed_model_source=embed_model_source)
        if llm_config != {}:
            instance._llm, instance._llm_tokenizer = instance._llm_loader.load_llm(**llm_config)
        if embed_model_config != {}:
            instance._embed_model, instance._embed_model_tokenizer = (
                instance._embed_model_loader.load_embed_model(**embed_model_config)
            )
        return instance

    def load_llm(self, **kwargs):
        self._llm, self._llm_tokenizer = self._llm_loader.load_llm(**kwargs)
        return self._llm, self._llm_tokenizer

    def load_embedding_model(self, **kwargs):
        self._embed_model, self._embed_model_tokenizer = self._embed_model_loader.load_embed_model(
            **kwargs
        )
        return self._embed_model, self._embed_model_tokenizer
