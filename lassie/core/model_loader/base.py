from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.utils import Tokenizer

from lassie.utils import get_logger

logger = get_logger(__name__)


class BaseModelLoader(BaseModel):
    def load_llm(self):
        raise NotImplementedError("load_llm() should be implemented")

    def load_embed_model(self):
        raise NotImplementedError("load_embed_model() should be implemented")
