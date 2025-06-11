import re
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import (
    LangchainNodeParser,
    SemanticSplitterNodeParser,
    SentenceSplitter,
    SentenceWindowNodeParser,
)
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.utils import Tokenizer

from lassie.core.model_loader import ModelLoader
from lassie.utils import get_logger

CHUNK_METHODS = {"sentence", "sentence_window", "recursive", "semantic"}

logger = get_logger(__name__)


# sentence splitter
def chinese_sentence_splitter(text: str) -> List[str]:
    sentence_split_regex = r"[^。？！]+[；，。？！]?"
    return re.findall(sentence_split_regex, text)


class Chunker(BaseModel):
    loaded_models: ModelLoader = Field(
        default=None, description="Model loader of embedding model and its tokenizer"
    )
    chunk_config: dict = Field(default={}, description="The config used for initializing chunker")
    _embed_model: BaseEmbedding
    _embed_tokenizer: Tokenizer

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._embed_model = self.loaded_models._embed_model
        self._embed_tokenizer = self.loaded_models._embed_model_tokenizer

    @classmethod
    def from_config(cls, loaded_models: ModelLoader, chunk_config: dict) -> NodeParser:
        instance = cls(loaded_models=loaded_models)
        return instance.create(**chunk_config)

    def create(self, chunk_method: str, **kwargs) -> NodeParser:
        if chunk_method not in CHUNK_METHODS:
            raise ValueError(f"chunk_method should be one of {CHUNK_METHODS}")
        logger.info(f"Initializing {chunk_method} splitter...")
        if chunk_method == "sentence":
            chunker = SentenceSplitter(
                tokenizer=self._embed_tokenizer.encode if self._embed_tokenizer else None,
                secondary_chunking_regex="[^；，。？！]+[；，。？！]?",
                **kwargs,
            )
        elif chunk_method == "sentence_window":
            chunker = SentenceWindowNodeParser(
                sentence_splitter=chinese_sentence_splitter,
                window_metadata_key="window",
                original_text_metadata_key="original_sentence",
                **kwargs,
            )
        elif chunk_method == "recursive":
            if self._embed_model.class_name() == "HuggingFaceEmbedding":
                separators_setting = kwargs.pop(
                    "separators",
                    ["\n\n", "\n", "。", "？", "！", "，", ".", "?", "!", ",", " ", ""],
                )
                chunker = LangchainNodeParser(
                    RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
                        tokenizer=self._embed_tokenizer,
                        separators=separators_setting,
                        **kwargs,
                    )
                )
        elif chunk_method == "semantic":
            if self._embed_tokenizer is None:
                raise ValueError(
                    "Embedding model should be specified when using semantic splitter..."
                )
            chunker = SemanticSplitterNodeParser(
                embed_model=self._embed_tokenizer,
                sentence_splitter=chinese_sentence_splitter,
                **kwargs,
            )
        return chunker
