from copy import deepcopy
from typing import List

from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers.base import BaseSynthesizer

from lassie.core.data_reader import DataReader
from lassie.core.indices import IndexLoader
from lassie.core.model_loader import ModelLoader
from lassie.core.node_transformer import PreProcessorBuilder
from lassie.core.postprocessor import PostProcessorBuilder
from lassie.core.retriever import RetrieverBuilder
from lassie.core.synthesizer import SynthesizerBuilder


class QueryEngineBuilder(BaseModel):
    retriever: BaseRetriever = Field(default=None, description="An initialized retriever")
    synthesizer: BaseSynthesizer = Field(default=None, description="An initialized synthesizer")
    postprocessors: List[BaseNodePostprocessor] = Field(
        default=None, description="The post-processors used in query engine"
    )

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_config(cls, config: dict):
        rag_config = deepcopy(config)
        data_config = rag_config.get("Data", {})
        model_config = rag_config.get("RAG_models", {})
        preprocess_config = rag_config.get("Preprocess", {})
        index_config = rag_config.get("Indexing", {})
        retriever_config = rag_config.get("Retriever", {})
        postprocess_config = rag_config.get("Postprocess", {})
        synthesizer_config = rag_config.get("Synthesizer", {})

        data_source = DataReader.from_config(config=data_config)
        rag_models = ModelLoader.from_config(config=model_config)
        preprocessors = PreProcessorBuilder.from_config(
            loaded_models=rag_models, config=preprocess_config
        )
        index = IndexLoader.from_config(
            loaded_models=rag_models,
            data_source=data_source,
            node_transformations=preprocessors,
            config=index_config,
        )
        retriever = RetrieverBuilder.from_config(
            loaded_models=rag_models, index=index, config=retriever_config
        )
        postprocessors = PostProcessorBuilder.from_config(
            loaded_models=rag_models, config=postprocess_config
        )
        synthesizer = SynthesizerBuilder.from_config(
            loaded_models=rag_models, config=synthesizer_config
        )

        instance = cls(retriever=retriever, synthesizer=synthesizer, postprocessors=postprocessors)

        return instance.build()

    def build(self):
        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=self.synthesizer,
            node_postprocessors=self.postprocessors,
        )
        return query_engine
