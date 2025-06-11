from typing import Any, List

from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser.interface import NodeParser

from lassie.core.indices.base import BaseIndexLoader
from lassie.core.indices.opensearch import OSIndexLoader
from lassie.core.model_loader.factory import ModelLoader

DB_OPTIONS = {
    "opensearch": OSIndexLoader,
}


class IndexLoader(BaseModel):
    loaded_models: ModelLoader = Field(default=None, description="Model loader of embedding model")
    database_type: str = Field(
        default="opensearch", description="The database used for creating index"
    )
    _index_loader: BaseIndexLoader

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._index_loader = DB_OPTIONS[self.database_type](loaded_models=self.loaded_models)

    @classmethod
    def from_config(
        cls,
        loaded_models: ModelLoader,
        config: dict,
        data_source: Any = None,
        node_transformations: List[NodeParser] = None,
    ) -> BaseIndex:
        if config is None or config == {}:
            ValueError("The config should be provided")
        database_type = config.pop("database_type", "opensearch")
        instance = cls(
            loaded_models=loaded_models,
            database_type=database_type,
        )
        return instance.load(
            data_source=data_source, node_transformations=node_transformations, **config
        )

    def load(
        self, data_source: Any = None, node_transformations: List[NodeParser] = None, **config
    ):
        return self._index_loader.run(
            data_source=data_source, node_transformations=node_transformations, **config
        )
