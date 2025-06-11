from typing import List, Tuple, Union

import nest_asyncio
import requests
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.schema import Document, BaseNode
from llama_index.core.vector_stores.types import BasePydanticVectorStore
from llama_index.vector_stores.opensearch import OpensearchVectorClient, OpensearchVectorStore
from requests.auth import HTTPBasicAuth

from lassie.core.indices.base import BaseIndexLoader
from lassie.core.model_loader.factory import ModelLoader
from lassie.utils import get_logger

logger = get_logger(__name__)

SEARCH_PIPELINE_CONFIG = {
    "hybrid_search": {
        "description": "Post processor for hybrid search",
        "phase_results_processors": [
            {
                "normalization-processor": {
                    "normalization": {"technique": "min_max"},
                    "combination": {
                        "technique": "arithmetic_mean",
                        "parameters": {"weights": [0.3, 0.7]},
                    },
                }
            }
        ],
    }
}


class OSIndexLoader(BaseIndexLoader):
    _embed_model: BaseEmbedding

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.loaded_models is not None:
            self._embed_model = self.loaded_models._embed_model
        else:
            raise ValueError(
                "The loaded_models should be provided when using OpenSearch index loader"
            )

    @classmethod
    def from_config(
        cls,
        loaded_models: ModelLoader,
        config: dict,
        data_source: List[Union[Document, BaseNode]] = None,
        node_transformations: List[NodeParser] = None,
    ):
        if config is None or config == {}:
            ValueError("The config should be provided")
        instance = cls(loaded_models=loaded_models)
        return instance.run(
            node_transformations=node_transformations, data_source=data_source, **config
        )

    def check_search_pipeline(
        self,
        host: str,
        http_auth: Tuple[str, str],
        search_pipeline_name: str,
        search_pipeline_config: dict = None,
    ):
        username, password = http_auth
        check_pipeline_exist = requests.get(
            f"{host}/_search/pipeline/{search_pipeline_name}",
            auth=HTTPBasicAuth(username, password),
            verify=False,
        )
        if check_pipeline_exist.status_code == 200:
            logger.info(f"Pipeine <{search_pipeline_name}> exist!")
        else:
            if search_pipeline_name in SEARCH_PIPELINE_CONFIG:
                pipeline_config = SEARCH_PIPELINE_CONFIG[search_pipeline_name]
            elif search_pipeline_config is None:
                raise ValueError("search_pipeline_config should be provided...")
            else:
                pipeline_config = search_pipeline_config

            logger.info(f"Creating pipeline <{search_pipeline_name}>...")
            create_pipeline = requests.put(
                f"{host}/_search/pipeline/{search_pipeline_name}",
                json=pipeline_config,
                auth=HTTPBasicAuth(username, password),
                verify=False,
            )
            if create_pipeline.status_code == 200:
                logger.info(f"Pipeline <{search_pipeline_name}> successfully created!")

    def check_index(
        self,
        host: str,
        index_name: str,
        http_auth: Tuple[str, str],
    ) -> bool:
        check_index = requests.get(
            f"{host}/{index_name}",
            auth=HTTPBasicAuth(http_auth[0], http_auth[1]),
            verify=False,
        )
        if check_index.status_code == 200:
            logger.info(f"Index <{index_name}> exist! Loading index from vector database...")
            return True
        else:
            logger.info(f"Index <{index_name}> not exist!")
            return False

    def get_index(
        self,
        index_name: str,
        vector_store: BasePydanticVectorStore,
        data_source: List[Union[Document, BaseNode]],
        node_transformations: List[NodeParser],
        index_exist: bool,
    ) -> BaseIndex:
        if index_exist:
            index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self._embed_model)
        else:
            if data_source is None:
                raise ValueError("data_source should be provided...")
            elif isinstance(data_source[0], Document):
                logger.info(f"Creating index <{index_name}> from documents...")
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex.from_documents(
                    documents=data_source,
                    transformations=node_transformations,
                    storage_context=storage_context,
                    embed_model=self._embed_model,
                    use_async=True,
                )
            elif isinstance(data_source[0], BaseNode):
                logger.info(f"Creating index <{index_name}> from nodes...")
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                index = VectorStoreIndex(
                    nodes=data_source,
                    # transformations=node_transformations,
                    storage_context=storage_context,
                    embed_model=self._embed_model,
                    use_async=True,
                )
        return index

    def run(
        self,
        host: str = "http://localhost:9200",
        http_auth: Tuple[str, str] = ("admin", "admin"),
        index_name: str = "sample_index",
        data_source: List[Document] = None,
        node_transformations: List[NodeParser] = None,
        search_pipeline_name: str = None,
        search_pipeline_config: dict = None,
        **kwargs,
    ) -> VectorStoreIndex:
        nest_asyncio.apply()
        # check searching pipeline
        if search_pipeline_name is not None:
            self.check_search_pipeline(
                host, http_auth, search_pipeline_name, search_pipeline_config
            )

        # check index
        index_exist = self.check_index(host, index_name, http_auth)
        # initialize client
        embedding = self._embed_model.get_text_embedding("This is an example.")
        client = OpensearchVectorClient(
            endpoint=host,
            index=index_name,
            http_auth=http_auth,
            search_pipeline=search_pipeline_name,
            dim=len(embedding),
            embedding_field="embedding",
            text_field="content",
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            **kwargs,
        )
        # initialize vector store
        vector_store = OpensearchVectorStore(client)
        # load or create index
        index = self.get_index(
            index_name, vector_store, data_source, node_transformations, index_exist
        )
        return index
