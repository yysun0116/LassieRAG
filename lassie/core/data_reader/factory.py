from copy import deepcopy

from llama_index.core import SimpleDirectoryReader
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.readers.base import BaseReader

from lassie.utils import get_logger

logger = get_logger(__name__)


READER_OPTIONS = {
    "simple_dir": SimpleDirectoryReader,
}


class DataReader(BaseModel):
    reader_type: str = Field(
        default="simple_dir", description="The reader used for loading the data source"
    )
    _reader: BaseReader

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.reader_type in READER_OPTIONS:
            logger.info(f"Using the {self.reader_type} reader...")
            self._reader = READER_OPTIONS[self.reader_type]
        else:
            raise ValueError(
                f"The {self.reader_type} reader has not been included, please try one of the following: {", ".join(READER_OPTIONS)}"
            )

    @classmethod
    def from_config(cls, config: dict = None):
        if config is None or config == {}:
            return None

        config_copy = deepcopy(config)
        reader_type = config_copy.pop("reader_type", "simple_dir")
        instance = cls(reader_type=reader_type)
        data_source = instance.load(**config_copy)
        return data_source

    def load(self, **kwargs):
        return self._reader(**kwargs).load_data()
