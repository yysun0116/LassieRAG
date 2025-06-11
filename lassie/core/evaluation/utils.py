import json
import re
from typing import Any, Awaitable, Callable, Dict, List, TypeVar, Union

import pysbd
from llama_index.core.bridge.pydantic import BaseModel, Field
from pysbd.segmenter import Segmenter

from lassie.utils import BaseMLFlowSetting, get_logger

logger = get_logger(__name__)
T = TypeVar("T")


def index_contents(contents_list: List[str]) -> str:
    indexed_contents = []
    for idx, state in enumerate(contents_list):
        indexed_contents.append(f"[{idx+1}] {state}")
    return "\n".join(indexed_contents)


class SentenceSegmentation(BaseModel):
    language: str = Field(default="zh", description="The language of the text to be segmented")
    segmenter: Segmenter = Field(default=None, description="The pysbd segmenter")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.segmenter = pysbd.Segmenter(language=self.language, clean=False)

    def segmentation(self, text: str, to_string: bool = False) -> Union[str, list]:
        sentences = self.segmenter.segment(text)
        if to_string:
            return index_contents(sentences)
        return sentences


def convert_dict_value_type(origin_dict: dict, expact_type_mapping: Dict[str, type]) -> dict:
    converted_dict = {}
    for key, value in origin_dict.items():
        if key in expact_type_mapping:
            expected_type = expact_type_mapping[key]
            if expected_type == str and isinstance(value, list) and len(value) == 1:
                converted_dict[key] = str(value[0])
            elif expected_type == int and isinstance(value, list) and len(value) == 1:
                converted_dict[key] = int(value[0])
            elif expected_type == bool and isinstance(value, list) and len(value) == 1:
                converted_dict[key] = bool(value[0])
            elif expected_type == List[str]:
                converted_dict[key] = value
            elif expected_type == List[int]:
                converted_dict[key] = [int(v) for v in value]
            elif expected_type == List[float]:
                converted_dict[key] = [float(v) for v in value]
            else:
                converted_dict[key] = value
        else:
            converted_dict[key] = value
    return converted_dict


def match_and_clean_json_content(invalid_json: str, check_keys: List[str]) -> dict:
    regex_patterns = [r"([\s\S]*)"] * (2 * (len(check_keys)))
    regex_patterns[0::2] = [re.escape(key) for key in check_keys]
    parse_regex = "".join(regex_patterns)

    content_match = re.search(parse_regex, invalid_json)
    if content_match:
        loaded_json = {
            key: [v.strip(' "\n') for v in value.strip(' \n{[:",]}').split(",")]
            for key, value in zip(
                check_keys,
                [content_match.group(idx + 1) for idx in range(len(check_keys))],
            )
        }
    else:
        logger.info("No matching values found in the input text...")
        loaded_json = None
    return loaded_json


def check_json_and_content(
    generated_json: str,
    check_keys_type_mapping: Dict[str, type],
    use_jsonl: bool = False,
) -> Union[dict, list]:
    # Parsed from markdown format json
    markdown_json_matched = re.search(r"```json([\s\S]*)```", generated_json)
    generated_json = markdown_json_matched.group(1) if markdown_json_matched else generated_json

    check_keys = list(check_keys_type_mapping.keys())
    try:
        # load json if valid
        loaded_json = json.loads(generated_json)
        if not use_jsonl:
            loaded_json = {key: loaded_json[key] for key in check_keys}
        else:
            loaded_json = [{key: line[key] for key in check_keys} for line in loaded_json]
    except:
        # parsing result from invalid JSON
        if not use_jsonl:
            loaded_json = match_and_clean_json_content(generated_json, check_keys)
        else:
            loaded_json = []
            json_lines = generated_json.split("}, {")
            for line in json_lines:
                loaded_json.append(match_and_clean_json_content(line, check_keys))
    # convert value type based on key-value type mapping
    loaded_json = (
        convert_dict_value_type(loaded_json, check_keys_type_mapping)
        if loaded_json and not use_jsonl
        else [convert_dict_value_type(line, check_keys_type_mapping) for line in loaded_json]
    )
    return loaded_json


async def retry_request_and_parse(
    retry_fn: Callable[..., Awaitable[Any]],
    parser_fn: Callable[..., T],
    retry_fn_kwargs: Dict[str, Any] = {},
    parser_kwargs: Dict[str, Any] = {},
    max_retries: int = 3,
    default_parsed_result: Any = None,
    retry_fn_name: str = "...",
) -> T:
    retries = 0
    parsed_result = default_parsed_result
    while parsed_result == default_parsed_result and retries < max_retries:
        # request
        fn_response = await retry_fn(**retry_fn_kwargs)
        # parsing result
        parsed_result = parser_fn(fn_response, **parser_kwargs)

        if parsed_result == default_parsed_result:
            retries += 1
            logger.info(f"Retrying {retry_fn_name}, attempt {retries}/{max_retries}")

    return parsed_result


class E2EEvalMLFlowSetting(BaseMLFlowSetting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.parameter_log = {
            "RAG_embed_model": None,
            "RAG_retrieval_method": None,
            "RAG_preprocess_chunking": None,
            "RAG_preprocess_rewrite": None,
            "RAG_postprocess_rerank": None,
            "RAG_postprocess_filter": None,
            "RAG_model": None,
            "Eval_model": None,
            "Eval_embed_model": None,
        }
        self.tag_log = {
            "mlflow.runName": None,
            "dataset": None,
            "dataset_num": None,
            "has_finetuned": False,
            "prompt_type": "MultiQA_zero_shot",
            "use_chat_template": False,
        }
        self.metric_log = {}

        for attr in ["parameter_log", "tag_log", "metric_log"]:
            if attr in kwargs:
                setattr(self, attr, {**getattr(self, attr), **kwargs[attr]})
