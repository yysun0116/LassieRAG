from unittest.mock import AsyncMock

import pytest

from lassie.core.evaluation.utils import (
    SentenceSegmentation,
    check_json_and_content,
    index_contents,
    retry_request_and_parse,
)


def test_index_contents():
    contents = [
        "This is sentence 1.",
        "This is sentence 2.",
        "This is sentence 3.",
        "This is sentence 4.",
        "This is sentence 5.",
    ]
    indexed_contents = index_contents(contents)
    assert indexed_contents == (
        "[1] This is sentence 1.\n"
        "[2] This is sentence 2.\n"
        "[3] This is sentence 3.\n"
        "[4] This is sentence 4.\n"
        "[5] This is sentence 5."
    )


def test_segmentor():
    chunk_zh = "這是一個測試資料集的段落字串，這些語句即將被用來斷句。斷句完後，預設應該要輸出一個list，或是也可以選擇輸出編號好的句子字串。"
    sentence_segmentor = SentenceSegmentation(language="zh")
    # return list
    segmented_sentence_list = sentence_segmentor.segmentation(chunk_zh)
    # return string
    segmented_sentence_string = sentence_segmentor.segmentation(chunk_zh, to_string=True)
    assert isinstance(segmented_sentence_list, list)
    assert isinstance(segmented_sentence_string, str)


def test_check_json():
    key_type_map = {"key_1": str, "key_2": list, "key_3": int}
    # test json
    json_froms = {
        "valid_json": '{"key_1": "value_1", "key_2": ["value_2_1", "value_2_2"], "key_3": 1}',
        "markdown_json": '```json\n{\n"key_1": "value_1",\n"key_2": ["value_2_1", "value_2_2"],\n"key_3": 1\n}\n```',
        "invalid_json_v1": '{"key_1": "value_1", "key_2": ["value_2_1", "value_2_2", "key_3": 1}',
        "invalid_json_v2": '{"key_1": "value_1", "key_2": ["value_2_1", "value_2_2"] "key_3": 1',
    }

    checked_json = {}
    for test_form, json_string in json_froms.items():
        checked_json[test_form] = check_json_and_content(
            json_string,
            check_keys_type_mapping=key_type_map,
        )

    assert all(result == checked_json["valid_json"] for result in checked_json.values())
    assert all(
        isinstance(checked_json["valid_json"][key], value_type)
        for key, value_type in key_type_map.items()
    )
    # test jsonl
    jsonl_forms = {
        "valid_jsonl": '[{"key_1": "value_1_1", "key_2": ["value_2_1", "value_2_2"], "key_3": 1}, {"key_1": "value_1_2", "key_2": ["value_2_3", "value_2_4"], "key_3": 0}]',
        "markdown_jsonl": '```json\n[\n{\n"key_1": "value_1_1",\n"key_2": ["value_2_1", "value_2_2"],\n"key_3": 1},\n{\n"key_1": "value_1_2",\n"key_2": ["value_2_3", "value_2_4"],\n"key_3": 0}]\n```',
        "invalid_jsonl_v1": '[{"key_1": "value_1_1", "key_2": ["value_2_1", "value_2_2", "key_3": 1}, {"key_1": "value_1_2", "key_2": ["value_2_3", "value_2_4"], "key_3": 0}',
        "invalid_jsonl_v2": '[{"key_1": "value_1_1", "key_2": ["value_2_1", "value_2_2"], "key_3": 1}, {"key_1": "value_1_2" "key_2": ["value_2_3", "value_2_4"], "key_3": 0]',
    }
    checked_jsonl = {}
    for test_form, json_string in jsonl_forms.items():
        checked_jsonl[test_form] = check_json_and_content(
            json_string,
            check_keys_type_mapping=key_type_map,
            use_jsonl=True,
        )
    assert all(result == checked_jsonl["valid_jsonl"] for result in checked_jsonl.values())
    assert all(
        isinstance(line[key], value_type)
        for line in checked_jsonl["valid_jsonl"]
        for key, value_type in key_type_map.items()
    )


def mock_parser(response):
    if response:
        return {"result": response}
    else:
        return {}


@pytest.mark.asyncio
async def test_retry_request_and_parse():
    # success
    mock_async_fn = AsyncMock(side_effect=["response_1", "response_2", "response_3"])

    parsed_result_success = await retry_request_and_parse(
        retry_fn=mock_async_fn,
        parser_fn=mock_parser,
        max_retries=3,
        retry_fn_name="testing",
        default_parsed_result={},
    )
    assert parsed_result_success == {"result": "response_1"}
    assert mock_async_fn.call_count == 1
    # retry
    mock_async_fn = AsyncMock(side_effect=[None, "response_2", "response_3"])

    parsed_result_retry = await retry_request_and_parse(
        retry_fn=mock_async_fn,
        parser_fn=mock_parser,
        max_retries=3,
        retry_fn_name="testing",
        default_parsed_result={},
    )
    assert parsed_result_retry == {"result": "response_2"}
    assert mock_async_fn.call_count == 2

    # fail
    mock_async_fn = AsyncMock(side_effect=[None, None, None, "response_4"])
    parsed_result_fail = await retry_request_and_parse(
        retry_fn=mock_async_fn,
        parser_fn=mock_parser,
        max_retries=3,
        retry_fn_name="testing",
        default_parsed_result={},
    )
    assert parsed_result_fail == {}
    assert mock_async_fn.call_count == 3
