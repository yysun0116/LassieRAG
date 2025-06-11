import asyncio
import json
import os
from typing import Dict, List, Optional

from llama_index.core import Settings
from llama_index.core.bridge.pydantic import Field
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptDictType

from lassie.core.evaluation.metric.ragas.prompt import NLI_PROMPT, STATEMENT_GENERATION_PROMPT
from lassie.core.evaluation.utils import (
    check_json_and_content,
    index_contents,
    retry_request_and_parse,
)

statement_gen_json_format = (
    '{"statements": ["extracted statement 1", "extracted statement 2", ...]}'
)
statements_example = json.dumps(
    {
        "statements": [
            "阿爾伯特·愛因斯坦是一位出生於德國的理論物理學家。",
            "阿爾伯特·愛因斯坦被公認為有史以來最偉大和最有影響力的物理學家之一。",
            "阿爾伯特·愛因斯坦在擔任威廉皇家物理研究所所長的期間提出了相對論。"
            "阿爾伯特·愛因斯坦最著名的成就是提出了相對論。",
            "阿爾伯特·愛因斯坦也對量子力學的發展做出了重要貢獻。",
        ]
    },
    ensure_ascii=False,
)

nli_example = json.dumps(
    [
        {
            "statement": "John主修生物學。",
            "reason": "John's major is explicitly mentioned as Computer Science. There is no information suggesting he is majoring in Biology.",
            "verdict": 0,
        },
        {
            "statement": "John正在修讀人工智慧課程。",
            "reason": "The context mentions the courses John is currently enrolled in, and Artificial Intelligence is not mentioned. Therefore, it cannot be deduced that John is taking a course on AI.",
            "verdict": 0,
        },
        {
            "statement": "John是一名專注的學生。",
            "reason": "The context states that he spends a significant amount of time studying and completing assignments. Additionally, it mentions that he often stays late in the library to work on his projects, which implies dedication.",
            "verdict": 1,
        },
        {
            "statement": "John有一份兼職工作。",
            "reason": "There is no information given in the context about John having a part-time job.",
            "verdict": 0,
        },
    ],
    ensure_ascii=False,
)
nli_json_format = '[{"statements": "statement 1", "reason:": "the reason of your verdict", "verdict": "your verdict (0 or 1)"}, ...]'


class FaithfulnessEvaluationResult(EvaluationResult):
    statements: str = Field(default="", description="The statements generated from response")
    all_score: List[int] = Field(
        default=[], description="The verdict of NLI in each statement from response"
    )


class FaithfulnessEvaluator(BaseEvaluator):
    def __init__(
        self,
        eval_model: Optional[LLM] = None,
        max_retries: int = 3,
    ) -> None:
        self.eval_model = eval_model if eval_model else Settings.llm
        self.statement_gen_prompt_template = STATEMENT_GENERATION_PROMPT
        self.nli_prompt_template = NLI_PROMPT
        self.max_retries = max_retries
        # parser setting
        self.parser_state_gen_content_types = {"statements": List[str]}
        self.parser_nli_content_types = {
            "statement": str,
            "reason": str,
            "verdict": int,
        }

    def _get_prompts(self) -> Dict[str, str]:
        return {
            "statement_generation": self.statement_gen_prompt_template.template,
            "nli": self.nli_prompt_template.template,
        }

    def _update_prompts(self, prompts: PromptDictType) -> None:
        if "eval_template" in prompts:
            self.statement_gen_prompt_template, self.nli_prompt_template = prompts["eval_template"]

    def _score(self, nli_result: List[dict]) -> dict:
        return {
            "all_score": [line["verdict"] for line in nli_result],
            "score": round(sum([state["verdict"] for state in nli_result]) / len(nli_result), 3),
        }

    async def aevaluate(
        self, query: str = None, contexts: List[str] = None, response: str = None
    ) -> EvaluationResult:
        await asyncio.sleep(0)
        if query is None or contexts is None or response is None:
            raise ValueError("Query, contexts and response must be provided")

        # Statements generation from response
        loaded_statements = await retry_request_and_parse(
            retry_fn=self.eval_model.apredict,
            retry_fn_kwargs={
                "prompt": self.statement_gen_prompt_template,
                "query_str": query,
                "response_str": response,
                "json_format_example": statement_gen_json_format,
                "response_example": statements_example,
            },
            parser_fn=check_json_and_content,
            parser_kwargs={"check_keys_type_mapping": self.parser_state_gen_content_types},
            max_retries=self.max_retries,
            default_parsed_result={},
            retry_fn_name="statements generation",
        )

        # NLI of statements
        nli_result = await retry_request_and_parse(
            retry_fn=self.eval_model.apredict,
            retry_fn_kwargs={
                "prompt": self.nli_prompt_template,
                "context_str": "\n".join(contexts),
                "response_statements": index_contents(loaded_statements["statements"]),
                "json_format_example": nli_json_format,
                "response_example": nli_example,
            },
            parser_fn=check_json_and_content,
            parser_kwargs={
                "use_jsonl": True,
                "check_keys_type_mapping": self.parser_nli_content_types,
            },
            max_retries=self.max_retries,
            default_parsed_result=[],
            retry_fn_name="NLI",
        )

        faithfulness_score = self._score(nli_result)
        return FaithfulnessEvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            statements=index_contents([line["statement"] for line in nli_result]),
            feedback=index_contents([line["reason"] for line in nli_result]),
            score=faithfulness_score["score"],
            all_score=faithfulness_score["all_score"],
        )
