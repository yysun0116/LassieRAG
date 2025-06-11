import asyncio
import json
from typing import List, Optional

import numpy as np
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import Field
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptDictType

from lassie.core.evaluation.metric.ragas.prompt import QUESTION_GENERATION_PROMPT
from lassie.core.evaluation.utils import (
    check_json_and_content,
    index_contents,
    retry_request_and_parse,
)

question_gen_json_format = '{"reason": "The explanation of the verdict of noncommital and the generated question", "noncommittal": "The answer is noncommittal or not (1 or 0)", "question": ["generated_question 1", "generated_question 2", ...]}'
question_gen_example = json.dumps(
    {
        "reason": "The answer is not noncommittal because it directly states that Albert Einstein was born in Germany. The question should be straightforward and directly related to this information.",
        "noncommittal": 0,
        "question": ["阿爾伯特·愛因斯坦出生於哪裡？", "阿爾伯特·愛因斯坦是哪裡人？"],
    },
    ensure_ascii=False,
)


class ResponseRelevanceEvaluationResult(EvaluationResult):
    generated_queries: str = Field(default="", description="The queries generated from response")
    all_score: List[float] = Field(
        default=[], description="The cosine similarities of generated queries and user query"
    )
    noncommittal: int = Field(
        default=0,
        description="Whether the response is ambiguous/evasive or not (1 for noncommittal; 0 for not noncommittal).",
    )


class ResponseRelevanceEvaluator(BaseEvaluator):
    def __init__(
        self,
        eval_model: Optional[LLM] = None,
        embed_model: Optional[BaseEmbedding] = None,
        max_retries: int = 3,
    ) -> None:
        self.eval_model = eval_model if eval_model else Settings.llm
        self.embed_model = embed_model if embed_model else Settings.embed_model
        self.eval_prompt_template = QUESTION_GENERATION_PROMPT
        self.max_retries = max_retries
        # parser setting
        self.parser_question_gen_content_types = {
            "reason": str,
            "noncommittal": int,
            "question": List[str],
        }

    def _get_prompts(self) -> str:
        return self.eval_prompt_template.template

    def _update_prompts(self, prompts: PromptDictType) -> None:
        if "eval_template" in prompts:
            self.eval_prompt_template = prompts["eval_template"]

    def _score(
        self, query_embedding: List[float], gen_questions_embbeddings: List[List[float]]
    ) -> dict:
        question_vec = np.asarray(query_embedding).reshape(1, -1)
        gen_question_vec = np.asarray(gen_questions_embbeddings).reshape(-1, len(query_embedding))
        dot_product = gen_question_vec @ question_vec.T
        norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(question_vec, axis=1)
        all_score = list(dot_product.squeeze() / norm)
        return {
            "score": round(sum(all_score) / len(gen_questions_embbeddings), 3),
            "all_score": [round(score, 3) for score in all_score],
        }

    async def aevaluate(
        self, query: str = None, contexts: List[str] = None, response: str = None
    ) -> EvaluationResult:
        await asyncio.sleep(0)
        if query is None or contexts is None or response is None:
            raise ValueError("Query, contexts and response must be provided")

        # Question generation from response
        loaded_questions = await retry_request_and_parse(
            retry_fn=self.eval_model.apredict,
            retry_fn_kwargs={
                "prompt": self.eval_prompt_template,
                "context_str": "\n".join(contexts),
                "response_str": response,
                "json_format_example": question_gen_json_format,
                "response_example": question_gen_example,
            },
            parser_fn=check_json_and_content,
            parser_kwargs={"check_keys_type_mapping": self.parser_question_gen_content_types},
            max_retries=self.max_retries,
            default_parsed_result={},
            retry_fn_name="question generation",
        )

        # get embeddings
        query_embedding = await self.embed_model.aget_text_embedding(query)
        generated_questions_embeddings = []
        for question in loaded_questions["question"]:
            q_emb = await self.embed_model.aget_text_embedding(question)
            generated_questions_embeddings.append(q_emb)

        # consine similarity
        response_relevance_score = self._score(query_embedding, generated_questions_embeddings)

        return ResponseRelevanceEvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            generated_queries=index_contents(loaded_questions["question"]),
            feedback=loaded_questions["reason"],
            noncommittal=loaded_questions["noncommittal"],
            score=(
                0.0 if loaded_questions["noncommittal"] == 1 else response_relevance_score["score"]
            ),
            all_score=response_relevance_score["all_score"],
        )
