import asyncio
from unittest.mock import AsyncMock

import numpy as np
import pytest
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms.llm import LLM
from pytest_mock import MockFixture

from lassie.core.evaluation.metric.ragas.context_relevance import ContextRelevanceEvaluator
from lassie.core.evaluation.metric.ragas.faithfulness import FaithfulnessEvaluator
from lassie.core.evaluation.metric.ragas.response_relevance import ResponseRelevanceEvaluator


class MockLLM(LLM):
    async def apredict(self, prompt, **prompt_args):
        return ""


class MockEmbedModel(BaseEmbedding):
    async def aget_text_embedding(self, text):
        return []


def generate_random_unit_vector(dim: int):
    vec = np.random.randn(dim)
    return vec / np.linalg.norm(vec)


def generate_orthogonal_vector(vec):
    random_vec = np.random.randn(*vec.shape)
    orthogonal_vec = random_vec - np.dot(random_vec, vec) * vec
    return orthogonal_vec / np.linalg.norm(orthogonal_vec)


def generate_embedding_pair_with_cosine_similarity(
    dim: int, target_similarity: float, unit_vec: np.array = None
):
    if unit_vec is None:
        unit_vec = generate_random_unit_vector(dim)

    orthogonal_vec = generate_orthogonal_vector(unit_vec)

    theta = np.arccos(target_similarity)
    sim_vec = np.cos(theta) * unit_vec + np.sin(theta) * orthogonal_vec

    return list(unit_vec), list(sim_vec)


class TestContextRelevance:
    @classmethod
    def setup_class(
        cls,
    ):
        # input data
        cls.queries = ["測試問題之一", "測試問題之二", "測試問題之三"]
        cls.contexts = [
            [
                "這是檢索出來的第一筆結果。有包含一些可能相關的句子，與測試問題一最相關的句子是這句。",  # 2
                "這是檢索出來的第二筆結果。這個結果中沒有跟測試問題一相關的句子。",  # 2
            ],
            [
                "這是檢索出來的第一筆結果。這是一個比較長的段落，包含很多與測試問題二無關的內容，而且有和很多很多的冗言贅字。其中只有一些些稍微跟測試問題二相關的內容，比如這句。",  # 3
                "這是檢索出來的第二筆結果。這是跟測試問題二最貼近的檢索結果。幾乎每個句子都跟測試問題二相關。",  # 3
            ],
            [
                "這是檢索出來的第一筆結果。沒有任何一句話跟測試問題三相關。",  # 2
                "這是檢索出來的第二筆結果。沒有任何一句話跟測試問題三相關。",  # 2
            ],
        ]
        cls.responses = ["測試回應之一", "測試回應之二", "測試回應之三"]
        # output
        cls.eval_model_responses = [
            "some reasons 1 [[B]] [[Extracted sentences]]:\n有包含一些可能相關的句子，與測試問題一最相關的句子是這句。",  # 1/2, 0/2 -> 0.5
            "some reasons 2 [[B]] [[Extracted sentences]]:\n其中只有一些些稍微跟測試問題二相關的內容，比如這句。\n這是跟測試問題二最貼近的檢索結果。幾乎每個句子都跟測試問題二相關。",  # 1/3, 2/3 -> 0.5
            "some reasons 3 [[A]] ",  # 0/2, 0/2 -> 0
        ]

    @pytest.mark.asyncio
    async def test_aevaluate(self, mocker: MockFixture):
        mock_LLM = mocker.Mock(spec=MockLLM)
        mock_LLM.apredict = AsyncMock(
            side_effect=lambda **prompt_args: self.eval_model_responses[
                self.queries.index(prompt_args.get("query_str"))
            ]
        )
        evaluator = ContextRelevanceEvaluator(eval_model=mock_LLM)
        eval_tasks = []
        for query, context, response in zip(self.queries, self.contexts, self.responses):
            print(f"Q: {query}, C: {context}, R: {response}")
            eval_tasks.append(evaluator.aevaluate(query=query, contexts=context, response=response))
        eval_results = await asyncio.gather(*eval_tasks)
        print(eval_results)

        assert all(
            eval_results[i].score == avg_coverage for i, avg_coverage in enumerate([0.5, 0.5, 0])
        )
        assert all(
            eval_results[i].all_score == coverage_per_context
            for i, coverage_per_context in enumerate([[0.5, 0], [0.333, 0.667], [0, 0]])
        )
        assert all(
            eval_results[i].precision == precision for i, precision in enumerate([0.5, 1, 0])
        )


class TestFaithfulness:
    @classmethod
    def setup_class(
        cls,
    ):
        # input data
        cls.queries = ["測試問題之一", "測試問題之二", "測試問題之三"]
        cls.contexts = [
            [
                "這是測試問題之一檢索出來的第一筆結果。",
                "這是測試問題之一檢索出來的第二筆結果。",
            ],
            [
                "這是測試問題之二檢索出來的第一筆結果。",
                "這是測試問題之二檢索出來的第二筆結果。",
            ],
            [
                "這是測試問題之三檢索出來的第一筆結果。",
                "這是測試問題之三檢索出來的第二筆結果。",
            ],
        ]
        cls.responses = ["測試回應之一。", "測試回應之二", "測試回應之三"]
        # output
        cls.eval_model_statements = [
            '{"statements": ["這是回應之一的第一個statement。", "這是回應之一的第二個statement。", "這是回應之一的第三個statement。"]}',
            '{"statements": ["這是回應之二的第一個statement。", "這是回應之二第二個statement。", "這是回應之二第三個statement。", "這是回應之二第四個statement。", "這是回應之二第五個statement。"]}',
            '{"statements": ["這是回應之三的第一個statement。", "這是回應之三的第二個statement。"]}',
        ]
        cls.eval_model_nli = [
            (
                '[{"statement": "這是回應之一的第一個statement。", "reason": "reason 1-1", "verdict": 1}, '  # 1
                '{"statement": "這是回應之一的第二個statement。", "reason": "reason 1-2", "verdict": 1}, '
                '{"statement": "這是回應之一的第三個statement。", "reason": "reason 1-3", "verdict": 1}]'
            ),
            (
                '[{"statement": "這是回應之二的第一個statement。", "reason": "reason 2-1", "verdict": 1}, '  # 0.4
                '{"statement": "這是回應之二第二個statement。", "reason": "reason 2-2", "verdict": 0}, '
                '{"statement": "這是回應之二第三個statement。", "reason": "reason 2-3", "verdict": 1}, '
                '{"statement": "這是回應之二第四個statement。", "reason": "reason 2-4", "verdict": 0}, '
                '{"statement": "這是回應之二第五個statement。", "reason": "reason 2-5", "verdict": 0}]'
            ),
            (
                '[{"statement": "這是回應之三的第一個statement。", "reason": "reason 3-1", "verdict": 1}, '  # 0.5
                '{"statement": "這是回應之三的第二個statement。", "reason": "reason 3-2", "verdict": 0}]'
            ),
        ]

    @pytest.mark.asyncio
    async def test_aevaluate(self, mocker: MockFixture):
        mock_LLM = mocker.Mock(spec=MockLLM)
        mock_LLM.apredict = AsyncMock(
            side_effect=lambda **prompt_args: (
                self.eval_model_statements[self.queries.index(prompt_args.get("query_str"))]
                if prompt_args.get("query_str")
                else self.eval_model_nli[
                    self.contexts.index(prompt_args.get("context_str").split("\n"))
                ]
            )
        )
        evaluator = FaithfulnessEvaluator(eval_model=mock_LLM)
        eval_tasks = []
        for query, context, response in zip(self.queries, self.contexts, self.responses):
            print(f"Q: {query}, C: {context}, R: {response}")
            eval_tasks.append(evaluator.aevaluate(query=query, contexts=context, response=response))
        eval_results = await asyncio.gather(*eval_tasks)
        print(eval_results)

        assert all(
            eval_results[i].score == faithfulness_score
            for i, faithfulness_score in enumerate([1, 0.4, 0.5])
        )
        assert all(
            eval_results[i].all_score == nli_results
            for i, nli_results in enumerate([[1, 1, 1], [1, 0, 1, 0, 0], [1, 0]])
        )


class TestResponseRelevance:
    @classmethod
    def setup_class(
        cls,
    ):
        # input data
        cls.queries = ["測試問題之一", "測試問題之二", "測試問題之三"]
        cls.contexts = [
            [
                "這是測試問題之一檢索出來的第一筆結果。",
                "這是測試問題之一檢索出來的第二筆結果。",
            ],
            [
                "這是測試問題之二檢索出來的第一筆結果。",
                "這是測試問題之二檢索出來的第二筆結果。",
            ],
            [
                "這是測試問題之三檢索出來的第一筆結果。",
                "這是測試問題之三檢索出來的第二筆結果。",
            ],
        ]
        cls.responses = ["測試回應之一。", "測試回應之二", "測試回應之三"]
        # output
        cls.eval_model_response = [
            '{"reason": "reason 1", "noncommittal": 0, "question": ["根據回應之一生成的問題一", "根據回應之一生成的問題二"]}',
            '{"reason": "reason 2", "noncommittal": 0, "question": ["根據回應之二生成的問題一", "根據回應之二生成的問題二", "根據回應之二生成的問題三"]}',
            '{"reason": "reason 3", "noncommittal": 1, "question": ["根據回應之三生成的問題一", "根據回應之三生成的問題二", "根據回應之三生成的問題三"]}',
        ]
        cls.query_embedding = generate_random_unit_vector(dim=5)

        cls.embeddings_similarity_to_query = {
            "根據回應之一生成的問題一": 0.8,
            "根據回應之一生成的問題二": 0.6,  # 0.7
            "根據回應之二生成的問題一": 0.5,
            "根據回應之二生成的問題二": 0.2,
            "根據回應之二生成的問題三": 0.2,  # 0.3
            "根據回應之三生成的問題一": 0.7,
            "根據回應之三生成的問題二": 0.6,
            "根據回應之三生成的問題三": 0.4,  # 0.567
        }

    @pytest.mark.asyncio
    async def test_aevaluate(self, mocker: MockFixture):
        mock_LLM = mocker.Mock(spec=MockLLM)
        mock_LLM.apredict = AsyncMock(
            side_effect=lambda **prompt_args: self.eval_model_response[
                self.responses.index(prompt_args.get("response_str"))
            ]
        )
        mock_embed_model = mocker.Mock(spec=MockEmbedModel)
        mock_embed_model.aget_text_embedding = AsyncMock(
            side_effect=lambda text: (
                list(self.query_embedding)
                if text in self.queries
                else generate_embedding_pair_with_cosine_similarity(
                    dim=5,
                    target_similarity=self.embeddings_similarity_to_query[text],
                    unit_vec=self.query_embedding,
                )[1]
            )
        )
        evaluator = ResponseRelevanceEvaluator(eval_model=mock_LLM, embed_model=mock_embed_model)
        eval_tasks = []
        for query, context, response in zip(self.queries, self.contexts, self.responses):
            print(f"Q: {query}, C: {context}, R: {response}")
            eval_tasks.append(evaluator.aevaluate(query=query, contexts=context, response=response))
        eval_results = await asyncio.gather(*eval_tasks)
        print(eval_results)

        assert all(
            eval_results[i].score == avg_similarity
            for i, avg_similarity in enumerate([0.7, 0.3, 0.0])
        )
        assert all(
            eval_results[i].all_score == all_similarity
            for i, all_similarity in enumerate([[0.8, 0.6], [0.5, 0.2, 0.2], [0.7, 0.6, 0.4]])
        )
        assert all(eval_results[i].noncommittal == verdict for i, verdict in enumerate([0, 0, 1]))
