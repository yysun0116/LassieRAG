import asyncio
from typing import List
from unittest.mock import AsyncMock

import pandas as pd
import pytest
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from pytest_mock import MockFixture

TEST_LLM = AzureOpenAI(
    model="gpt-4o",
    azure_deployment="test_llm_deploy_name",
    azure_endpoint="test_api_endpoint",
    api_key="test_api_key",
    api_version="test_api_version",
)

TEST_EMBED_MODEL = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    azure_deployment="test_embed_deploy_name",
    azure_endpoint="test_api_endpoint",
    api_key="test_api_key",
    api_version="test_api_version",
)


class MockEvaluator(BaseEvaluator):
    def __init__(
        self,
        eval_model: LLM = None,
        embed_model: BaseEmbedding = None,
        mock_score: float = 0.2,
        mock_feedback: str = "test feedback",
    ) -> None:
        self.eval_model = eval_model
        self.embed_model = embed_model
        self._mock_score = mock_score
        self._mock_feedback = mock_feedback

    def _get_prompts(self) -> PromptDictType:
        """Get prompts."""
        return {}

    def _update_prompts(self, prompts: PromptDictType) -> None:
        """Update prompts."""

    async def aevaluate(
        self, query: str = None, contexts: List[str] = None, response: str = None
    ) -> EvaluationResult:
        return EvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            score=self._mock_score,
            feedback=self._mock_feedback,
        )


class MockQueryEngine(BaseQueryEngine):
    def __init__(self, callback_manager=None):
        super().__init__(callback_manager=callback_manager)

    def _get_prompt_modules(self):
        """Get prompt sub-modules."""
        return {}

    async def _aquery(self):
        await asyncio.sleep(0)
        return [""]

    def _query(self):
        return None

    async def aquery(self):
        await asyncio.sleep(0)
        return ""


@pytest.fixture(autouse=True)
def mock_env_variables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AOAI_DEPLOY_NAME", "test_llm_deploy_name")
    monkeypatch.setenv("AOAI_API_ENDPOINT", "test_api_endpoint")
    monkeypatch.setenv("AOAI_API_KEY", "test_api_key")
    monkeypatch.setenv("AOAI_API_VERSION", "test_api_version")
    monkeypatch.setenv("AOAI_EMBED_DEPLOY_NAME", "test_embed_deploy_name")


@pytest.fixture(autouse=True)
def mock_module_variables(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "lassie.core.evaluation.runner.evaluator_setting",
        {
            "test_metric_1": MockEvaluator,
            "test_metric_2": MockEvaluator,
        },
    )
    monkeypatch.setattr(
        "lassie.core.evaluation.runner.default_evaluator_kwargs",
        {
            "test_metric_1": {"eval_model": TEST_LLM},
            "test_metric_2": {"eval_model": TEST_LLM, "embed_model": TEST_EMBED_MODEL},
        },
    )


class TestBatchRunner:
    @classmethod
    def setup_class(
        cls,
    ):
        # runner kwargs
        cls.evaluator_kwargs = {
            "test_metric_1": {"mock_score": 0.25, "mock_feedback": "test feedback 1"},
            "test_metric_2": {"mock_score": 0.75, "mock_feedback": "test feedback 2"},
        }
        # input data
        cls.queries = ["test query 1", "test query 2"]
        cls.query_engine = MockQueryEngine()
        cls.contexts = [
            ["test chunk 1-1", "test chunk 1-2"],
            ["test chunk 2-1", "test chunk 2-2"],
        ]
        cls.responses = ["test response 1", "test response 2"]
        cls.response_objs = [
            Response(
                response=response,
                source_nodes=[NodeWithScore(node=TextNode(text=c), score=0.8) for c in context],
            )
            for response, context in zip(cls.responses, cls.contexts)
        ]

        # output result
        cls.evaluator_outputs = {
            metric: [
                EvaluationResult(
                    query=q,
                    contexts=c,
                    response=r,
                    score=metric_kwargs["mock_score"],
                    feedback=metric_kwargs["mock_feedback"],
                )
                for q, c, r in zip(cls.queries, cls.contexts, cls.responses)
            ]
            for metric, metric_kwargs in cls.evaluator_kwargs.items()
        }

    def test_format_to_df(self):
        from lassie.core.evaluation.runner import RAGEvaluationRunner

        eval_pipeline = RAGEvaluationRunner(evaluator_kwargs=self.evaluator_kwargs)
        combined_metrics, save_path = eval_pipeline._format_to_df(
            self.evaluator_outputs, output_dir="test_output_dir"
        )
        assert combined_metrics.equals(
            pd.DataFrame(
                {
                    "query": self.queries,
                    "contexts": [
                        "\n".join([f"[{i + 1}] {c}" for i, c in enumerate(context)])
                        for context in self.contexts
                    ],
                    "response": self.responses,
                    "test_metric_1": [0.25, 0.25],
                    "test_metric_2": [0.75, 0.75],
                }
            )
        )
        assert save_path == "test_output_dir/rag_performance.csv"

    def test_run_evaluation(self, mocker: MockFixture):
        from lassie.core.evaluation.runner import RAGEvaluationRunner

        eval_pipeline = RAGEvaluationRunner(evaluator_kwargs=self.evaluator_kwargs, num_worker=6)
        with mocker.patch(
            target="lassie.core.evaluation.utils.E2EEvalMLFlowSetting.record",
            return_value=None,
        ):
            # input set 1: queries, query_engine
            ## mock query_engine
            mock_query_engine = mocker.Mock(spec=MockQueryEngine)
            mock_query_engine.aquery = AsyncMock(
                side_effect=lambda query: self.response_objs[self.queries.index(query)]
            )
            rag_performance_1, eval_result_by_metric_1, mlflow_recorder_1 = eval_pipeline.run(
                queries=self.queries,
                query_engine=mock_query_engine,
                mlflow_run_name="test_query_and_query_engine",
                return_mlflow_log=True,
                output_dir="test_output_dir",
            )

            # input set 2: queries, response_objs
            rag_performance_2, eval_result_by_metric_2, mlflow_recorder_2 = eval_pipeline.run(
                queries=self.queries,
                response_objs=self.response_objs,
                mlflow_run_name="test_query_and_response_objs",
                return_mlflow_log=True,
                output_dir="test_output_dir",
            )

            # input set 3: queries, contexts, responses
            rag_performance_3, eval_result_by_metric_3, mlflow_recorder_3 = eval_pipeline.run(
                queries=self.queries,
                contexts_list=self.contexts,
                responses=self.responses,
                mlflow_run_name="test_query_context_and_response",
                return_mlflow_log=True,
                output_dir="test_output_dir",
            )

            assert rag_performance_1.equals(rag_performance_2) and rag_performance_2.equals(
                rag_performance_3
            )
            assert eval_result_by_metric_1 == eval_result_by_metric_2 == eval_result_by_metric_3
            assert mlflow_recorder_1.tag_log["mlflow.runName"] == "test_query_and_query_engine"
            assert mlflow_recorder_2.tag_log["mlflow.runName"] == "test_query_and_response_objs"
            assert mlflow_recorder_3.tag_log["mlflow.runName"] == "test_query_context_and_response"

            assert (
                mlflow_recorder_1.parameter_log["Eval_model"]
                == mlflow_recorder_2.parameter_log["Eval_model"]
                == mlflow_recorder_3.parameter_log["Eval_model"]
                == {"test_metric_1": "gpt-4o", "test_metric_2": "gpt-4o"}
            )
            assert (
                mlflow_recorder_1.parameter_log["Eval_embed_model"]
                == mlflow_recorder_2.parameter_log["Eval_embed_model"]
                == mlflow_recorder_3.parameter_log["Eval_embed_model"]
                == {"test_metric_2": "text-embedding-3-small"}
            )
