import dotenv

dotenv.load_dotenv(override=True)

import json
import logging
import os
from functools import reduce
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import seaborn as sns
from llama_index.core.base.base_query_engine import BaseQueryEngine
from llama_index.core.base.response.schema import Response
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.evaluation.batch_runner import BatchEvalRunner
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI

from lassie.core.evaluation.metric.ragas.context_relevance import ContextRelevanceEvaluator
from lassie.core.evaluation.metric.ragas.faithfulness import FaithfulnessEvaluator
from lassie.core.evaluation.metric.ragas.response_relevance import ResponseRelevanceEvaluator
from lassie.core.evaluation.utils import E2EEvalMLFlowSetting, index_contents
from lassie.utils import get_logger

logger = get_logger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

DEFAULT_EVAL_MODEL = AzureOpenAI(
    model="gpt-4o",
    engine="gpt-4o",
    azure_deployment=os.getenv("AOAI_DEPLOY_NAME"),
    azure_endpoint=os.getenv("AOAI_API_ENDPOINT"),
    api_key=os.getenv("AOAI_API_KEY"),
    api_version=os.getenv("AOAI_API_VERSION"),
    max_tokens=4096,
    max_retries=3,
)

DEFAULT_EMBED_MODEL = AzureOpenAIEmbedding(
    model="text-embedding-3-small",
    azure_deployment=os.getenv("AOAI_EMBED_DEPLOY_NAME"),
    azure_endpoint=os.getenv("AOAI_API_ENDPOINT"),
    api_key=os.getenv("AOAI_API_KEY"),
    api_version=os.getenv("AOAI_API_VERSION"),
)

evaluator_setting = {
    "context_relevance": ContextRelevanceEvaluator,
    "faithfulness": FaithfulnessEvaluator,
    "response_relevance": ResponseRelevanceEvaluator,
}

default_evaluator_kwargs = {
    "context_relevance": {"eval_model": DEFAULT_EVAL_MODEL},
    "faithfulness": {"eval_model": DEFAULT_EVAL_MODEL},
    "response_relevance": {"eval_model": DEFAULT_EVAL_MODEL, "embed_model": DEFAULT_EMBED_MODEL},
}


class RAGEvaluationRunner(BaseModel):
    evaluator_kwargs: Dict[str, dict] = Field(
        default={"context_relevance": {}, "faithfulness": {}, "response_relevance": {}},
        description="The evaluators to be used in the evaluation, along with the kwargs for each evaluator.",
    )
    num_worker: int = Field(default=6, description="The number of workers used in the evaluation")
    show_progress: bool = Field(default=True, description="To show progess bar or not")
    eval_runner: BatchEvalRunner = Field(
        default=None, description="LlamaIndex batch evaluation runner"
    )
    eval_model_name: dict = Field(default=None, exclude=True)
    embed_model_name: dict = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.evaluator_kwargs = {
            metric: {**default_evaluator_kwargs[metric], **kwarg}
            for metric, kwarg in self.evaluator_kwargs.items()
        }
        # set up evaluators
        evaluator_used = {
            metric: evaluator_setting[metric](**kwarg)
            for metric, kwarg in self.evaluator_kwargs.items()
        }
        logger.info(f"Evaluators used in this pipeline: {list(evaluator_used.keys())}")
        # record evaluation model and embedding model used in each evaluator
        self.eval_model_name = {
            metric: kwarg["eval_model"].metadata.model_name
            for metric, kwarg in self.evaluator_kwargs.items()
            if "eval_model" in kwarg
        }
        self.embed_model_name = {
            metric: kwarg["embed_model"].to_dict()["model_name"]
            for metric, kwarg in self.evaluator_kwargs.items()
            if "embed_model" in kwarg
        }
        # set up evaluation runner
        self.eval_runner = BatchEvalRunner(
            evaluator_used, workers=self.num_worker, show_progress=self.show_progress
        )

    def _format_to_df(
        self, eval_result: Dict[str, List[EvaluationResult]], output_dir: str
    ) -> Tuple[pd.DataFrame, str]:
        metric_scores = [
            pd.DataFrame(
                {
                    "query": [d.query for d in eval_result[metric]],
                    "contexts": [index_contents(d.contexts) for d in eval_result[metric]],
                    "response": [d.response for d in eval_result[metric]],
                    metric: [d.score for d in eval_result[metric]],
                }
            )
            for metric in eval_result.keys()
        ]
        combined_metrics = reduce(
            lambda left, right: pd.merge(left, right, on=["query", "contexts", "response"]),
            metric_scores,
        )
        # save evaluation result
        save_path = os.path.join(output_dir, "rag_performance.csv")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        combined_metrics.to_csv(save_path, index=False)
        return combined_metrics, save_path

    def _metric_histogram(self, rag_performance: pd.DataFrame, output_dir: str) -> None:
        metrics = list(self.evaluator_kwargs.keys())
        metric_data = {"metric": [], "score": []}
        for m in metrics:
            metric_data["metric"] += [m] * len(rag_performance)
            metric_data["score"] += list(rag_performance[m])
        metric_data = pd.DataFrame(metric_data)

        metric_score_histogram = sns.displot(
            data=metric_data, x="score", col="metric", element="step"
        )
        # add statistics on plot
        for ax in metric_score_histogram.axes.flatten():
            m_data = metric_data[metric_data["metric"] == ax.get_title().split(" = ")[-1]]

            stats_text = (
                f"Max: {round(m_data['score'].max(), 2):.2f}\n"
                f"Min: {round(m_data['score'].min(), 2):.2f}\n"
                f"Mean: {round(m_data['score'].mean(), 2):.2f}\n"
                f"Std: {round(m_data['score'].std(), 2):.2f}"
            )
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"),
            )
        # save plot
        metric_score_histogram.savefig(
            os.path.join(output_dir, "RAG_performance_histogram.png"), dpi=300
        )
        return

    def run(
        self,
        queries: List[str],
        query_engine: BaseQueryEngine = None,
        response_objs: Optional[List[Response]] = None,
        contexts_list: Optional[List[List[str]]] = None,
        responses: Optional[List[str]] = None,
        output_dir: str = os.getcwd(),
        record_to_mlflow: bool = True,
        mlflow_run_name: str = None,
        mlflow_logs: Dict[str, dict] = {},
        return_mlflow_log=False,
    ) -> Union[
        Tuple[pd.DataFrame, Dict[str, List[dict]]],
        Tuple[pd.DataFrame, Dict[str, List[dict]], E2EEvalMLFlowSetting],
    ]:
        # check input
        check_input = [
            value for value in [query_engine, response_objs, responses] if value is not None
        ]
        if len(check_input) < 1:
            raise ValueError(
                "One of [query_engine, response_objs, (contexts_list, responses)] should be provided"
            )

        # Evaluation
        if responses is not None and contexts_list is not None:
            logger.info("queries, contexts, and responses will be used in the evaluation")
            eval_result_by_metric = self.eval_runner.evaluate_response_strs(
                queries=queries, response_strs=responses, contexts_list=contexts_list
            )
        elif response_objs is not None:
            logger.info("queries and response_objs will be used in the evaluation")
            eval_result_by_metric = self.eval_runner.evaluate_responses(
                queries=queries, responses=response_objs
            )
        elif query_engine is not None:
            logger.info("queries and query_engine will be used in the evaluation")
            eval_result_by_metric = self.eval_runner.evaluate_queries(
                queries=queries,
                query_engine=query_engine,
            )
        elif (responses is not None and contexts_list is None) or (
            responses is None and contexts_list is not None
        ):
            raise ValueError("Both responses and contexts should be provided in the evaluation.")

        # format result
        rag_performance, save_path = self._format_to_df(eval_result_by_metric, output_dir)
        # statistic & histogram
        self._metric_histogram(rag_performance, output_dir)
        # evaluation result by metric
        eval_result_by_metric = {
            metric: [eval_obj.__dict__ for eval_obj in eval_obj_list]
            for metric, eval_obj_list in eval_result_by_metric.items()
        }
        os.makedirs(os.path.dirname(f"{output_dir}/eval_result_by_metric.json"), exist_ok=True)
        with open(f"{output_dir}/eval_result_by_metric.json", "w") as f:
            json.dump(eval_result_by_metric, f, ensure_ascii=False)

        # mlflow recording
        if record_to_mlflow:
            mlflow_recorder = E2EEvalMLFlowSetting(**mlflow_logs)
            mlflow_recorder.tag_log["mlflow.runName"] = mlflow_run_name
            mlflow_recorder.parameter_log.update(
                {
                    "Eval_model": self.eval_model_name,
                    "Eval_embed_model": self.embed_model_name,
                }
            )
            mlflow_recorder.metric_log.update(
                {metric: rag_performance[metric].mean() for metric in self.evaluator_kwargs.keys()}
            )
            mlflow_recorder.file_log.update(
                {save_path: rag_performance, "rag_eval_by_metric.json": eval_result_by_metric}
            )
            mlflow_recorder.record()

            if return_mlflow_log:
                return rag_performance, eval_result_by_metric, mlflow_recorder

        return rag_performance, eval_result_by_metric
