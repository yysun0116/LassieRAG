import dotenv

dotenv.load_dotenv(override=True)

import argparse
import functools
import json
import sys
import os
import time
from copy import deepcopy
from itertools import product

import yaml

from lassie.core.evaluation.runner import RAGEvaluationRunner
from lassie.core.query_engine import QueryEngineBuilder
from lassie.utils import get_logger

logger = get_logger(__name__)
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_path", type=str, default=f"{os.getcwd()}/rag_config.yml"
)
parser.add_argument(
    "--chunker_test_param",
    type=json.loads,
    default=(
        "{"
        '"sentence": {"chunk_size": [100, 300, 500], "chunk_overlap": [0, 20, 40]}, '
        '"recursive": {"chunk_size": [100, 300, 500], "chunk_overlap": [0, 20, 40]}, '
        '"sentence_window": {"window_size": [1, 2]}, '
        '"semantic": {"buffer_size": [1], "breakpoint_percentile_threshold": [10, 30, 50, 70, 90]}'
        "}"
    ),
)
parser.add_argument(
    "--data_source_path",
    type=json.loads,
    default=(
        "{"
        '"fps": "/dataset/fps/source_files",'
        '"etc": "/dataset/etc/source_files",'
        '"gss": "/dataset/gss/source_files"'
        "}"
    ),
)
parser.add_argument(
    "--eval_dataset_path",
    type=json.loads,
    default='{"fps": "fps_qa.json","etc": "etc_qa.json","gss": "gss_qa.json"}',
)
parser.add_argument(
    "--retrieval_topk",
    nargs="+",
    type=int,
    default=[5],
)
parser.add_argument(
    "--output_dir", type=str, default="/output"
)
parser.add_argument("--num_worker", type=int, default=6)
parser.add_argument("--is_ft_model", type=bool, default=False)

args = parser.parse_args()
for arg in vars(args):
    print(f"{arg: <18}: {getattr(args, arg)}")


chunker_param_map = {
    "sentence": ["chunk_size", "chunk_overlap"],
    "recursive": ["chunk_size", "chunk_overlap"],
    "sentence_window": ["window_size"],
    "semantic": ["buffer_size", "breakpoint_percentile_threshold"],
}


# generate query engine config with different chunking methods and parameters
def generate_config(
    base_config: dict, data_source_path: dict, chunker_test_param: dict, retrieval_topk: list
) -> dict:
    method_param_combinations = []
    for method, param_set in chunker_test_param.items():
        param_set = {"similarity_top_k": retrieval_topk, **param_set}
        keys, values = zip(*param_set.items())
        combinations = [
            {"chunk_method": method, **dict(zip(keys, comb))} for comb in product(*values)
        ]
        method_param_combinations += combinations

    all_test_config = {}
    for dataset_name, source_path in data_source_path.items():
        for comb in method_param_combinations:
            index_name = f"{dataset_name}_{comb['chunk_method']}_{'_'.join([str(comb[key]) for key in comb.keys() if key not in ['chunk_method', 'similarity_top_k']])}"
            config_i = deepcopy(base_config)
            if "chunk_overlap" in comb:
                comb["chunk_overlap"] = round(comb["chunk_size"] * (comb["chunk_overlap"] / 100))

            config_i["Data"]["input_dir"] = source_path
            config_i["Preprocess"]["chunking"] = {
                key: value for key, value in comb.items() if key != "similarity_top_k"
            }
            config_i["Indexing"]["index_name"] = index_name
            config_i["Retriever"]["similarity_top_k"] = comb.get("similarity_top_k")
            all_test_config[f"{index_name}_k{comb.get("similarity_top_k")}"] = config_i
    return all_test_config


# time record decorator
class TimeLogger:
    def __init__(self, question_to_id: dict):
        self.query_time_records = {}
        self.question_to_id = question_to_id

    def async_time_log(self, stage_name):
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                query_str = args[0].query_str if args else kwargs["query"].query_str
                query_id = self.question_to_id[query_str]

                start_time = time.perf_counter()
                # logger.info(f"{stage_name} {query_id} start: {start_time}")
                result = await func(*args, **kwargs)
                end_time = time.perf_counter()
                # logger.info(f"{stage_name} {query_id} end: {end_time}")

                if query_id not in self.query_time_records:
                    self.query_time_records[query_id] = {}

                self.query_time_records[query_id][stage_name] = end_time - start_time
                return result

            return wrapper

        return decorator


def main():
    # create configs
    logger.info(f"Loading base config for RAG query engine from {args.config_path}...")
    with open(args.config_path, "r") as file:
        base_config = yaml.safe_load(file)
    logger.info(f"Generate testing configs from input chunking methods and parameters...")
    all_test_config = generate_config(
        base_config, args.data_source_path, args.chunker_test_param, args.retrieval_topk
    )

    # load evaluation QA dataset
    logger.info(f"Loading evaluation QA datasets of {", ".join(args.eval_dataset_path.keys())}")
    QA_dataset = {dataset_name: [] for dataset_name in args.eval_dataset_path.keys()}
    dataset_question_dict = {dataset_name: [] for dataset_name in args.eval_dataset_path.keys()}
    id_to_question = {}
    for dataset_name, dataset_dir in args.eval_dataset_path.items():
        QA_dataset[dataset_name] = []
        with open(dataset_dir, "r") as f:
            for line in f:
                QA_dataset[dataset_name].append(json.loads(line))
        ## get questions from QA dataset
        for i, d in enumerate(QA_dataset[dataset_name]):
            dataset_question_dict[dataset_name].append(d["question"])
            ## create id to question mapping (for time log)
            id_to_question.update({f"{dataset_name}_{i}": d["question"]})
    ## create question to id mapping (for time log)
    question_to_id = {v: k for k, v in id_to_question.items()}

    # initialize evaluation runner
    logger.info(f"Initialize evaluation runner with num_worker = {args.num_worker}...")
    eval_pipeline = RAGEvaluationRunner(num_worker=args.num_worker)

    # Experiments
    for config_name, test_config in all_test_config.items():
        logger.info("")
        # get questions
        dataset_name = [name for name in args.eval_dataset_path if name in config_name][0]
        queries = dataset_question_dict[dataset_name]

        # initialize query engine
        time_logger = TimeLogger(question_to_id)
        query_engine = QueryEngineBuilder.from_config(config=test_config)

        query_engine.aretrieve = time_logger.async_time_log("retrieval")(query_engine.aretrieve)
        query_engine._response_synthesizer.asynthesize = time_logger.async_time_log("generation")(
            query_engine._response_synthesizer.asynthesize
        )

        # mlflow log setting
        mlflow_logs = {
            "parameter_log": {
                "RAG_embed_model": test_config["RAG_models"]["Embedding_model"]["model_name"],
                "RAG_retrieval_method": test_config["Retriever"]["vector_store_query_mode"],
                "RAG_preprocess_chunking": test_config["Preprocess"]["chunking"]["chunk_method"],
                "RAG_model": test_config["RAG_models"]["LLM"]["model_name"],
            },
            "tag_log": {
                "dataset": dataset_name,
                "dataset_num": len(queries),
                "use_chat_template": test_config["RAG_models"]["LLM"]["is_chat_model"],
                "has_finetuned": args.is_ft_model,
            },
        }

        # run RAG E2E Evaluation
        rag_performance, rag_eval_by_metric = eval_pipeline.run(
            queries=queries,
            query_engine=query_engine,
            # record_to_mlflow=False,
            mlflow_run_name=test_config["Indexing"]["index_name"],
            mlflow_logs=mlflow_logs,
            output_dir=f"{args.output_dir}/{config_name}",
        )

        with open(f"{args.output_dir}/time_log.json", "w") as f:
            json.dump(time_logger.query_time_records, f)

    return


if __name__ == "__main__":
    main()
    sys.exit()
