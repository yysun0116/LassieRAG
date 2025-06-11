import logging
import sys
from typing import Any, Dict

import mlflow
from pydantic import BaseModel, Field


def get_logger(name):
    date_strftime_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(message)s",
        datefmt=date_strftime_format,
    )
    return logging.getLogger(name)


logger = get_logger(__name__)


class BaseMLFlowSetting(BaseModel):
    parameter_log: Dict[str, Any] = Field(
        default={}, description="The parametes that will be recorded to mlflow"
    )
    tag_log: Dict[str, Any] = Field(
        default={}, description="The tags that will be recorded to MLflow"
    )
    metric_log: Dict[str, Any] = Field(
        default={}, description="The metric that will be recorded to mlflow"
    )
    file_log: Dict[str, Any] = Field(
        default={}, description="The file that will be recorded to mlflow"
    )

    def record(self, mlflow_run_id=None, mlflow_experiment_id=None):
        with mlflow.start_run(run_id=mlflow_run_id, experiment_id=mlflow_experiment_id) as run:
            if self.tag_log:
                mlflow.set_tags(self.tag_log)

            # record metric
            if self.metric_log:
                for k in self.metric_log:
                    try:
                        float(self.metric_log[k])
                        mlflow.log_metric(k, self.metric_log[k])
                    except ValueError:
                        mlflow.log_param(k, self.metric_log[k])

            # record params
            if self.parameter_log:
                mlflow.log_params(self.parameter_log)
            # record file
            if self.file_log:
                for k in self.file_log:
                    if "json" in k or "yaml" in k:
                        mlflow.log_dict(self.file_log[k], k)
                    elif "txt" in k:
                        mlflow.log_text(self.file_log[k], k)
                    else:
                        mlflow.log_artifact(k)

            logger.info("Your experiment record at:")
            logger.info(
                f"{mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}"
            )
