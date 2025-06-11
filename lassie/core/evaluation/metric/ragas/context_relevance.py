import asyncio
import re
from typing import Dict, List, Optional

from Levenshtein import distance
from llama_index.core import Settings
from llama_index.core.bridge.pydantic import Field
from llama_index.core.evaluation.base import BaseEvaluator, EvaluationResult
from llama_index.core.llms.llm import LLM
from llama_index.core.prompts.mixin import PromptDictType

from lassie.core.evaluation.metric.ragas.prompt import CONTEXT_RELEVANCE_PROMPT
from lassie.core.evaluation.utils import SentenceSegmentation, retry_request_and_parse


class ContextRelevanceEvaluationResult(EvaluationResult):
    relevant_sentence: List[str] = Field(
        default=[],
        description="Sentences that are relevant to the query from the contexts",
    )
    precision: float = Field(
        default=0.0,
        description="The proportion of contexts that contain relevant sentences",
    )
    all_score: List[float] = Field(
        default=[], description="The coverage of relevant sentences in each context"
    )


class ContextRelevanceEvaluator(BaseEvaluator):
    def __init__(
        self,
        eval_model: Optional[LLM] = None,
        language: str = "zh",
        max_retries: int = 3,
    ) -> None:
        self.eval_model = eval_model or Settings.llm
        self.eval_prompt_template = CONTEXT_RELEVANCE_PROMPT
        self.max_retries = max_retries
        # parser setting
        self.parser_regex = r"([\s\S]*)\[\[([AB])\]\]([\s\S]*)\[\[Extracted sentences\]\]:([\s\S]*)"
        self.parser_output_keys = ["feedback", "judgment", "other_info", "extraction"]
        # senetence segmentor setting
        self.segmentor = SentenceSegmentation(language=language)

    def _get_prompts(self) -> str:
        return self.eval_prompt_template.template

    def _update_prompts(self, prompts: PromptDictType) -> None:
        if "eval_template" in prompts:
            self.eval_prompt_template = prompts["eval_template"]

    def _parser(self, eval_response: str) -> Dict[str, str]:
        answer_status = re.search(r"\[\[([AB])\]\]", eval_response)
        if answer_status:
            if answer_status.group(1) == "B":
                output_match = re.search(self.parser_regex, eval_response)
                if output_match:
                    eval_response_content = {
                        key: output_match.group(i + 1).strip("\n")
                        for i, key in enumerate(self.parser_output_keys)
                    }
            else:
                output_match = re.search(r"([\s\S]*)\[\[([AB])\]\]([\s\S]*)", eval_response)
                eval_response_content = {
                    key: output_match.group(i + 1).strip("\n")
                    for i, key in enumerate(self.parser_output_keys[:-1])
                }
                eval_response_content.update({"extraction": ""})
        else:
            eval_response_content = {}
        return eval_response_content

    def _score(self, contexts: List[str], eval_response_content: Dict[str, str]) -> dict:
        # sentence segmentation of extracted sentences
        if eval_response_content["extraction"] != "":
            segmented_rel_sentence = self.segmentor.segmentation(
                eval_response_content["extraction"]
            )
            segmented_rel_sentence = [sen.strip().strip('"`') for sen in segmented_rel_sentence]
        else:
            segmented_rel_sentence = []
        # sentence segmentation of contexts
        all_context_score = []
        n_hit = 0
        for context_i in contexts:
            segmented_context_sentence = self.segmentor.segmentation(context_i)
            segmented_context_sentence = [sen.strip() for sen in segmented_context_sentence]
            # intersection
            # intersect_sentence = set(segmented_context_sentence) & set(segmented_rel_sentence)
            # if intersect_sentence:
            #     score = round(len(intersect_sentence) / len(segmented_context_sentence), 3)
            #     n_hit += 1
            match = 0
            for sen1 in segmented_context_sentence:
                for sen2 in segmented_rel_sentence:
                    if distance(sen1, sen2) <= 2:
                        match += 1
                        break
            if match > 0:
                score = round(match / len(segmented_context_sentence), 3)
                n_hit += 1
            else:
                score = 0
            all_context_score.append(score)
        return {
            "all_score": all_context_score,
            "score": (
                round(sum(all_context_score) / n_hit, 3) if n_hit > 0 else sum(all_context_score)
            ),
            "precision": round(n_hit / len(all_context_score), 3),
            "extraction": segmented_rel_sentence,
        }

    async def aevaluate(
        self, query: str = None, contexts: List[str] = None, response: str = None
    ) -> EvaluationResult:
        await asyncio.sleep(0)
        if query is None or contexts is None:
            raise ValueError("Both query and contexts must be provided")

        # sentence extraction
        eval_response_content = await retry_request_and_parse(
            retry_fn=self.eval_model.apredict,
            retry_fn_kwargs={
                "prompt": self.eval_prompt_template,
                "query_str": query,
                "context_str": "\n".join(contexts),
            },
            parser_fn=self._parser,
            max_retries=self.max_retries,
            default_parsed_result={},
            retry_fn_name="sentence extraction",
        )

        score_and_extraction = self._score(
            contexts=contexts, eval_response_content=eval_response_content
        )
        return ContextRelevanceEvaluationResult(
            query=query,
            contexts=contexts,
            response=response,
            feedback=eval_response_content["feedback"],
            score=score_and_extraction["score"],
            relevant_sentence=score_and_extraction["extraction"],
            all_score=score_and_extraction["all_score"],
            precision=score_and_extraction["precision"],
        )
