from typing import Tuple

from llama_index.llms.openai_like import OpenAILike
from transformers import AutoTokenizer

from lassie.core.model_loader.base import BaseModelLoader
from lassie.core.model_loader.embeddings.openai_like import OpenAILikeEmbedding
from lassie.utils import get_logger

logger = get_logger(__name__)


# OpenAILike繼承自OpenAI的介面，因OpenAI API目前沒有使用Beam Search作為生成機制
# 因此沒辦法直接設定num_beams來使用Beam Search
GENERATION_CONFIG = {
    "basic_greedy": {},
    "basic_sample": {
        "temperature": 0.2,
        "top_p": 0.7,
    },
}


class OpenAILikeModelLoader(BaseModelLoader):
    def load_llm(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        context_window: int = 8192,
        max_new_tokens: int = 500,
        generation_config: str = "basic_greedy",
        tokenizer_name: str = None,
        tokenizer_kwargs: dict = {},
        hf_token: str = None,
        is_chat_model: bool = False,
        **kwargs,
    ) -> None:
        logger.info(f"loading the model and tokenizer of {model_name}...")
        if tokenizer_name is not None:
            # 在OpenAILike中tokenizer使用model name的起始方式不支援放入tokenizer相關參數，
            # 無法使用需要權限的tokenizer
            default_tokenizer_kwargs = {
                "token": hf_token,
                "padding_side": "left",
                "max_length": context_window,
            }
            default_tokenizer_kwargs.update(tokenizer_kwargs)
            llm_tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, **default_tokenizer_kwargs
            )

        model = OpenAILike(
            model=model_name,
            api_base=base_url,
            api_key=api_key,
            context_window=context_window,
            max_tokens=max_new_tokens,
            additional_kwargs=GENERATION_CONFIG[generation_config],
            tokenizer=llm_tokenizer,
            is_chat_model=is_chat_model,
            **kwargs,
        )
        return model, model.tokenizer

    def load_embed_model(
        self,
        model_name: str,
        base_url: str,
        api_key: str,
        prefix_tokens: Tuple[str, str] = None,
        tokenizer_name: str = None,
        **kwargs,
    ):
        if prefix_tokens:
            logger.info(
                f"{prefix_tokens[0]} and {prefix_tokens[1]} will be used as the query and text instruction, respectively."
            )
        logger.info(f"loading embedding model {model_name}...")
        embed_model = OpenAILikeEmbedding(
            model_name=model_name,
            api_base=base_url,
            api_key=api_key,
            query_instruction=prefix_tokens[0],
            text_instruction=prefix_tokens[1],
            **kwargs,
        )

        if tokenizer_name is not None:
            logger.info("loading the tokenizer of embedding model...")
            embed_model_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            logger.info(
                "The tokenizer_name is not provided, the tokenizer of the embedding model will not be initialized..."
            )
            embed_model_tokenizer = None

        return embed_model, embed_model_tokenizer
