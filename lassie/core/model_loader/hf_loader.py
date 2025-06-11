from typing import Tuple

import torch
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import AutoTokenizer, BitsAndBytesConfig

from lassie.core.model_loader.base import BaseModelLoader
from lassie.utils import get_logger

logger = get_logger(__name__)

BNB_CONFIG = {
    "4bit": BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
}

GENERATION_CONFIG = {
    "basic_greedy": {
        "do_sample": False,
        "num_beams": 1,
    },
    "basic_beam": {
        "do_sample": False,
        "num_beams": 2,
    },
    "basic_sample": {
        "do_sample": True,
        "num_beams": 1,
        "temperature": 0.2,
        "top_p": 0.7,
        "num_return_sequences": 1,
    },
}

TORCH_DTYPE = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


class HFModelLoader(BaseModelLoader):
    def load_llm(
        self,
        model_name: str,
        model_kwargs: dict = {},
        tokenizer_kwargs: dict = {},
        hf_token: str = None,
        context_window: int = 8192,
        max_new_tokens: int = 500,
        dtype: str = "fp16",
        device_map: str = "auto",
        quantization_type: str = None,
        generation_config: str = "basic_greedy",
        is_chat_model: bool = False,
        **kwargs,
    ) -> None:
        default_model_kwargs = {
            "token": hf_token,
            "torch_dtype": TORCH_DTYPE[dtype],
            "trust_remote_code": True,
            "quantization_config": BNB_CONFIG[quantization_type] if quantization_type else None,
        }
        default_tokenizer_kwargs = {
            "token": hf_token,
            "padding_side": "left",
            "max_length": context_window,
        }
        default_model_kwargs.update(model_kwargs)
        default_tokenizer_kwargs.update(tokenizer_kwargs)

        logger.info(f"loading the model and tokenizer of {model_name}...")
        model = HuggingFaceLLM(
            model_name=model_name,
            model_kwargs=default_model_kwargs,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            device_map=device_map,
            generate_kwargs=GENERATION_CONFIG[generation_config],
            tokenizer_name=model_name,
            tokenizer_kwargs=default_tokenizer_kwargs,
            is_chat_model=is_chat_model,
        )
        return model, model._tokenizer

    def load_embed_model(
        self,
        model_name: str,
        max_length: int = 512,
        prefix_tokens: Tuple[str, str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        if prefix_tokens:
            logger.info(
                f"{prefix_tokens[0]} and {prefix_tokens[1]} will be used as the query and text instruction, respectively."
            )
        logger.info(f"loading embedding model {model_name}...")
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            max_length=max_length,
            query_instruction=prefix_tokens[0] if prefix_tokens else None,
            text_instruction=prefix_tokens[1] if prefix_tokens else None,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        logger.info("loading the tokenizer of embedding model...")
        embed_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return embed_model, embed_model_tokenizer
