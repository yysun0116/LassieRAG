from llama_index.core.prompts import SelectorPromptTemplate
from llama_index.core.prompts.utils import is_chat_model

from lassie.core.prompts.chat_prompts import (
    CHAT_MULTIQA_ZERO_SHOT_NO_SYS_ROLE_PROMPT,
    CHAT_MULTIQA_ZERO_SHOT_PROMPT,
)
from lassie.core.prompts.default_prompts import MULTIQA_ZERO_SHOT_PROMPT
from lassie.core.prompts.utils import is_system_role_available

# Multi QA
multi_qa_conditionals = [
    (
        is_system_role_available,
        CHAT_MULTIQA_ZERO_SHOT_PROMPT,
    ),  # chat model and system role available
    (is_chat_model, CHAT_MULTIQA_ZERO_SHOT_NO_SYS_ROLE_PROMPT),  # chat model but no system role
]

MULTI_QA_PROMPT_SEL = SelectorPromptTemplate(
    default_template=MULTIQA_ZERO_SHOT_PROMPT,
    conditionals=multi_qa_conditionals,
)
