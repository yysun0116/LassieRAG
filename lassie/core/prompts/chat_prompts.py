from llama_index.core import ChatPromptTemplate
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from lassie.core.prompts.default_prompts import DEFAULT_SYSTEM_PROMPT, MULTIQA_ZERO_SHOT_TMPL

# Multi QA
CHAT_MULTIQA_ZERO_SHOT_NO_SYS_ROLE_PROMPT = ChatPromptTemplate(
    message_templates=[
        ChatMessage(
            role=MessageRole.USER, content=f"{DEFAULT_SYSTEM_PROMPT}\n{MULTIQA_ZERO_SHOT_TMPL}"
        ),
    ]
)

CHAT_MULTIQA_ZERO_SHOT_PROMPT = ChatPromptTemplate(
    message_templates=[
        ChatMessage(role=MessageRole.SYSTEM, content=DEFAULT_SYSTEM_PROMPT),
        ChatMessage(role=MessageRole.USER, content=MULTIQA_ZERO_SHOT_TMPL),
    ]
)
