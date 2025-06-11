from llama_index.core import ChatPromptTemplate
from llama_index.core.base.llms.base import BaseLLM
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.prompts.utils import is_chat_model


def is_system_role_available(llm: BaseLLM) -> bool:
    if is_chat_model(llm):
        chat_messages_with_system_role = ChatPromptTemplate(
            [
                ChatMessage(role=MessageRole.SYSTEM, content="This is a sample."),
                ChatMessage(role=MessageRole.USER, content="This is a sample."),
            ]
        )
        try:
            llm.messages_to_prompt(chat_messages_with_system_role.format_messages())
            return True
        except:
            return False
    else:
        return False
