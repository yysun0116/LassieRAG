from llama_index.core.prompts.base import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant built by Galaxy Software Service (GSS). "
    "The user you are helping speaks Traditional Chinese and comes from Taiwan, so in most cases, you should respond in Traditional Chinese."
)

# Multi QA
MULTIQA_ZERO_SHOT_TMPL = (
    "Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results (some of which might be irrelevant) and cite them properly. "
    "Use an unbiased and journalistic tone. "
    "Always cite for any factual claim. "
    "When citing several search results, use [1][2][3]. "
    "Cite at least one document and at most three documents in each sentence. "
    "If multiple documents support the sentence, only cite a minimum sufficient subset of the documents.\n"
    "{context_str}\n"
    "Question: {query_str}"
)

MULTIQA_ZERO_SHOT_PROMPT = PromptTemplate(
    template=f"{DEFAULT_SYSTEM_PROMPT}\n{MULTIQA_ZERO_SHOT_TMPL}\nAnswer: ",
    prompt_type=PromptType.QUESTION_ANSWER,
)
