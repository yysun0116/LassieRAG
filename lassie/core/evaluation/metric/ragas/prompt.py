from llama_index.core import PromptTemplate

CONTEXT_RELEVANCE_PROMPT = PromptTemplate(
    "[System] Please act as an evidence sentence extractor and extract the relevance sentences from the provided Context that can be used to answer the given Question. Your extraction should strictly follow the following rules:\n"
    "1. Begin your extraction by examing whether the Question could be answered using the given Context.\n"
    "2. Extract as many relevant sentences as needed to fully answer the Question.\n"
    "3. Extract the sentences that are relevant to the Question as completely as possible, ensuring you capture all necessary details.\n"
    "4. Include all sentences, sections, and subsections that contain information which can be used to judge or answer the Question.\n"
    "5. While extracting candidate sentences you're not allowed to make any changes to sentences from given context.\n"
    "6. If no relevant sentences are found, or if you believe the question cannot be answered from the given context, return [[A]]; otherwise, return [[B]].\n"
    "Please provide your explanation first, then the judgment of whether relevant sentences are found or not ([[A]] or [[B]]), and finally output your extraction after the key '[[Extracted sentences]]:'\n"
    "[User Query] {query_str}\n"
    "[Context] {context_str}"
)

STATEMENT_GENERATION_PROMPT = PromptTemplate(
    "[System] Given a Query and the Response. Please act as a statement extractor and extract the statement from the provided Response. Your extraction should strictly follow the following rules:\n"
    "1. Begin your extraction by splitting the Response into independent sentences.\n"
    "2. Break down each sentence into one or more fully understandable statements, ensuring no pronouns are used in each statement.\n"
    "3. Replace pronouns with the appropriate nouns to ensure clarity.\n"
    "4. Keep only one of the redundant statements if they have the same meaning.\n"
    "5. The generated statements cannot be questions.\n"
    "6. Ensure each extracted statement maintains the context and meaning of the original Answer.\n"
    "7. Format the extracted statement with JSON format as follows: {json_format_example}\n\n"
    "Examples:\n"
    "[User Query] 阿爾伯特·愛因斯坦是誰，他最著名的成就是什麼？\n"
    "[Response]\n他是一位出生於德國的理論物理學家，是個廣為人知的有史以來最偉大且最有影響力的物理學家之一。他在1914年至1932年期間在威廉皇家物理研究所所長。在這期間他提出了相對論，是他最著名的成就。除此之外他也對量子力學的發展做出了重要貢獻。\n"
    "[Statements]\n{response_example}\n\n"
    "Your actual task:\n"
    "[User Query] {query_str}\n"
    "[Response]\n{response_str}\n"
)

NLI_PROMPT = PromptTemplate(
    "[System] Your task is to judge the faithfulness of a series of statements based on a given context. For each statement you must return the reason first, and then return verdict as 1 if the statement can be directly inferred based on the context or 0 if the statement can not be directly inferred based on the context. Format the outputs in JSON as follows: {json_format_example}.\n\n"
    "Examples:\n"
    "[Context]\n"
    "John是XYZ大學的學生，他正在攻讀資訊工程學系。本學期他選修了幾門資工課程，包括資料結構、演算法等等。John 是一名勤奮的學生，總是花大量時間學習和完成作業，並經常在圖書館裡待到很晚來完成作業。\n"
    "[Statements]\n"
    "[1] John主修生物學。\n"
    "[2] John正在修讀人工智慧課程。\n"
    "[3] John是一名專注的學生。\n"
    "[4] John有一份兼職工作。\n"
    "[NLI]\n"
    "{response_example}\n\n"
    "Your actual task:\n"
    "[Context]\n{context_str}\n"
    "[Statements]\n{response_statements}"
)

QUESTION_GENERATION_PROMPT = PromptTemplate(
    "[System] Please act as a question generator and generate a possible question for the provided Answer and Context. Your generation should strictly follow the following rules:\n"
    '1. Begin your generation by examing whether the Answer is noncommittal or not. A noncommittal answer is one that is evasive, vague, or ambiguous. For example, "I don\'t know" or "I\'m not sure" are noncommittal answers. If the Answer is noncommittal, return 1; otherwise, return 0.\n'
    "2. Ensure the generated question is directly answerable by the provided Answer and does not solely rely on the Context. The question should be something that the Answer itself clearly addresses.\n"
    '3. Note that ALL the generated questions should NOT be reading comprehension-type questions, even if the source is mentioned in the Answer. Avoid questions that start with phrases such as "According to <any source>, ...", "Based on <any source>, ...", or any similar phrases.\n'
    "4. Consider all the questions that possibly fit the provided Answer and Context, and generate no more than 5.\n"
    "5. Generate questions even if the Answer is noncommittal.\n"
    "6. After providing your explanation, format the explanation, the verdict of noncommittal and the generated question with JSON format as follows: {json_format_example}\n\n"
    "Examples:\n"
    "[Context]\n阿爾伯特·愛因斯坦是一位出生於德國的理論物理學家，為大家公認有史以來最偉大和最具影響力的科學家之一。\n"
    "[Response]\n阿爾伯特·愛因斯坦出生於德國。\n"
    "[Question generation]\n{response_example}\n\n"
    "Your actual task:\n"
    "[Context] {context_str}\n"
    "[Response]\n{response_str}\n"
)
