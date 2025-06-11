# LassieRAG CHANGELOG

## v0.2.1
2025-04-29
- 修改 model_loader module 中的 model loader 使用 `from_config` 的方式起始模型時，不會因只放入LLM或Embedding model而出錯
- 修改 node_transformation module 中 chunking 方法 recursive splitter 為可自訂 separators list
- 新增 indices module 使用 opensearch 建立index之流程中，使用nodes作為data source的方式

## v0.2.0
2024-12-24

- 自 v0.2.0 版開始所有RAG流程中的modules移動至core中，並移除modules路徑
- 新增 model_loader module 中的 Huggingface 與 openAILike 介面
- 新增 model_loader module 中的不同 model loader 通用介面 (`ModelLoader`)
- 新增 data_reader module 中的不同 data loader 通用介面 (`DataReader`)
- 新增 node_transformation module 中使用不同 chunking 方法之介面
- 新增 node_transformation module 中的不同 preprocessors 通用介面 (`PreProcessorBuilder`)
- 新增 indices module 中使用opensearch 建立index之流程 (`OSIndexLoader`)
- 新增 indices module 中的不同 indexing 方法通用介面 (`IndexLoader`)
- 新增 retriever module 中透過 Index 建立 retriever 之通用介面 (`RetrieverBuilder`)
- 新增 postprocessor module 中建立不同 postprocessors 通用介面 (`PostProcessorBuilder`)
- 新增 prompts 中用於 MultiQA 任務的 prompt (一般/chat版本) 及判斷模型是否使用 chat mode 的 selector
- 新增 synthesizer module 中 multiQA 專用 synthesizer (`MultiQA`)
- 新增 synthesizer module 中建立不同 synthesizer 的通用介面 (`SynthesizerBuilder`)
- 新增 query_engine module 中建立使用 LlamaIndex RetrieverQueryEngine 來建立 query_engine 的方法 (`QueryEngineBuilder`)
- 修改 evaluation module 中的 `ContextRelevanceEvaluator` 為使用 edit distance <= 2 來進行各文本相關語句的對應，以避免評估模型的回覆語句差異造成對應錯誤
- 修改 evaluation module 中的 `ResponseRelevanceEvaluator` 為當 noncommittal 指標為 1 時，response relevance 分數為0，以避免高估模型回應相關程度
- 修改 evaluation module 中 mlflow recorder 不預設各 metric 數值為 None，以避免在使用單一指標時 mlflow 上傳紀錄出錯
- 修改 evaluation module 中 `RAGEvaluationRunner` 為可自行選擇是否上傳 mlflow 紀錄
- 修改 evaluation module 中 `RAGEvaluationRunner` 為自動儲存各 metric 評估結果
- 新增 chunking experiment script 及相關 rag_config 檔案 (scripts/chunking_experiment.py, chunk_exp.sh, rag_config.yml)
- 修改 E2E Evaluation 範例為使用新的 RAG query_engine 建立流程 (examples/E2E_Evaluation.ipynb)


## v0.1.0
2024-10-29

- 新增 RAG evaluation module 中的 batch evaluation runner
- 新增 modified RAGAs reference-free metrics (ContextRelevanceEvaluator, FaithfulnessEvaluator, ResponseRelevanceEvaluator)
