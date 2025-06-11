config_path="/LassieRAG/scripts/rag_config.yml"
chunker_test_param='{"recursive":{"chunk_size":[250],"chunk_overlap":[20]}}'
data_source_path='{"etc": "/LassieRAG/dataset/etc/source_files"}'
eval_dataset_path='{"etc": "/LassieRAG/dataset/etc/etc_qa.json"}'
output_dir='/LassieRAG/chunking_strategy/test'

python3 -u chunking_experiment.py \
    --config_path $config_path \
    --chunker_test_param "$chunker_test_param" \
    --data_source_path "$data_source_path" \
    --eval_dataset_path "$eval_dataset_path" \
    --retrieval_topk 5 \
    --output_dir $output_dir \
    --num_worker 2 \
    --is_ft_model 'True'