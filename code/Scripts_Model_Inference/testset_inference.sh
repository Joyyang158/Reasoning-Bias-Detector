# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"

export HF_TOKEN='xxx'

huggingface-cli login --token "$HF_TOKEN"


python Scripts_Model_Inference/testset_infernece.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --base_model_name DeepSeek-R1-Distill-Qwen-14B \
    --dataset_name joyfine/LLM_Bias_Detection_CoT_Training \
    --experiment_tag Zero-shot \
    --output_method save-csv

python Scripts_Model_Inference/testset_infernece.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --base_model_name DeepSeek-R1-Distill-Qwen-14B \
    --dataset_name joyfine/LLM_Bias_Detection_CoT_Training \
    --experiment_tag Few-shot-QA \
    --output_method save-csv

python Scripts_Model_Inference/testset_infernece.py \
    --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --base_model_name DeepSeek-R1-Distill-Qwen-14B \
    --dataset_name joyfine/LLM_Bias_Detection_CoT_Training \
    --experiment_tag Few-shot-CoT \
    --output_method save-csv

# python Scripts_Model_Inference/testset_infernece.py \
#     --model_path /qumulo/haoyan/DeepSeek-R1-Distill-Qwen-7B-full-fine-tuned_CoT_with_bias_type/checkpoint-416 \
#     --base_model_name DeepSeek-R1-Distill-Qwen-7B \
#     --dataset_name joyfine/LLM_Bias_Detection_CoT_Training \
#     --experiment_tag Fine-tune-CoT_with_Bias_Type \
#     --output_method save-csv

# python Scripts_Model_Inference/testset_infernece.py \
#     --model_path /qumulo/haoyan/DeepSeek-R1-Distill-Qwen-14B-full-fine-tuned_CoT_with_bias_type/final_model \
#     --base_model_name DeepSeek-R1-Distill-Qwen-14B \
#     --dataset_name joyfine/LLM_Bias_Detection_CoT_Training \
#     --experiment_tag Fine-tune-CoT_with_Bias_Type \
#     --output_method save-csv

# python Scripts_Model_Inference/testset_infernece.py \
#     --model_path deepseek-ai/DeepSeek-R1 \
#     --base_model_name DeepSeek-R1-671B \
#     --dataset_name joyfine/LLM_Bias_Detection_CoT_Training \
#     --experiment_tag Zero-shot \
#     --output_method api-deepseek

# python Scripts_Model_Inference/testset_infernece.py \
#     --model_path /qumulo/haoyan/Qwen2.5-Math-1.5B-full-fine-tuned_QA_Instruction_with_bias_type/checkpoint-300  \
#     --base_model_name Qwen2.5-Math-1.5B \
#     --dataset_name joyfine/LLM_Bias_Detection_CoT_Training \
#     --experiment_tag Fine-tune-QA \
#     --output_method save-csv


