# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"

export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/gehealthcarerootca1.crt


python Scripts_Model_Inference/testset_classification_inference.py \
    --model_path /qumulo/haoyan/DeepSeek-R1-Distill-Qwen-1.5B-full-fine-tuned_QA_Classification_with_bias_type/checkpoint-380 \
    --base_model_name DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset_name joyfine/LLM_Bias_Detection_CoT_Training \
    --experiment_tag Fine-tune-QA-Classification \
    --output_method save-csv



