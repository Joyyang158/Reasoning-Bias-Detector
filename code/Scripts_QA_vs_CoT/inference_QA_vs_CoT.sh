# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"

export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/gehealthcarerootca1.crt


# python Scripts_QA_vs_CoT/inference_QA_vs_CoT.py \
#     --model_path /qumulo/haoyan/DeepSeek-R1-Distill-Qwen-1.5B-full-fine-tuned_QA_Classification_with_bias_type/checkpoint-380 \
#     --dataset_file_path Results_Test_Data/GSM8K/gsm8k_QA_CoT_test_500.json \
#     --output_file Results_Model_Inference/QA_vs_CoT_verbosity.csv \
#     --bias_type verbosity \
#     --experiment_tag QA

# python Scripts_QA_vs_CoT/inference_QA_vs_CoT.py \
#     --model_path /qumulo/haoyan/DeepSeek-R1-Distill-Qwen-1.5B-full-fine-tuned_CoT_with_bias_type/checkpoint-400 \
#     --dataset_file_path Results_Test_Data/GSM8K/gsm8k_QA_CoT_test_500.json \
#     --output_file Results_Model_Inference/QA_vs_CoT_verbosity.csv \
#     --bias_type verbosity \
#     --experiment_tag CoT


python Scripts_QA_vs_CoT/inference_QA_vs_CoT.py \
    --model_path /qumulo/haoyan/DeepSeek-R1-Distill-Qwen-1.5B-full-fine-tuned_QA_Classification_with_bias_type/checkpoint-380 \
    --dataset_file_path Results_Test_Data/ArenaHumanPreference/Arena_reversed_test.json \
    --output_file Results_Model_Inference/QA_vs_CoT_bandwagon.csv \
    --bias_type bandwagon \
    --experiment_tag QA

python Scripts_QA_vs_CoT/inference_QA_vs_CoT.py \
    --model_path /qumulo/haoyan/DeepSeek-R1-Distill-Qwen-1.5B-full-fine-tuned_CoT_with_bias_type/checkpoint-400 \
    --dataset_file_path Results_Test_Data/ArenaHumanPreference/Arena_reversed_test.json \
    --output_file Results_Model_Inference/QA_vs_CoT_bandwagon.csv \
    --bias_type bandwagon \
    --experiment_tag CoT



