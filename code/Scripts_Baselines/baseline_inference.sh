# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"


# python Scripts_Baselines/baseline_inference.py \
#     --model_path Skywork/Skywork-Critic-Llama-3.1-8B \
#     --bias_type verbose


# python Scripts_Baselines/baseline_inference.py \
#     --model_path prometheus-eval/prometheus-7b-v2.0 \
#     --bias_type verbose


# python Scripts_Baselines/baseline_inference.py \
#     --model_path prometheus-eval/prometheus-7b-v2.0 \
#     --bias_type bandwagon

# python Scripts_Baselines/baseline_inference.py \
#     --model_path prometheus-eval/prometheus-7b-v2.0 \
#     --bias_type sentiment


python Scripts_Baselines/baseline_inference.py \
    --model_path prometheus-eval/prometheus-7b-v2.0 \
    --bias_type position


python Scripts_Baselines/baseline_inference.py \
    --model_path prometheus-eval/prometheus-7b-v2.0 \
    --bias_type sentiment

python Scripts_Baselines/baseline_inference.py \
    --model_path prometheus-eval/prometheus-8x7b-v2.0 \
    --bias_type bandwagon




    

