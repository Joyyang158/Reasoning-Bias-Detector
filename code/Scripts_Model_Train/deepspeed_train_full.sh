# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"

export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/gehealthcarerootca1.crt

export WANDB_API_KEY="f1ac3b92a14678e4bb758f670554d9fd36549410"

export HF_TOKEN='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'

huggingface-cli login --token "$HF_TOKEN"


accelerate launch --config_file Scripts_Model_Train/deepspeed_zero3.yaml Scripts_Model_Train/deepspeed_train_full.py --config Scripts_Model_Train/deepspeed_config_train_full.yaml --experiment_tag CoT