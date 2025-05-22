# export CUDA_VISIBLE_DEVICES=4,5

export HF_HOME="/qumulo/shared_data/haoyan_hf_cache/"

export REQUESTS_CA_BUNDLE=/usr/local/share/ca-certificates/gehealthcarerootca1.crt

export WANDB_API_KEY="f1ac3b92a14678e4bb758f670554d9fd36549410"

accelerate launch --num_processes 2 Scripts_Model_Train/train_classification.py --config Scripts_Model_Train/config_train_classification.yaml