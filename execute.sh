#!/bin/bash

train() {
   # 1: yaml_path, 2: yaml_name, $3: update_args
   pkill -9 python; python -c 'import streaming; streaming.base.util.clean_stale_shared_memory()' # alternative hack: rm -rf /dev/shm/0000*
   rm -rf /tmp/streaming/*
   wait;
   sleep 3

   CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 composer train.py --config-path $1 --config-name $2  $3
}

# Step-1: Pre-training at 256x256 image resolution with 75% patch masking
train ./configs 256_pretrain.yaml "exp_name=mk1_pretrain model.train_mask_ratio=0.75"

# Step-2: Finetuning at 256x256 image resolution with no patch masking
train ./configs 256_finetune_nomask.yaml "exp_name=mk1_finetune model.train_mask_ratio=0.0 trainer.load_path=./trained_models/mk1_pretrain/latest-rank0.pt"

# Step-3: Pre-training at 512x512 image resolution with 75% patch masking
train ./configs 512_pretrain.yaml "exp_name=mk2_pretrain model.train_mask_ratio=0.75"

# Step-4: Finetuning at 512x512 image resolution with no patch masking
train ./configs 512_finetune_nomask.yaml "exp_name=mk2_finetune model.train_mask_ratio=0.0 trainer.load_path=./trained_models/mk2_pretrain/latest-rank0.pt"