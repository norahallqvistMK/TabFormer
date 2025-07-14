#!/bin/bash

#run with embeddings features
CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py \
    --do_train \
    --do_eval \
    --mlm \
    --field_ce \
    --lm_type bert \
    --output_dir results/13072025_400k_v1/finetune_v1 \
    --num_train_epochs 40\
    --data_fname card_transaction_400000_sampled  \
    --batch_size 32 \
    --checkpoint 19656 \
    --field_hs 64 \
    --path_to_checkpoint results/13072025_400k_v1\
    --use_embeddings 


#run with raw features
 CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py \
    --do_train \
    --do_eval \
    --mlm \
    --field_ce \
    --lm_type bert \
    --output_dir results/13072025_400k_v1/finetune_raw_v1 \
    --num_train_epochs 40\
    --data_fname card_transaction_400000_sampled \
    --batch_size 32 \
    --checkpoint 19656 \
    --field_hs 64 \
    --path_to_checkpoint results/13072025_400k_v1