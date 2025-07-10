#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune.py \
    --do_train \
    --do_eval \
    --mlm \
    --field_ce \
    --lm_type bert \
    --output_dir results/09072025_v30k/finetune_v4 \
    --num_train_epochs 30\
    --data_fname card_transaction_30000_sampled \
    --batch_size 4 \
    --checkpoint 14500 \
    --field_hs 64 \
    --path_to_checkpoint results/09072025_v30k \
    --use_embeddings 
 
