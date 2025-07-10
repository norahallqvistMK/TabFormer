#!/bin/bash

python cluster.py \
    --mlm \
    --field_ce \
    --lm_type bert \
    --num_train_epochs 30\
    --output_dir results/09072025_v30k/cluster_analysis \
    --data_fname card_transaction_30000_sampled \
    --batch_size 4 \
    --checkpoint 14500 \
    --field_hs 64 \
    --path_to_checkpoint results/09072025_v30k \
    --pooling_strategy "average" 
    
    
 
