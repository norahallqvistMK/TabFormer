#!/bin/bash

python cluster.py \
    --mlm \
    --field_ce \
    --lm_type bert \
    --num_train_epochs 30\
    --output_dir results/09072025_v50k/cluster_analysis \
    --data_fname card_transaction_50000_sampled \
    --batch_size 4 \
    --checkpoint 12870 \
    --field_hs 64 \
    --path_to_checkpoint results/11072025_50k_v1 \
    --pooling_strategy "average" 
    
    
 
