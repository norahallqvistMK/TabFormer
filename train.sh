#!/bin/bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --do_train \
    --do_eval \
    --data_type card \
    --mlm \
    --field_ce \
    --lm_type bert \
    --output_dir results/09072025_v30k_v2 \
    --num_train_epochs 100 \
    --data_fname card_transaction_40k_sampled \
    --batch_size 8 \
    --field_hs 64 \
 
