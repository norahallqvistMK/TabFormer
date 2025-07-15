#!/bin/bash
#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4 python main.py \
    --do_train \
    --do_eval \
    --data_type card \
    --mlm \
    --field_ce \
    --lm_type bert \
    --output_dir results/15072025_200k_v1 \
    --num_train_epochs 200 \
    --data_fname card_transaction_400000_sampled \
    --batch_size 64 \
    --field_hs 64  \
    --nrows 200000
