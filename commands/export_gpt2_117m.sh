#!/bin/bash

python3 export_gpt2.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 1024 \
    --seq-length 1024 \
    --batch-size 16 \
    --vocab-size 32128 \
    --load "$LOAD_DIR" \
    --save "$SAVE_DIR" \
    --fp16
