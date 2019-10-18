#!/bin/bash

python3 evaluate_gpt2.py \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --max-position-embeddings 1024 \
        --seq-length 1024 \
        --batch-size 16 \
        --load "$MODEL_DIR" \
        --valid-data "$VALID_DATA" \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-path "$SPM_MODEL" \
        --fp16