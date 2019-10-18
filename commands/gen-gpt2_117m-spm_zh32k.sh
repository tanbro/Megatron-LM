#!/bin/bash

python3 generate_samples.py \
    --model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 1024 \
    --out-seq-length 256 \
    --load "$LOAD_DIR" \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-path "$TOKERIZER_PATH" \
    --fp16 \
    --top_p 1