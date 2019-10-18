#!/bin/bash

python3 predict_gpt2.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 1024 \
    --out-seq-length 256 \
    --load "$LOAD_DIR" \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-path "$TOKENIZER_PATH" \
    --sample-input-file "$INPUT_FILE" \
    --sample-output-file "$OUTPUT_FILE" \
    --fp16 \
    --top_p 1