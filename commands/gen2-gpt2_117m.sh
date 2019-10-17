#!/bin/bash

python3 generate_samples2.py \
    --model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 1024 \
    --out-seq-length 256 \
    --load "$MODEL_DIR" \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-path "$SPM_MODEL" \
    --fp16 \
    --sample-input-file "$INPUT_FILE" \
    --sample-output-file "$OUTPUT_FILE" \
    --top_p 1