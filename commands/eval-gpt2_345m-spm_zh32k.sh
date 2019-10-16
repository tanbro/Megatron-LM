#!/bin/bash

python3 evaluate_gpt2.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --max-position-embeddings 1024 \
        --seq-length 1024 \
        --batch-size 16 \
        --load "checkpoints/gpt2-345m-$DATASET_NAME" \
        --valid-data "$VALID_DATA" \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-path data/spm/gpt2_huamei_corpus_bpe_32k_v2.model \
        --fp16 \
        --cloze-eval