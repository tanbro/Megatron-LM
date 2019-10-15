#!/bin/bash

python3 evaluate_gpt2.py \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --max-position-embeddings 1024 \
        --seq-length 1024 \
        --batch-size 16 \
        --load checkpoints/gpt2-117m-$DSNAME \
        --valid-data data/val.json \
        --cloze-eval \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-path data/spm/gpt2_huamei_corpus_bpe_32k_v2.model \
        --fp16