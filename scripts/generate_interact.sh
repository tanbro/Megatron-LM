#!/bin/bash

python generate_samples_interactive.py \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-path data/spm/gpt2_huamei_corpus_bpe_32k_v2.model \
    --load checkpoints/gpt2-117m-emotion \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 1024 \
    --out-seq-length 128