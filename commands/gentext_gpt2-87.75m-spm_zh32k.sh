#!/bin/bash

python3 generate_samples.py \
    --model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 768 \
    --seq-length 256 \
    --load checkpoints/gpt2_87.75m_hm8g \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-path data/spm/gpt2_huamei_corpus_bpe_32k_v2.model \
    --fp16 \
    --cache-dir cache \
    --out-seq-length 768 \
    --top_p 0.9