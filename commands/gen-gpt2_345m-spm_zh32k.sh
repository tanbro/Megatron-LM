#!/bin/bash

python3 generate_samples.py \
    --model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --max-position-embeddings 1024 \
    --out-seq-length 256 \
    --load checkpoints/gpt2_345m_hm10g \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-path data/spm/gpt2_huamei_corpus_bpe_32k_v2.model \
    --fp16 \
    --cache-dir cache \
    --top_p 1