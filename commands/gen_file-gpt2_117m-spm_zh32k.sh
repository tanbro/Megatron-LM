#!/bin/bash

python3 generate_samples2.py \
    --model-parallel-size 1 \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 1024 \
    --out-seq-length 256 \
    --load checkpoints/gpt2-117m-emotion \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-path data/spm/gpt2_huamei_corpus_bpe_32k_v2.model \
    --fp16 \
    --sample-input-file data/text1.txt \
    --sample-output-file output.txt \
    --top_p 1