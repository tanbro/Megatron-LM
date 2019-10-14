#!/bin/bash

python3 -m torch.distributed.launch \
    --nnodes 1 \
    --nproc_per_node 2 \
    \
    pretrain_gpt2.py \
        --num-layers 12 \
        --hidden-size 768 \
        --num-attention-heads 12 \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --batch-size 16 \
        --train-iters 1000000 \
        --save-interval 1000 \
        --save checkpoints/gpt2_117m_$DSNAME \
        --load checkpoints/gpt2_117m_$DSNAME \
        --tensorboard-dir logs/gpt2_117m_$DSNAME \
        --resume-dataloader \
        --train-data wikipedia \
        --lazy-loader \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-path data/spm/gpt2_huamei_corpus_bpe_32k_v2.model \
        --cache-dir cache \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup .01 \
        --checkpoint-activations \
        --fp16