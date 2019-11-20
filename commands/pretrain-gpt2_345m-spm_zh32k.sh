#!/bin/bash

python3 -m torch.distributed.launch \
    --nnodes 1 \
    --nproc_per_node 8 \
    \
    pretrain_gpt2.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --max-position-embeddings 1024 \
        --seq-length 1024 \
        --batch-size 16 \
        --save /public/megatronlm/checkpoints/345m-hmwebmix \
        --load /public/megatronlm/checkpoints/345m-hmwebmix \
        --tensorboard-dir /public/megatronlm/runs/345m-hmwebmix \
        --resume-dataloader \
        --train-data /public/megatronlm/data/hmwebmix/hmwebmix.train.json \
        --shuffle \
        --lazy-loader \
        --tokenizer-type GPT2BPETokenizer_CN \
        --tokenizer-path /public/megatronlm/data/spm/gpt2_huamei_corpus_bpe_32k_v2.model \
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