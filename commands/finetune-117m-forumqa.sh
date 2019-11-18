#!/bin/bash

python3 -m torch.distributed.launch \
    --nnodes 1 \
    --nproc_per_node 2 \
    pretrain_gpt2_forumqa.py \
    --finetune \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 1024 \
    --seq-length 1024 \
    --batch-size 16 \
    --load "./checkpoints/gpt2-117m-emotion" \
    --save "./checkpoints/xinliqa.117m" \
    --tensorboard-dir "./runs/xinliqa.117m" \
    --resume-dataloader \
    --train-data "./data/xinliqa/train.json" \
    --valid-data "./data/xinliqa/val.json" \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-path "./data/spm/gpt2_huamei_corpus_bpe_32k_v2.model" \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --checkpoint-activations \
    --fp16