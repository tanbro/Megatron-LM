#!/bin/bash

python3 -m torch.distributed.launch \
    --nnodes 1 \
    --nproc_per_node 2 \
    \
    pretrain_gpt2.py \
        --num-layers 36 \
        --hidden-size 1280 \
        --num-attention-heads 20 \
        --max-position-embeddings 1024 \
        --seq-length 1024 \
        --batch-size 4 \
        --load "$LOAD_DIR" \
        --save "$SAVE_DIR" \
        --tensorboard-dir "$LOG_DIR" \
        --resume-dataloader \
        --train-data "$TRAIN_DATA" \
        --lazy-loader \
        --tokenizer-type SentencePieceTokenizer \
        --tokenizer-path "$TOKENIZER_PATH" \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup .01 \
        --checkpoint-activations \
        --fp16