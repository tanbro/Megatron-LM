#!/bin/bash

python3 -m torch.distributed.launch \
    --nnodes 1 \
    --nproc_per_node 2 \
pretrain_gpt2.py \
    --finetune \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 1024 \
    --seq-length 1024 \
    --batch-size 16 \
    --load "$LOAD_DIR" \
    --save "$SAVE_DIR" \
    --tensorboard-dir "$LOG_DIR" \
    --resume-dataloader \
    --train-data wikipedia \
    --lazy-loader \
    --tokenizer-type SentencePieceTokenizer \
    --tokenizer-path "$TOKENIZER_PATH" \
    --split 949,50,1 \
    --lr 0.00015 \
    --lr-decay-style cosine \
    --checkpoint-activations \
    --fp16