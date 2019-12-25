#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK"


CHECKPOINT_PATH=checkpoints/774m-hmwebmix-32kv32.bak
MPSIZE=2
NLAYERS=36
NHIDDEN=1280
NATT=20
MAXSEQLEN=1024

TOKTYPE=GPT2BPETokenizer_CN
TOKPATH=data/spm/gpt2_huamei_corpus_bpe_32k_v3.2.model

#SAMPLING ARGS
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       generate_samples.py \
       --model-parallel-size $MPSIZE \
       --num-layers $NLAYERS \
       --hidden-size $NHIDDEN \
       --load $CHECKPOINT_PATH \
       --num-attention-heads $NATT \
       --max-position-embeddings 1024 \
       --fp16 \
       --out-seq-length $MAXSEQLEN \
       --temperature $TEMP \
       --top_k $TOPK \
       --genfile dbg_unconditional.json \
       --num-samples 1 \
       --top_p $TOPP \
       --tokenizer-type $TOKTYPE \
       --tokenizer-path $TOKPATH \
       --recompute
