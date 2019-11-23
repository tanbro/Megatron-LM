python3 generate_samples.py \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --max-position-embeddings 1024 \
    --seq-length 1024 \
    --load "checkpoints/345m-hmwebmix-bpe-v2/" \
    --tokenizer-type "GPT2BPETokenizer_CN" \
    --tokenizer-path "data/spm/gpt2_huamei_corpus_bpe_32k_v2.model" \
    --out-seq-length=256 \
    --fp16

我喜欢我喜欢


python3 generate_samples.py \
    --num-layers 12 \
    --hidden-size 768 \
    --num-attention-heads 12 \
    --max-position-embeddings 1024 \
    --seq-length 1024 \
    --load "checkpoints/gpt2-117m-emotion/" \
    --tokenizer-type "SentencePieceTokenizer" \
    --tokenizer-path "data/spm/gpt2_huamei_corpus_bpe_32k_v2.model" \
    --out-seq-length=256 \
    --fp16