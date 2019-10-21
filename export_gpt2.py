"""
将 NVIDIA/Megatron-LM 的 GPT2 模型检查点导出到 huggingface/transformers 的 GPT2LMHeadModel 模型对象和配置文件
"""

import os
import sys

from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

from arguments import get_args
from model import GPT2Model
from pretrain_gpt2 import initialize_distributed
from utils import load_checkpoint, move_weights, print_args


def main(args):
    # 几个强行修改默认值的参数
    args.release = True
    args.no_save_optim = True
    args.no_load_optim = True
    args.no_save_rng = True
    args.no_load_rng = True

    #
    print_args(args)
    print()

    print('Initialize torch.distributed...')
    initialize_distributed(args)
    print()

    print('Building NVIDIA/Megatron-LM GPT2 model ...')
    nv_gpt2_model = GPT2Model(
        num_layers=args.num_layers,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        embedding_dropout_prob=args.hidden_dropout,
        attention_dropout_prob=args.attention_dropout,
        output_dropout_prob=args.hidden_dropout,
        max_sequence_length=args.max_position_embeddings,
        checkpoint_activations=args.checkpoint_activations,
        checkpoint_num_layers=args.checkpoint_num_layers
    )
    print()

    print('Load NVIDIA/Megatron-LM GPT2 model from checkpoint ...')
    load_checkpoint(nv_gpt2_model, None, None, args)
    print()

    print('Construct huggingface/transformers.GPT2LMHeadModel object ...')
    hf_gpt2_config = GPT2Config(
        vocab_size_or_config_json_file=args.vocab_size,
        n_positions=args.max_position_embeddings,
        n_ctx=args.seq_length,
        n_embd=args.hidden_size,
        n_layer=args.num_layers,
        n_head=args.num_attention_heads,
        embd_pdrop=args.hidden_dropout,
        attn_pdrop=args.attention_dropout,
        layer_norm_epsilon=args.layernorm_epsilon
    )
    hf_gpt2_model = GPT2LMHeadModel(hf_gpt2_config)
    print()

    print('Loads weights from NVIDIA/Megatron-LM GPT2 model to huggingface/transformers.GPT2LMHeadModel via in place copy ...')
    move_weights(nv_gpt2_model, hf_gpt2_model, True)
    print()

    print(
        f'Save huggingface/transformers.GPT2LMHeadModel model and its configuration file to directory {args.save} ...'
    )
    os.makedirs(args.save, exist_ok=True)
    hf_gpt2_model.save_pretrained(args.save)
    print()


if __name__ == "__main__":
    args = get_args()
    status = main(args)
    sys.exit(status)
