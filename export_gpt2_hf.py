from pretrain_gpt2 import initialize_distributed
from utils import load_checkpoint
from model import GPT2Model
import os
from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from utils import *

model_dir = "/home/Public/Megatron-LM/checkpoints/gpt2_345m_hm10g"
hf_model_dir = os.path.join(model_dir, "gpt2_345m_hf")

if not os.path.exists(hf_model_dir):
    os.makedirs(hf_model_dir)


class c_args(object):
    def __init__(self):
        self.distributed_backend = "nccl"
        self.world_size = 1
        self.rank = 0
        self.load = "/home/Public/Megatron-LM/checkpoints/gpt2_345m_hm10g"
        self.finetune = False
        self.no_load_optim = True
        self.no_load_rng = True
        self.model_parallel_size = 1
        self.local_rank = 0
        self.fp16 = True


args = c_args()

initialize_distributed(args)

print('building GPT2 model ...\n')
model = GPT2Model(num_layers=24,
                  vocab_size=32128,
                  hidden_size=1024,
                  num_attention_heads=16,
                  embedding_dropout_prob=0.1,
                  attention_dropout_prob=0.1,
                  output_dropout_prob=0.1,
                  max_sequence_length=1024,
                  checkpoint_activations=True,
                  checkpoint_num_layers=1,
                  parallel_output=False)

load_checkpoint( model, None, None, args)

print(model)

config_class = GPT2Config(
        vocab_size_or_config_json_file=32128,
        n_positions=1024,
        n_ctx=1024,
        n_embd=1024,
        n_layer=24,
        n_head=16,
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,

        num_labels=1,
        summary_type='cls_index',
        summary_use_proj=True,
        summary_activation=None,
        summary_proj_to_labels=True,
        summary_first_dropout=0.1,
)

print(config_class)

config_class.save_pretrained(hf_model_dir)
