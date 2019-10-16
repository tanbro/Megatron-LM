# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2

从 NVIDIA/Megatron-LM 的 generate_samples.py 复制，加以修改，适应我们的需求
"""

import csv
import os
import sys
import random
import json
import copy
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time

from tqdm import tqdm

from arguments import get_args
from contextlib import closing
from utils import Timers

from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from pretrain_gpt2 import get_train_val_test_data
from pretrain_gpt2 import get_masks_and_position_ids

from utils import load_checkpoint
from data_utils import make_tokenizer
from configure_data import configure_data

import mpu

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0


def get_model(args):
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_layers=args.num_layers,
                      vocab_size=args.vocab_size,
                      hidden_size=args.hidden_size,
                      num_attention_heads=args.num_attention_heads,
                      embedding_dropout_prob=args.hidden_dropout,
                      attention_dropout_prob=args.attention_dropout,
                      output_dropout_prob=args.hidden_dropout,
                      max_sequence_length=args.max_position_embeddings,
                      checkpoint_activations=args.checkpoint_activations,
                      checkpoint_num_layers=args.checkpoint_num_layers,
                      parallel_output=False)

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    # Wrap model for distributed training.
    model = DDP(model)

    return model


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    if args.load is not None:
        _ = load_checkpoint(
            model, None, None, args)

    return model


def get_batch(context_tokens, args):
    tokens = context_tokens
    tokens = tokens.view(args.batch_size, -1).contiguous()
    device = args.device
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens,
        args.eod_token,
        args.reset_position_ids,
        args.reset_attention_mask)

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[
            0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        # logits=logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[...,
                                 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
        # going back to 2D
        # logits=logits.view(1, -1).contiguous()

    return logits


def generate_samples_input_from_file(model, tokenizer, args):

    if args.sample_input_file == "":
        if mpu.get_model_parallel_rank() == 0:
            print("args.sample_input_file CAN NOT BE empty!\n")
        return

    if mpu.get_model_parallel_rank() == 0:
        # 输入函数
        if os.path.isfile(args.sample_input_file):
            input_lines = sum(1 for _ in tqdm(open(args.sample_input_file), f'Get lines of {args.sample_input_file}'))
        else:
            input_lines = None
        _, ext = os.path.splitext(args.sample_input_file)
        if ext.lower in ('.json', 'jsonl', 'jsonline', 'jsonlines'):
            # 视作 loose-json/json-lines

            def read_fn():
                with open(args.sample_input_file, encoding='utf8') as fp:
                    for line in fp:
                        line = line.strip()
                        yield json.loads(line)['text'].strip() if line else ''

        else:
            # 视作按行分割的平面文本

            def read_fn():
                with open(args.sample_input_file) as fp:
                    yield from (line.strip() for line in fp)

        # 输出函数
        _, ext = os.path.splitext(args.sample_output_file)
        ext = ext.lower()
        if ext in ('.json', 'jsonl', 'jsonline', 'jsonlines'):
            # 视作 loose-json/json-lines

            def write_fn():
                with open(args.sample_output_file, 'w+') as fp:
                    while True:
                        in_text, out_text = yield
                        if any(m is None for m in (in_text, out_text)):
                            break
                        print(json.dumps(
                            {'input': in_text, 'output': out_text}, ensure_ascii=False), file=fp)

        elif ext in ('.csv', '.tsv'):
            # 视作 csv/tsv
            delimiter = ',' if ext == '.csv' else '\t'

            def write_fn():
                with open(args.sample_output_file, 'w+') as fp:
                    writer = csv.writer(
                        args.sample_output_file, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
                    while True:
                        in_text, out_text = yield
                        if any(m is None for m in (in_text, out_text)):
                            break
                        writer.writerow([in_text, out_text])

        else:
            # 平面文本，只要输出结果

            def write_fn():
                with open(args.sample_output_file, 'w+') as fp:
                    while True:
                        _, out_text = yield
                        if out_text is None:
                            break
                        print(out_text, file=fp)

    else:
        raise NotImplementedError()

    model.eval()
    with torch.no_grad(), closing(write_fn()) as writer:
        next(writer)
        for raw_text in tqdm(read_fn(), total=input_lines):
            if not raw_text:
                continue
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0

            if mpu.get_model_parallel_rank() == 0:
                context_tokens = tokenizer.EncodeAsIds(
                    raw_text).tokenization
                context_length = len(context_tokens)

                if context_length >= args.seq_length//2:
                    print(
                        f"\nContext length {context_length}"
                        "\nPlease give smaller context (half of the sequence length)!",
                        file=sys.stderr
                    )
                    continue
            else:
                context_tokens = tokenizer.EncodeAsIds(
                    "EMPTY TEXT").tokenization
                context_length = len(context_tokens)

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(
                terminate_runs_tensor,
                mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group()
            )
            terminate_runs = terminate_runs_tensor[0].item()

            token_stream = get_token_stream(
                model, [context_tokens], tokenizer, args)
            for decode_tokens in token_stream:
                # token_end = decode_tokens.find("<|endoftext|>")
                # if token_end > 0:
                #     break
                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()

            if mpu.get_model_parallel_rank() == 0:
                trim_decode_tokens = tokenizer.DecodeIds(decode_tokens)[
                    len(raw_text):]
                writer.send((raw_text, trim_decode_tokens))

            torch.distributed.barrier(group=mpu.get_model_parallel_group())


def pad_batch(batch, tokenizer, args):
    pad_id = tokenizer.get_command('pad').Id
    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < args.seq_length:
            tokens.extend([pad_id]*(args.seq_length-context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def get_token_stream(model, context_tokens, tokenizer, args):
    pad_id = tokenizer.get_command('pad').Id
    # context_length = len(context_tokens)
    # if context_length < args.seq_length:
    #     context_tokens = context_tokens + [pad_id] * (args.seq_length - context_length)
    context_tokens, context_lengths = pad_batch(
        context_tokens, tokenizer, args)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)
    # context_length_tensor = torch.cuda.LongTensor([context_length])

    torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(
    ), group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(
    ), group=mpu.get_model_parallel_group())

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(
        context_tokens_tensor, args)

    counter = 0
    org_context_length = context_length

    layer_past = None

    batch_token_iterator = sample_sequence_batch(
        model, context_tokens_tensor, context_length_tensor, attention_mask, position_ids, tokenizer, args)
    for tokens, lengths in batch_token_iterator:
        context_length += 1
        yield tokens[:, :context_length], lengths


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1-boolean)*val1 + boolean*val2


def sample_sequence_batch(model, context_tokens, context_lengths, attention_mask, position_ids, tokenizer, args, maxlen=None):
    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()
        eos_id = tokenizer.get_command('eos').Id

        counter = 0
        org_context_length = context_length

        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        if maxlen is None:
            maxlen = args.seq_length - 1
            if maxlen > (org_context_length + args.out_seq_length):
                maxlen = org_context_length + args.out_seq_length

        lengths = torch.ones([batch_size]).long().cuda()*maxlen

        while context_length <= (maxlen):

            if args.recompute:
                logits = model(tokens, position_ids, attention_mask)
                logits = logits[:, context_length - 1, :]
            else:
                if counter == 0:
                    tokens2use = tokens[:, :context_length]
                    positions2use = position_ids[:, :context_length]
                else:
                    tokens2use = tokens[:, context_length -
                                        1].view(batch_size, -1)
                    positions2use = position_ids[:,
                                                 context_length - 1].view(batch_size, -1)
                logits, layer_past = model(
                    tokens2use, positions2use, attention_mask, layer_past=layer_past, get_present=True)
                logits = logits[:, -1].view(batch_size, -1).contiguous()

            if args.greedy:
                prev = torch.argmax(logits, dim=-1).view(-1)
            else:
                logits /= args.temperature
                logits = top_k_logits(
                    logits, top_k=args.top_k, top_p=args.top_p)
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1).view(-1)

            print_logits = []
            for p in prev:
                print_logits.append([logits[i, p].item()
                                     for i in range(batch_size)])
            started = context_lengths <= context_length
            tokens[:, context_length] = switch(
                tokens[:, context_length].view(-1), prev, started)
            context_length += 1
            counter += 1

            done_token = (prev == eos_id).byte()
            just_finished = (done_token & ~is_done).bool()
            lengths[just_finished.view(-1)] = context_length
            was_done = is_done
            is_done = is_done | done_token
            done = torch.all(is_done)

            yield tokens, lengths
            if done:
                break


def prepare_tokenizer(args):

    tokenizer_args = {
        'tokenizer_type': args.tokenizer_type,
        'corpus': None,
        'model_path': args.tokenizer_path,
        'vocab_size': args.vocab_size,
        'model_type': args.tokenizer_model_type,
        'cache_dir': args.cache_dir}
    tokenizer = make_tokenizer(**tokenizer_args)

    args.tokenizer_num_tokens = tokenizer.num_tokens
    args.tokenizer_num_type_tokens = tokenizer.num_type_tokens
    args.eod_token = tokenizer.get_command('eos').Id

    after = tokenizer.num_tokens
    multiple = args.make_vocab_size_divisible_by * \
        mpu.get_model_parallel_world_size()
    if multiple != 0:
        while (after % multiple) != 0:
            after += 1

    args.vocab_size = after
    print("prepare tokenizer done", flush=True)

    return tokenizer


def main():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Timer.
    timers = Timers()

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    args.device = torch.cuda.current_device()

    # setting default batch size to 1
    args.batch_size = 1

    #
    if args.sample_input_file == "":
        print("Argument: sample-input-file can't be empty\n", file=sys.stderr)
        sys.exit(1)
    if args.sample_output_file == "":
        print("Argument: sample-output-file can't be empty\n", file=sys.stderr)
        sys.exit(1)

    # generate samples

    generate_samples_input_from_file(model, tokenizer, args)


if __name__ == "__main__":
    main()
