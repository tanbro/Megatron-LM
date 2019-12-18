"""GPT2 续写 - 交互式命令行
"""

import os
import sys

import torch

from arguments import get_args
from generate_samples import get_token_stream, prepare_tokenizer, setup_model
from pretrain_gpt2 import (get_masks_and_position_ids, initialize_distributed,
                           set_random_seed)


def gen(context_tokens, model, tokenizer, args):
    with torch.no_grad():
        context_length = len(context_tokens)
        token_stream = get_token_stream(
            model, [context_tokens], tokenizer, args)
        for i, (output_tokens, _) in enumerate(token_stream):
            if context_length + i >= args.seq_length:
                break
            ids = output_tokens.cpu().numpy().tolist()[0]
            yield ids[-1]


def main():
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

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

    # generate samples
    # setting default batch size to 1
    args.batch_size = 1

    # interact
    print('Started', file=sys.stderr, flush=True)  # 作为启动标志
    try:
        while True:
            contex_text = input('>>> ').strip()
            if not contex_text:
                print(f'输入不可为空{os.linesep}', file=sys.stderr, flush=True)
                continue
            contex_ids = tokenizer.EncodeAsIds(contex_text).tokenization
            for id_ in gen(contex_ids, model, tokenizer, args):
                s = tokenizer.DecodeIds([id_])
                print(s, end='')
            print(flush=True)
    except (KeyboardInterrupt, EOFError) as err:
        print(err, file=sys.stderr)


if __name__ == '__main__':
    main()
