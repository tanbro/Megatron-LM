import fileinput
import json
import math
import random
import re
from copy import copy
from itertools import chain

from more_itertools import windowed
from torch.utils.data import IterableDataset

_RE_SPLIT_SENTENCES = re.compile(r'.*?[\.|\?|\!|。|？|！]|.+')


def split_sentences(text):
    return re.findall(_RE_SPLIT_SENTENCES, text)


class ForumQaDataset(IterableDataset):
    """论坛 QA 数据集

    论坛 QA 包括：

        * 问题标题
        * 问题正文
        * 问题类型标签列表
        * 回答列表

    这个数据集是一个 Infinit(Iterable-style) datasets，他将问题与类型、回答反复组合，循环迭代，返回给 Dataloader

    数据文件是 JSON lines 格式，每一行都是 JSON Object,其格式形如::

        {
            title: "...",
            text: "...",
            tags: ["...", "..."],  # Optional
            answers: [
                {text: "..."},
                {text: "..."},
                # ...
                {text: "..."},
            ]
        }

    """

    def __init__(self, files, tokenizer, max_seq_len=1024, repeat_times=math.inf):
        self._files = files
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len
        self._repeat_times = repeat_times

    def __iter__(self):
        i = 0
        while i < self._repeat_times:
            i += 1
            with fileinput.input(self._files) as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        yield self.extract(line)

    def pickup_sentences(self, text):
        text = text.strip()
        if not text:
            return ''
        sentences = split_sentences(text)
        tokenizer = self._tokenizer
        sentences = [
            tokenizer.EncodeAsIds(s.strip()).tokenization
            for s in sentences
        ]
        windows = chain.from_iterable(
            windowed(sentences, n)
            for n in range(1, 1+len(sentences))
        )
        candidates = [
            list(chain.from_iterable(x))
            for x in windows
        ]
        weights = [math.log(len(ids)) for ids in candidates]
        return random.choices(candidates, weights)[0]

    def extract(self, text):
        data = json.loads(text)
        tokenizer = self._tokenizer
        input_ids = []
        # 标题, 问题
        for k in ('title', 'text'):
            s = data[k].strip()
            ids = self.pickup_sentences(s)
            input_ids.extend(ids)
            input_ids.append(tokenizer.TokenToId('<sep>'))
        # 标签
        tags = data.get('tags')
        if tags:
            tag = random.choice(tags)
            input_ids.extend(tokenizer.EncodeAsIds(tag.strip()).tokenization)
            input_ids.append(tokenizer.TokenToId('<sep>'))
        # 问题/回答 的分隔符，保证连续两个 `<sep>`
        input_ids.append(tokenizer.TokenToId('<sep>'))
        # 回答
        texts = [m['text'] for m in data['answers']]
        if texts:
            s = random.choice(texts)
            ids = self.pickup_sentences(s)
            input_ids.append(tokenizer.TokenToId('<bos>'))
            input_ids.extend(ids)
        ###
        # Pad / Strip
        total_tokens = self._max_seq_len + 1
        num_pad_tokens = total_tokens - len(input_ids)
        if num_pad_tokens > 0:
            input_ids.extend([tokenizer.TokenToId('<pad>')] * num_pad_tokens)
        else:
            input_ids = input_ids[:total_tokens]

        return input_ids

    @property
    def tokenizer(self):
        return self._tokenizer
