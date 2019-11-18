import fileinput
import json
import math
import random
import re
from itertools import chain

import numpy as np

from more_itertools import windowed
from torch.utils.data import Dataset, IterableDataset

_REGEX = r'.*?(([\.|\!|\?|。|！|？]\s*)+|.+)'
_RE_SPLIT_SENTENCES = re.compile(_REGEX)
del _REGEX


def split_sentences(text):
    groups = re.findall(_RE_SPLIT_SENTENCES, text)
    return [''.join(s) for s in groups]


class ForumQaDataset(Dataset):
    """论坛 QA 数据集

    论坛 QA 包括：

        * 问题标题
        * 问题正文
        * 问题类型标签列表
        * 回答列表

    这个数据集是将问题与类型、回答反复组合，循环迭代，返回给 Dataloader

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

    返回的数据是 token ID 的一维数组，形如::

        xxxxx<sep>xxxxxx[<sep>xxxx]<sep><sep>[bos]xxxxxxxxx[eos]
        Title      Text      tag                    Answer

    """

    def __init__(self, files, tokenizer, max_seq_len=1024, repeat_times=1000):
        self._files = files
        self._tokenizer = tokenizer
        self._max_seq_len = int(max_seq_len)
        if not self._max_seq_len > 0:
            raise ValueError('`max_seq_len` must be greater than 0')
        self._repeat_times = int(repeat_times)
        if not self._repeat_times > 0:
            raise ValueError('`repeat_times` must be greater than 0')
        self._files_lines = self._get_files_lines()
        self._pos = 0
        self._no_extract = False
        self._generator = self._create_generator()

    def __del__(self):
        self.close()

    def __len__(self):
        return self._files_lines * self._repeat_times

    def __getitem__(self, key):
        key = int(key)
        if key < 0:
            raise IndexError('Key must be greater than or equal 0')
        if key >= len(self):
            raise IndexError('Key is too big')
        if key == self._pos:
            value = next(self._generator)
            self._pos += 1
        else:
            if key < self.pos:
                self._generator.close()
                self._generator = self._create_generator()
                self._pos = 0
            self._no_extract = True
            try:
                while self.pos < key:
                    next(self._generator)
                    self._pos += 1
            finally:
                self._no_extract = False
            value = next(self._generator)
            self._pos += 1
        return value

    def _get_files_lines(self):
        with fileinput.input(self._files) as fp:
            return sum(1 for _ in fp)

    def _create_generator(self):
        i = 0
        while i < self._repeat_times:
            i += 1
            with fileinput.input(self._files) as fp:
                for s in fp:
                    s = s.strip()
                    if s:
                        if self._no_extract:
                            yield
                        else:
                            yield self.extract(s)

    def close(self):
        self._generator.close()

    def pickup_sentences(self, text, bos=False, eos=False):
        text = text.strip()
        if not text:
            return ''
        sentences = split_sentences(text)
        sentences = [
            self.tokenizer.EncodeAsIds(s.strip()).tokenization
            for s in sentences
        ]
        if bos:
            sentences[0].insert(0, self.tokenizer.TokenToId('<bos>'))
        if eos:
            sentences[-1].append(self.tokenizer.TokenToId('<eos>'))
        indices = list(range(len(sentences)))
        windows = list(chain.from_iterable(
            windowed(indices, n)
            for n in range(1, 1+len(indices))
        ))
        weights = [
            math.log(sum(len(sentences[idx]) for idx in indices))
            for indices in windows
        ]
        indices = random.choices(windows, weights)[0]
        ids = chain.from_iterable(sentences[i] for i in indices)
        return list(ids)

    def extract(self, text):
        if self.tokenizer is None:
            raise AttributeError('tokenizer was not set')
        data = json.loads(text)
        input_ids = []
        # 标题, 问题
        for k in ('title', 'text'):
            s = data[k].strip()
            ids = self.pickup_sentences(s)
            input_ids.extend(ids)
            input_ids.append(self.tokenizer.TokenToId('<sep>'))
        # 标签
        tags = data.get('tags')
        if tags:
            tag = random.choice(tags)
            input_ids.extend(self.tokenizer.EncodeAsIds(
                tag.strip()).tokenization)
            input_ids.append(self.tokenizer.TokenToId('<sep>'))
        # 问题/回答 的分隔符，保证连续两个 `<sep>`
        input_ids.append(self.tokenizer.TokenToId('<sep>'))
        # 回答
        texts = [m['text'] for m in data['answers']]
        if texts:
            s = random.choice(texts)
            ids = self.pickup_sentences(s, bos=True, eos=True)
            input_ids.extend(ids)
        ###
        # Pad / Strip
        total_tokens = self._max_seq_len + 1
        num_pad_tokens = total_tokens - len(input_ids)
        if num_pad_tokens > 0:
            input_ids.extend(
                [self.tokenizer.TokenToId('<pad>')] * num_pad_tokens)
        else:
            input_ids = input_ids[:total_tokens]

        return {'text': np.array(input_ids),}

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        self._tokenizer = value

    def SetTokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    @property
    def pos(self):
        return self._pos
