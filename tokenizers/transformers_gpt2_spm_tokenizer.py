import json
import logging
import sentencepiece as spm
from transformers import PreTrainedTokenizer


def _get_logger():
    return logging.getLogger(__name__)


class TransformersGpt2SentencePieceTokenizer(PreTrainedTokenizer):
    def __init__(self, model_path, max_len=None, **kwargs):
        self._sp = spm.SentencePieceProcessor()
        super().__init__(max_len=max_len, **kwargs)

    @property
    def vocab_size(self):
        return len(self.encoder)
