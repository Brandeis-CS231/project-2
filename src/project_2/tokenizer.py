"""
The tokenizer module
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Self, Sequence

BOS = "<bos>"
EOS = "<eos>"
PAD = "<pad>"
UNK = "<unk>"
SPECIAL_TOKENS = [BOS, EOS, PAD, UNK]


@dataclass
class Pair:
    tgt: str
    src: str
    sent: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.tgt, str):
            raise ValueError("Target sentence is not string")
        if not isinstance(self.src, str):
            raise ValueError("Source sentence is not string")
        self.sent = self.src + " " + self.tgt

    @classmethod
    def from_json(cls, pair: dict[str, str]) -> "Pair":
        return Pair(tgt=pair["tgt"], src=pair["src"])


@dataclass
class TokenizerConfig:
    special_tokens: List[str] = field(default_factory=lambda: [BOS, EOS, PAD, UNK])
    min_freq: int = 2


@dataclass
class Tokenizer:
    config: TokenizerConfig = field(default_factory=TokenizerConfig)
    word2idx: Dict[str, int] = field(init=False, default_factory=dict)
    idx2word: Dict[int, str] = field(init=False, default_factory=dict)
    _is_built: bool = field(init=False, default=False)

    def from_file(self, fpath: str | Path) -> Self:
        tokens = []

        with open(fpath, "r") as json_file:
            pairs_dict = json.load(json_file)

            for pair_dict in pairs_dict:
                pair = Pair(src=pair_dict["src"], tgt=pair_dict["tgt"])

                tokens.extend(Tokenizer.tokenize(pair.sent))

        token_count = Counter(tokens)
        valid_tokens = [
            token for token, freq in token_count.items() if freq >= self.config.min_freq
        ]
        valid_tokens += self.config.special_tokens

        for i, token in enumerate(valid_tokens):
            self.word2idx[token] = i
            self.idx2word[i] = token

        self._is_built = True
        return self

    @staticmethod
    def tokenize(sentence: str) -> List[str]:
        return sentence.lower().strip().split()

    def encode(self, tokens: Sequence[str]) -> List[int]:
        if not self._is_built:
            raise RuntimeError("Tokenizer has not been built")
        return [self.word2idx.get(token, self.word2idx[UNK]) for token in tokens]

    def decode(self, ids: Sequence[int]) -> List[str]:
        if not self._is_built:
            raise RuntimeError("Tokenizer has not been built")
        return [self.idx2word[id_] for id_ in ids]

    @property
    def bos_id(self) -> int:
        return self.word2idx[BOS]

    @property
    def eos_id(self) -> int:
        return self.word2idx[EOS]

    @property
    def pad_id(self) -> int:
        return self.word2idx[PAD]

    @property
    def unk_id(self) -> int:
        return self.word2idx[UNK]

    @property
    def src_vocab(self) -> Dict[str, int]:
        return self.word2idx

    @property
    def tgt_vocab(self) -> Dict[int, str]:
        return self.idx2word

    @property
    def src_vocab_size(self) -> int:
        return len(self.word2idx)

    @property
    def tgt_vocab_size(self) -> int:
        return len(self.idx2word)
