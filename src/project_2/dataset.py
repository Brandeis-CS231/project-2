"""
This module implements class(es) and function(s) for dataset representation
"""

import json
from typing import Self, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from .tokenizer import Pair, Tokenizer


@dataclass
class Encoding:
    encoding: list[int]
    tokenizer: Tokenizer
    max_len: int

    @classmethod
    def from_tokens(
        cls, tokens: list[str], tokenizer: Tokenizer, max_len: int
    ) -> "Encoding":
        return Encoding(
            encoding=tokenizer.encode(tokens), tokenizer=tokenizer, max_len=max_len
        )

    def _trim(self) -> Self:
        if len(self.encoding) > self.max_len - 2:
            self.encoding = self.encoding[: self.max_len - 2]
        return self

    def _add_special_tokens(self) -> Self:
        self.encoding = (
            [self.tokenizer.bos_id] + self.encoding + [self.tokenizer.eos_id]
        )
        return self

    def _pad(self) -> Self:
        self.encoding = self.encoding + [
            self.tokenizer.pad_id for _ in range(self.max_len - len(self.encoding))
        ]
        return self

    def _shift_right(self) -> Self:
        self.encoding = self.encoding[1:]
        return self

    def _pop_eos(self) -> Self:
        self.encoding = self.encoding[:-1]
        return self

    def as_ids(self) -> list[int]:
        return self.encoding


@dataclass
class Sample:
    enc_inp_ids: torch.LongTensor
    dec_inp_ids: torch.LongTensor
    label_ids: torch.LongTensor

    @classmethod
    def from_pair(
        cls, pair: Pair, tokenizer: Tokenizer, max_src_len: int, max_tgt_len: int
    ) -> "Sample":
        source_encoding = (
            Encoding.from_tokens(tokenizer.tokenize(pair.src), tokenizer, max_src_len)
            ._trim()
            ._add_special_tokens()
            ._pad()
        )
        ## Shift left after special tokens and trimming -1
        ## Pop off eos and then pad
        target_encoding = (
            Encoding.from_tokens(tokenizer.tokenize(pair.tgt), tokenizer, max_tgt_len)
            ._trim()
            ._add_special_tokens()
            ._pop_eos()
            ._pad()
        )
        ## SHIFT Right - pop off bos 1
        labels = (
            Encoding.from_tokens(tokenizer.tokenize(pair.tgt), tokenizer, max_tgt_len)
            ._trim()
            ._add_special_tokens()
            ._shift_right()
            ._pad()
        )

        return Sample(
            enc_inp_ids=torch.LongTensor(source_encoding.as_ids()),
            dec_inp_ids=torch.LongTensor(target_encoding.as_ids()),
            label_ids=torch.LongTensor(labels.as_ids()),
        )

    def as_tuple(self) -> tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
        return (self.enc_inp_ids, self.dec_inp_ids, self.label_ids)


class SeqPairDataset(Dataset):
    def __init__(
        self, data_file: str, tokenizer: Tokenizer, max_src_len: int, max_tgt_len: int
    ):
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        pairs: list[Pair] = []
        with open(data_file, "r") as file:
            data = json.load(file)
            [pairs.append(Pair.from_json(pair)) for pair in data]

        self.samples: list[Sample] = [
            Sample.from_pair(pair, self.tokenizer, self.max_src_len, self.max_tgt_len)
            for pair in pairs
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        return self.samples[idx].as_tuple()
