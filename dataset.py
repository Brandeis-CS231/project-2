"""
This module implements class(es) and function(s) for dataset representation
"""
import json
from typing import Tuple, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from tokenizer import Pair, Tokenizer


@dataclass
class Sample:
    enc_inp_ids: torch.LongTensor
    dec_inp_ids: torch.LongTensor
    label_ids: torch.LongTensor


class SeqPairDataset(Dataset):
    def __init__(
            self,
            data_file: str,
            tokenizer: Tokenizer,
            max_src_len: int,
            max_tgt_len: int
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id
        self.pad_id = tokenizer.pad_id

        with open(data_file, "r", encoding="utf-8") as f:
            data_pairs = json.load(f)

        for p in data_pairs:
            src = p['src']
            tgt = p['tgt']
            pair = Pair(tgt, src)
            src = Tokenizer.tokenize(src)
            tgt = Tokenizer.tokenize(tgt)
            src_ids = tokenizer.encode(src)
            tgt_ids = tokenizer.encode(tgt)
            encoder_input_ids = torch.LongTensor(self._add_specials_and_trim(src_ids, max_src_len))
            tgt_ids = self._add_specials_and_trim(tgt_ids, max_tgt_len)
            decoder_input_ids = torch.LongTensor(tgt_ids[:-1])
            label_ids = torch.LongTensor(tgt_ids[1:])
            self.samples.append((encoder_input_ids, decoder_input_ids, label_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        return self.samples[idx]

    def _add_specials_and_trim(self, token_ids: list[int], max_len: int):
        if len(token_ids) > (max_len - 2):
            token_ids = token_ids[:(max_len - 2)]
        token_ids.insert(0, self.bos_id)
        token_ids.append(self.eos_id)
        token_ids = self._pad(token_ids, max_len)
        return token_ids

    def _pad(self, token_ids: list[int], max_len: int):
        if len(token_ids) < max_len:
            token_ids.extend([self.pad_id] * (max_len - len(token_ids)))
        return token_ids

