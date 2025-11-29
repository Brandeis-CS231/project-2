"""
This module implements class(es) and function(s) for dataset representation
"""
import json
from typing import Tuple
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
        # load data from data_file
        raw = json.load(open(data_file, 'r'))

        self.pairs = []
        self.samples = []

        # process each sequence pair
        for item in raw:
            # tokenize and encode
            src = tokenizer.encode(tokenizer.tokenize(item['src']))
            # trim/pad and add special tokens
            src = self._pad(
                self._add_specials_and_trim(
                    src, max_src_len, tokenizer.bos_id, tokenizer.eos_id),
                max_src_len,
                tokenizer.pad_id
            )

            # tokenize and encode target
            tgt = tokenizer.encode(tokenizer.tokenize(item['tgt']))
            # trim/pad and add special tokens
            tgt = self._pad(
                self._add_specials_and_trim(
                    tgt, max_tgt_len, tokenizer.bos_id, tokenizer.eos_id),
                max_tgt_len,
                tokenizer.pad_id
            )

            # processed pair
            # pair = Pair(src=' '.join(src), tgt=' '.join(tgt))
            # could remove this if taking too much mem and not needed
            # self.pairs.append(pair)

            # create training sample
            training_sample = Sample(
                enc_inp_ids=torch.LongTensor(src),
                # tgt seq shifted right by removing last token
                dec_inp_ids=torch.LongTensor(tgt[:-1]),
                # tgt seq shifted left by removing first token
                label_ids=torch.LongTensor(tgt[1:])
            )
            # store samples in a list
            self.samples.append(training_sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        return (self.samples[idx].enc_inp_ids,
                self.samples[idx].dec_inp_ids,
                self.samples[idx].label_ids)

    def _add_specials_and_trim(self, token_ids: list[int], max_len: int, bos_id: int, eos_id: int) -> list[int]:
        if len(token_ids) > max_len - 2:
            # truncate if too long
            token_ids = token_ids[:max_len - 2]
        return [bos_id] + token_ids + [eos_id]

    def _pad(self, token_ids: list[int], max_len: int, pad_id: int) -> list[int]:
        # pad if too short
        padding = [pad_id] * (max_len - len(token_ids))
        return token_ids + padding
