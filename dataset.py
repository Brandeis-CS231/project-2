"""
This module implements class(es) and function(s) for dataset representation
"""
from typing import Tuple, List
from dataclasses import dataclass
import json

import torch
from torch.utils.data import Dataset

from tokenizer import Tokenizer, Pair


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
        """
        Args:
            data_file: path to json file containing list of {"src", "tgt"}
            tokenizer: built Tokenizer (call tokenizer.from_file(train_file) before)
            max_src_len: including BOS/EOS
            max_tgt_len: including BOS/EOS
        """
        self.tokenizer = tokenizer
        self.max_src_len = int(max_src_len)
        self.max_tgt_len = int(max_tgt_len)

        # load json
        with open(data_file, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        self.samples: List[Sample] = []
        for entry in raw:
            # validate using Pair dataclass (will raise on bad types)
            p = Pair(src=entry["src"], tgt=entry["tgt"])

            # tokenize and encode
            src_tokens = tokenizer.tokenize(p.src)
            tgt_tokens = tokenizer.tokenize(p.tgt)

            src_ids = tokenizer.encode(src_tokens)
            tgt_ids = tokenizer.encode(tgt_tokens)

            # add specials and trim
            src_proc = self._add_specials_and_trim(src_ids, self.max_src_len)
            tgt_proc = self._add_specials_and_trim(tgt_ids, self.max_tgt_len)

            # pad to exact lengths
            src_pad = self._pad(src_proc, self.max_src_len)
            tgt_pad = self._pad(tgt_proc, self.max_tgt_len)

            # Create decoder_input (teacher forcing): decoder input is target shifted right,
            # i.e., starts with BOS and excludes the final EOS (so same length as target)
            # Labels are target shifted left, i.e., exclude BOS and include EOS.
            # We will keep decoder_input and labels same length = max_tgt_len.

            # decoder input: keep first max_tgt_len tokens of tgt_pad (already BOS..EOS..PAD)
            decoder_input = tgt_pad.copy()
            # labels: shift left by 1, pad end with PAD to keep length
            labels = tgt_pad[1:] + [tokenizer.pad_id]

            assert len(decoder_input) == self.max_tgt_len
            assert len(labels) == self.max_tgt_len

            enc_tensor = torch.LongTensor(src_pad)
            dec_tensor = torch.LongTensor(decoder_input)
            lab_tensor = torch.LongTensor(labels)

            self.samples.append(Sample(enc_tensor, dec_tensor, lab_tensor))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        return s.enc_inp_ids, s.dec_inp_ids, s.label_ids

    def _add_specials_and_trim(self, token_ids: List[int], max_len: int) -> List[int]:
        """
        Prepend BOS and append EOS. If length exceeds max_len, truncate middle tokens.
        """
        ids = [self.tokenizer.bos_id] + token_ids + [self.tokenizer.eos_id]
        if len(ids) > max_len:
            # keep BOS and EOS, truncate the token_ids middle to fit
            keep_mid = max_len - 2
            ids = [self.tokenizer.bos_id] + token_ids[:keep_mid] + [self.tokenizer.eos_id]
        return ids

    def _pad(self, token_ids: List[int], max_len: int) -> List[int]:
        if len(token_ids) >= max_len:
            return token_ids[:max_len]
        return token_ids + [self.tokenizer.pad_id] * (max_len - len(token_ids))


