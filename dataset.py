"""
This module implements class(es) and function(s) for dataset representation
"""
import json
from typing import Tuple, List
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
        """
        Initialize the dataset by loading and preprocessing sequence pairs.

        Args:
            data_file: Path to JSON file containing sequence pairs
            tokenizer: A built Tokenizer instance
            max_src_len: Maximum length for source sequences (including special tokens)
            max_tgt_len: Maximum length for target sequences (including special tokens)
        """
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.samples = []

        # Load JSON data
        with open(data_file, 'r') as f:
            data = json.load(f)

        # Process each sequence pair
        for item in data:
            # Create and validate pair
            pair = Pair(src=item['src'], tgt=item['tgt'])

            # Tokenize source and target
            src_tokens = Tokenizer.tokenize(pair.src)
            tgt_tokens = Tokenizer.tokenize(pair.tgt)

            # Encode tokens to IDs
            src_ids = self.tokenizer.encode(src_tokens)
            tgt_ids = self.tokenizer.encode(tgt_tokens)

            # Add special tokens and trim if needed
            src_ids = self._add_specials_and_trim(src_ids, self.max_src_len)
            tgt_ids = self._add_specials_and_trim(tgt_ids, self.max_tgt_len)

            # Pad sequences to max length
            src_ids = self._pad(src_ids, self.max_src_len)
            tgt_ids = self._pad(tgt_ids, self.max_tgt_len)

            # Create training pairs for teacher forcing
            # Encoder input: full padded source sequence
            encoder_input = torch.LongTensor(src_ids)

            # Decoder input: target sequence with last token removed (shifted right)
            decoder_input = torch.LongTensor(tgt_ids[:-1] + [self.tokenizer.pad_id])

            # Labels: target sequence with first token removed (shifted left)
            labels = torch.LongTensor(tgt_ids[1:] + [self.tokenizer.pad_id])

            # Store sample
            sample = Sample(
                enc_inp_ids=encoder_input,
                dec_inp_ids=decoder_input,
                label_ids=labels
            )
            self.samples.append(sample)

    def _add_specials_and_trim(self, token_ids: List[int], max_len: int) -> List[int]:
        """
        Add <bos> and <eos> tokens, and trim if the sequence exceeds max_len.

        Args:
            token_ids: List of token IDs
            max_len: Maximum sequence length (including special tokens)

        Returns:
            List of token IDs with special tokens added and trimmed if needed
        """
        # Reserve space for <bos> and <eos>
        max_content_len = max_len - 2

        # Trim if needed
        if len(token_ids) > max_content_len:
            token_ids = token_ids[:max_content_len]

        # Add <bos> at the beginning and <eos> at the end
        return [self.tokenizer.bos_id] + token_ids + [self.tokenizer.eos_id]

    def _pad(self, token_ids: List[int], max_len: int) -> List[int]:
        """
        Pad sequence to target length with <pad> tokens.

        Args:
            token_ids: List of token IDs
            max_len: Target sequence length

        Returns:
            Padded list of token IDs
        """
        padding_len = max_len - len(token_ids)
        return token_ids + [self.tokenizer.pad_id] * padding_len

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        """
        Return a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (encoder_input_ids, decoder_input_ids, label_ids)
        """
        sample = self.samples[idx]
        return sample.enc_inp_ids, sample.dec_inp_ids, sample.label_ids

