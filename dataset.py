"""
This module implements class(es) and function(s) for dataset representation
"""
from typing import Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from tokenizer import Pair, Tokenizer
import json


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
        Initialize the dataset.

        Args:
            data_file: Path to JSON file with sequence pairs
            tokenizer: Built tokenizer instance with vocabulary
            max_src_len: Maximum source sequence length (including special tokens)
            max_tgt_len: Maximum target sequence length (including special tokens)
        """
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.samples = []

        # Load and process data
        with open(data_file, 'r') as f:
            data = json.load(f)

        for item in data:
            # Create and validate pair
            pair = Pair(src=item['src'], tgt=item['tgt'])

            # Tokenize source and target
            src_tokens = self.tokenizer.tokenize(pair.src)
            tgt_tokens = self.tokenizer.tokenize(pair.tgt)

            # Encode to IDs
            src_ids = self.tokenizer.encode(src_tokens)
            tgt_ids = self.tokenizer.encode(tgt_tokens)

            # Add special tokens and trim
            src_ids = self._add_specials_and_trim(src_ids, self.max_src_len)
            tgt_ids = self._add_specials_and_trim(tgt_ids, self.max_tgt_len)

            # Pad to max lengths
            src_ids = self._pad(src_ids, self.max_src_len)
            tgt_ids = self._pad(tgt_ids, self.max_tgt_len)

            # Create training samples with teacher forcing
            # Encoder input: full source with BOS and EOS
            enc_inp_ids = torch.LongTensor(src_ids)

            # Decoder input: target shifted right (remove last token)
            # This means: [BOS, tok1, tok2, ..., tokN-1]
            dec_inp_ids = torch.LongTensor(tgt_ids[:-1])

            # Labels: target shifted left (remove first token)
            # This means: [tok1, tok2, ..., tokN, EOS]
            label_ids = torch.LongTensor(tgt_ids[1:])

            # Store the sample
            sample = Sample(
                enc_inp_ids=enc_inp_ids,
                dec_inp_ids=dec_inp_ids,
                label_ids=label_ids
            )
            self.samples.append(sample)

    def _add_specials_and_trim(self, token_ids: list, max_len: int) -> list:
        """
        Add BOS and EOS tokens, and trim if necessary.

        Args:
            token_ids: List of token IDs
            max_len: Maximum length including special tokens

        Returns:
            List with BOS prepended and EOS appended, trimmed to max_len
        """
        # Reserve 2 positions for BOS and EOS
        max_content_len = max_len - 2

        # Trim if needed
        if len(token_ids) > max_content_len:
            token_ids = token_ids[:max_content_len]

        # Add BOS at start and EOS at end
        result = [self.tokenizer.bos_id] + token_ids + [self.tokenizer.eos_id]

        return result

    def _pad(self, token_ids: list, max_len: int) -> list:
        """
        Pad sequence to target length with PAD tokens.

        Args:
            token_ids: List of token IDs
            max_len: Target length

        Returns:
            Padded list of length max_len
        """
        # Calculate how much padding is needed
        padding_needed = max_len - len(token_ids)

        # Add padding tokens to the end
        result = token_ids + [self.tokenizer.pad_id] * padding_needed

        return result

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        """
        Get a single sample.

        Args:
            idx: Index of the sample

        Returns:
            Tuple of (encoder_input_ids, decoder_input_ids, label_ids)
        """
        sample = self.samples[idx]
        return sample.enc_inp_ids, sample.dec_inp_ids, sample.label_ids


