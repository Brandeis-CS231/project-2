"""
This module implements class(es) and function(s) for dataset representation
"""
from typing import Tuple
from dataclasses import dataclass
import json
import torch
from torch.utils.data import Dataset
from tokenizer import Pair, Tokenizer
from tqdm import tqdm


@dataclass
class Sample:
    enc_inp_ids: torch.LongTensor
    dec_inp_ids: torch.LongTensor
    label_ids: torch.LongTensor


class SeqPairDataset(Dataset):
    def __init__(
            self,
            data_file: str,
            tokenizer: Tokenizer, # vocabulary already created
            max_src_len: int,
            max_tgt_len: int
    ):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.vocab_size = len(tokenizer.word2idx)
        
        # Load the JSON data from data_file:
        with open(data_file, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for line in data:
            
            # create 'Pair' to validate data:
            pair = Pair(line['tgt'],line['src'])
            # tokenize src and tgt 
            src_tokens = tokenizer.tokenize(line['src'])
            tgt_tokens = tokenizer.tokenize(line['tgt'])
            # encode src and tgt
            encoded_src = tokenizer.encode(src_tokens)
            encoded_tgt = tokenizer.encode(tgt_tokens)
            # trim and add special tokens
            src_size = self.max_src_len-2
            tgt_size = self.max_tgt_len-2
            
            if len(encoded_src) > src_size:
                encoded_src = encoded_src[:src_size]
            if len(encoded_tgt) > tgt_size:
                encoded_tgt = encoded_tgt[:tgt_size]
                
            src = [tokenizer.bos_id]+encoded_src+[tokenizer.eos_id]
            tgt = [tokenizer.bos_id]+encoded_tgt+[tokenizer.eos_id]
            
            while len(src) < self.max_src_len:
                src.append(tokenizer.pad_id)
            while len(tgt) < self.max_tgt_len:
                tgt.append(tokenizer.pad_id)
            
            if len(src) != self.max_src_len or len(tgt)!= self.max_tgt_len:
                print("YOU FUCKED UP BIG TIME")
            
            # encoder_input is full padded src sequence
            encoder_input = torch.tensor(src)
            
            # decoder_input is tgt shifted right by removing last token
            decoder_input = torch.tensor(tgt[:-1])
            
            # labels is tgt shifted left by removing first token 
            labels = torch.tensor(tgt[1:])
            
            self.samples.append((encoder_input,decoder_input,labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        return self.samples[idx]
        

