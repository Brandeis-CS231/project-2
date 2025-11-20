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
    def _add_specials_and_trim(self, token_ids, max_len):
        # trim if sentence too long before adding special tokens
        if len(token_ids) > max_len - 2:
            token_ids = token_ids[:(max_len - 2)]
            print(f"Excessive sentence trimmed to: {len(token_ids)} "
                  f"prior to adding <BOS> and <EOS>")



        # add special tokens for beginning and end of sentence
        token_ids.insert(0, self.bos_id)
        token_ids.append(self.eos_id)
        print(f"Special tokens added:\n{token_ids[:10]}...\n")

        return token_ids
    
    def _pad(self, token_ids, max_len):
        # pad if token_ids < max length
        if len(token_ids) < max_len:
            pad_seq = [self.pad_id for i in range (max_len - len(token_ids))]
            token_ids = token_ids + pad_seq
        
        print(f"Sentence length after padding: {len(token_ids)}")
        
        # ensure padded length = max length
        if len(token_ids) != max_len:
            raise ValueError(f"Error: token sequence does not match " 
                             f"max length after padding!"
            )
        
        print(f"Padding tokens added:\n...{token_ids[-10:]}\n")

        return token_ids
        
    
    def __init__(
            self,
            data_file: str,
            tokenizer: Tokenizer,
            max_src_len: int,
            max_tgt_len: int
    ):

        self.pad_id = tokenizer.pad_id
        self.bos_id = tokenizer.bos_id
        self.eos_id = tokenizer.eos_id

        print("Success: vocabulary built!\n")
        
        # create pairs of sentence strings
        with open(data_file, 'r', encoding='utf-8') as f:
            json_str = f.read()
        str_dicts = json.loads(json_str)
        pairs = [Pair(entry['tgt'],entry['src']) for entry in str_dicts]
        
         
        # initialize samples list as attribute
        self.samples = []

        for pair in pairs:
            # tokenize string pairs
            src_tok = tokenizer.tokenize(pair.src)
            tgt_tok = tokenizer.tokenize(pair.tgt)
            print(f"Sentence tokenized!\n"
                  f"src: {src_tok}\n"
                  f"tgt: {tgt_tok}\n\n"
            )

            # encode tokens as IDs
            src = tokenizer.encode(src_tok)
            tgt = tokenizer.encode(tgt_tok)


            ### add special and padding tokens, trim pairs to max length
            ## source
            src = self._add_specials_and_trim(src, max_src_len)
            
            # ensure trimmed sentences do not exceed max length
            if len(src) > max_src_len:
                raise ValueError(f"Error: sentence length exceeds "
                                f"max length after trimming: {len(src)}")
            
            src = self._pad(src, max_src_len)
            
            ## target
            tgt = self._add_specials_and_trim(tgt, max_tgt_len)

            # ensure trimmed sentences do not exceed max length
            if len(src) > max_src_len:
                raise ValueError(f"Error: sentence length exceeds "
                                f"max length after trimming")

            tgt = self._pad(tgt, max_tgt_len)


            ## create samples of enc_inp, dec_inp and labels
            # encoder input: full source sequence
            enc_inp = torch.tensor(src, dtype=torch.long)

            # decoder input: tgt seq shifted RIGHT (last token removed)
            dec_inp = torch.tensor(tgt[:-1], dtype=torch.long)
            print(f'Decoder input last 3: {dec_inp[-3:]}\n'
                  f'Target last 3       : {tgt[-3:]}\n'
                  f'Decoder input size  : {len(dec_inp)}\n'
            )

            #

            # labels: tgt seq shifted LEFT (first token removed)
            labels = torch.tensor(tgt[1:], dtype=torch.long)
            print(f'Labels first 3: {labels[:3]}\n'
                  f'Target first 3: {tgt[:3]}\n'
                  f'Labels size   : {len(labels)}\n'
            )

            # store in Sample object
            sample = Sample(enc_inp, dec_inp, labels)
            self.samples.append(sample)
            print(f'New sample created:\n'
                  f'    enc_inp size: {len(enc_inp)}'
                  f'    dec_inp size: {len(dec_inp)}'
                  f'    labels size : {len(labels)}\n'
                  f'    enc_inp type: {type(enc_inp)}\n'
                  f'    dec_inp type: {type(dec_inp)}\n'
                  f'    labels  type: {type(labels)}\n'
                  f'-----------------------------------------------\n\n'
            )
        
        # 

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, ...]:
        enc_inp = self.samples[idx].enc_inp_ids
        dec_inp = self.samples[idx].dec_inp_ids
        labels = self.samples[idx].label_ids

# enc_inp_ids: torch.LongTensor
#     dec_inp_ids: torch.LongTensor
#     label_ids: torch.

        return (enc_inp, dec_inp, labels)

