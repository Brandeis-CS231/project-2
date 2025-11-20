from typing import Callable

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tokenizer import Tokenizer
from dataset import SeqPairDataset
from model import EncoderDecoder

def remove_spec_tokens(
    tokens: list[str],
    tokenizer: Tokenizer
) -> list[str]:
    """
    Helper method for compute_bleu_score.
    Removes BOS, EOS and PAD tokens from sentence
    """

    cleaned_tokens = []

    for token in tokens:
        if token == "<bos>" or \
        token == "<eos>" or \
        token == "<pad>":
            pass
        else:
            cleaned_tokens.append(token) 

    return cleaned_tokens   


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    
    # 1 set model to training mode
    model.train()

    # 2 move model to device
    model.to(device)

    # 3 initialize loss
    total_loss = 0.0

    # 4 iterate through batches from dataloader
    for batch_idx, (enc_inp_ids, dec_inp_ids, label_ids) in \
        enumerate(tqdm(dataloader, desc='Training')):

        # move tensors to device
        enc_inp_ids = enc_inp_ids.to(device)
        dec_inp_ids = dec_inp_ids.to(device)
        label_ids = label_ids.to(device)

        # forward pass
        logits = model(enc_inp_ids, dec_inp_ids)


        ## reshape for loss computation
        # flatten logits
        flattened_logits = torch.flatten(logits, 0, 1)

        # flatten labels
        flattened_labels = torch.flatten(label_ids, 0, -1)

        # compute loss
        loss = loss_fn(flattened_logits, flattened_labels)


        ## backward pass
        # clear previous gradients
        optimizer.zero_grad()

        # compute gradients
        loss.backward()

        # update parameters
        optimizer.step()


        # accumulate loss
        total_loss += loss.item()

        # print batch in progress every once in a while for validation
        if batch_idx % 50 == 0:
            print(f'Batch {batch_idx} processed!    '
                f'{len(enc_inp_ids)} samples trained'
            )


    # 5 return average loss
    return total_loss / len(dataloader)


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    
    # 1 set model to eval mode
    model.eval()

    # 2 move model to device
    model.to(device)

    # 3 initialize loss
    total_loss = 0.0

    # 4 disable gradient computation
    torch.no_grad()

    # 5 iterate through batches from dataloader
    for batch_idx, (enc_inp_ids, dec_inp_ids, label_ids) in \
            enumerate(tqdm(dataloader, desc='Testing')):
        
        # move tensors to device
        enc_inp_ids = enc_inp_ids.to(device)
        dec_inp_ids = dec_inp_ids.to(device)
        label_ids = label_ids.to(device)

        # forward pass
        logits = model(enc_inp_ids, dec_inp_ids)


        ## reshape for loss computation
        # flatten logits
        flattened_logits = torch.flatten(logits, 0, 1)

        # flatten labels
        flattened_labels = torch.flatten(label_ids, 0, -1)

        # compute loss
        loss = loss_fn(flattened_logits, flattened_labels)

        # accumulate loss
        total_loss += loss.item()

        # print batch size for validation
        print(f'\nBatch {batch_idx} processed!    '
              f'{len(enc_inp_ids)} samples trained'
        )


    # 5 return average loss
    return total_loss / len(dataloader)


def compute_bleu_score(
        model: nn.Module,
        dataloader: DataLoader,
        tokenizer: Tokenizer
) -> float:
    # 1 imports:
    ##### done above ######

    # 2 set model to eval mode
    model.eval()

    # 3 initialize counters total_bleu and num_samples
    total_bleu = 0
    num_samples = 0

    # 4 iterate through test batches
    for batch_idx, (enc_inp_ids, dec_inp_ids, label_ids) in \
            enumerate(tqdm(dataloader, desc='BLEU Score Calculation')):
        
        # generate sequences
        bos_id = tokenizer.bos_id
        eos_id = tokenizer.eos_id
        # dec_inp_ids has shape (batch_size, seq_len) so use seq_len
        max_len = dec_inp_ids.shape[1] + 1

        # Ensure batch dimension exists (DataLoader may return 1-D tensors)
        if enc_inp_ids.dim() == 1:
            enc_inp_ids = enc_inp_ids.unsqueeze(0)
        if dec_inp_ids.dim() == 1:
            dec_inp_ids = dec_inp_ids.unsqueeze(0)
        if label_ids.dim() == 1:
            label_ids = label_ids.unsqueeze(0)

        seqs = model.generate(
            enc_inp_ids, 
            bos_id, 
            eos_id, 
            max_len=max_len
        )

        # iterate over samples in batch to compute bleu score
        for i in range(len(seqs)):

            # Convert tensor ids to Python lists (avoid tensor type errors in decode)
            pred_ids = [int(x) for x in seqs[i]]
            true_ids = label_ids[i].tolist()

            # decode predictions into tokens
            pred_sent = tokenizer.decode(pred_ids)

            # remove special tokens (pass tokenizer so helper can check BOS/EOS/PAD)
            pred_sent = remove_spec_tokens(pred_sent, tokenizer)

            # decode ground truth, i.e. convert label ids to tokens
            true_sent = tokenizer.decode(true_ids)

            # remove specials and padding from ground truth labels
            true_sent = remove_spec_tokens(true_sent, tokenizer)

            # compute score using sentence_bleu
            smoothie = SmoothingFunction()
            batch_bleu = sentence_bleu(
                [true_sent],
                pred_sent,
                weights=(1.0,),
                smoothing_function=smoothie.method1
            )
            
            # accumulate score
            total_bleu += batch_bleu
            num_samples += 1

            ## TRACKING: occasionally print output to assess progress
            # print the last sample in the batch every 10 batches
            if batch_idx % 10 == 0 and i == len(seqs) - 1:
                print(f'Batch {batch_idx}: sample output:'
                      f'PREDICTED: {pred_sent}\n'
                      f'ACTUAL:    {true_sent}\n\n'            
                )
    
    return total_bleu / num_samples




@click.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('dev_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
def main(
    train_file: str,
    dev_file: str,
    test_file: str
):
    ## 1 define hyperparameters
    num_epochs = 3
    lr = 0.001 
    
    # TODO: experiment with values for these
    d_model = 128        # (128, 256)
    num_heads = 2       # (2, 4, 8)
    d_ff = 256          # (256, 512)
    num_enc_layers = 1  # (1, 2, 4)
    num_dec_layers = 1  # (1, 2, 4)
    
    batch_size = 256
    dropout = 0.1
    max_src_len = 50
    max_tgt_len = 50


    # 2 build tokenizer
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)

    ## 3 create datasets
    # create train, dev and test DATASETS
    train_data = SeqPairDataset(
        train_file,
        tokenizer,
        max_src_len,
        max_tgt_len
    )
    dev_data = SeqPairDataset(
        dev_file,
        tokenizer,
        max_src_len,
        max_tgt_len
    )
    test_data = SeqPairDataset(
        test_file,
        tokenizer,
        max_src_len,
        max_tgt_len
    )

    # wrap train, dev and test datasets in DATALOADERS
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True # true only for training
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False
    )


    # 5 initialize model
    pad_id = tokenizer.pad_id
    vocab_size = len(tokenizer.src_vocab)

    model = EncoderDecoder(
        src_vocab_size=vocab_size, 
        tgt_vocab_size=vocab_size,
        pad_idx=pad_id, 
        max_len=max_tgt_len, # tgt_len serves as hard stop for generation
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        dropout=dropout,
    )

    # 6 initialize optimizer and loss function
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    ## 7 training loop
    # iterate over num epochs
    for epoch in range(num_epochs):
        # TRACKING: print epoch
        print(f'\nTRAINING AND VALIDATION\n'
              f'----epoch:       {epoch}'
        )
        
        # run training
        tr_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn
        )
        # TRACKING: print train loss
        print(f'Training loss:   {tr_loss}')

        # run validation
        val_loss = test_epoch(
            model=model,
            dataloader=dev_loader,
            loss_fn=loss_fn
        )
        # TRACKING: print validation loss
        print(f'Validation loss: {val_loss}\n')

    # save model checkpoints
    torch.save(model.state_dict(), 'checkpoint.pt')

    ## 8 final evaluation
    # run testing
    test_loss = test_epoch(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn
    )
    # TRACKING: print test loss
    print(f'Test loss: {test_loss}\n')


    # evaluate BLEU score
    bleu_score = compute_bleu_score(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer
    )
    # TRACKING: print BLEU score
    print(f'***BLEU SCORE:  {bleu_score}***\n\n')

    # TRACKING: print hyperparameter config
    print("TESTING COMPLETE!")
    print(f'-----------Hyperparameters----------\n'
            f'    model depth (enc layers): {num_enc_layers}\n'
            f'    model dimensions:         {d_model}\n'
            f'    attention heads:          {num_heads}\n'
            f'    feedforward dimension:    {d_ff}\n'
            f'--------------END-----------------\n\n\n'
    )

    

    

if __name__ == "__main__":
    main()
