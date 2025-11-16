import time
from typing import Callable

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tokenizer import BOS, EOS, PAD, UNK, Tokenizer
from dataset import SeqPairDataset
from model import EncoderDecoder


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:

    model.train()
    model.to(device)
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        enc_inp_ids, dec_inp_ids, label_ids = batch
        enc_inp_ids = enc_inp_ids.to(device)
        dec_inp_ids = dec_inp_ids.to(device)
        label_ids = label_ids.to(device)

        logits = model(enc_inp_ids, dec_inp_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)),
                       label_ids.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:

    model.eval()
    model.to(device)
    total_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            enc_inp_ids, dec_inp_ids, label_ids = batch
            enc_inp_ids = enc_inp_ids.to(device)
            dec_inp_ids = dec_inp_ids.to(device)
            label_ids = label_ids.to(device)

            logits = model(enc_inp_ids, dec_inp_ids)
            loss = loss_fn(logits.view(-1, logits.size(-1)),
                           label_ids.view(-1))

            total_loss += loss.item()

    return total_loss / len(dataloader)


def compute_bleu(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    strategy: str = "greedy",
    beam_width: int = 5
):

    model.eval()
    model.to(device)

    total_bleu = 0.0
    num_samples = 0

    smoothing_fn = SmoothingFunction().method1

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing BLEU", leave=False):
            enc_inp_ids, dec_inp_ids, label_ids = batch
            enc_inp_ids = enc_inp_ids.to(device)
            dec_inp_ids = dec_inp_ids.to(device)
            label_ids = label_ids.to(device)

            model_outputs = model.generate(
                enc_inp_ids,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_len=label_ids.size(1),
                strategy=strategy,
                beam_width=beam_width
            )

            for i in range(label_ids.size(0)):
                pred_tokens = tokenizer.decode(model_outputs[i])
                filtered_pred_tokens = [
                    token for token in pred_tokens if token not in (tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)]

                print(label_ids[i])
                label_tokens = tokenizer.decode(label_ids[i])
                filtered_label_tokens = [
                    token for token in label_tokens if token not in (tokenizer.bos_id, tokenizer.eos_id, tokenizer.pad_id)]

                bleu_score = sentence_bleu(
                    filtered_label_tokens,
                    filtered_pred_tokens,
                    weights=(1.0, 0.0),
                    smoothing_function=smoothing_fn
                )

                total_bleu += bleu_score
                num_samples += 1

    return total_bleu / num_samples


@click.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('dev_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--saved_model', type=click.Path(exists=True), default=None)
def main(
    train_file: str,
    dev_file: str,
    test_file: str,
    saved_model: str | None = None
):
    # training hyperparameters
    epochs = 1
    lr = 0.001
    batch_size = 32
    max_src_len = 50
    max_tgt_len = 50
    # model hyperparameters
    d_model = 128
    num_heads = 2
    d_ff = 256
    num_enc_layers = 1
    num_dec_layers = 1
    dropout = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize tokenizer and build vocab
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)

    # create datasets and dataloaders
    train_dataset = SeqPairDataset(
        data_file=train_file,
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )
    dev_dataset = SeqPairDataset(
        data_file=dev_file,
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )
    test_dataset = SeqPairDataset(
        data_file=test_file,
        tokenizer=tokenizer,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # initialize model, loss function, optimizer
    model = EncoderDecoder(
        src_vocab_size=len(tokenizer.src_vocab),
        tgt_vocab_size=len(tokenizer.tgt_vocab),
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        max_len=max_src_len,
        dropout=dropout,
        pad_idx=tokenizer.pad_id
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # training loop
    if saved_model is not None:
        model.load_state_dict(torch.load(saved_model))
    else:

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}")

            # TODO: implement early stopping?
            # TODO: save checkpoints after each epoch?
            # TODO: print example outputs to assess quality?
            # TODO: monitor training and validation loss and create graphs?
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn, device)
            print(f"Train Loss: {train_loss:.4f}")

            dev_loss = test_epoch(
                model, dev_loader, loss_fn, device)
            print(f"Dev Loss: {dev_loss:.4f}")

        test_loss = test_epoch(
            model, test_loader, loss_fn, device)
        print(f"Test Loss: {test_loss:.4f}")

        torch.save(model.state_dict(), "test1_model.pth")

    bleu_score = compute_bleu(
        model, test_loader, tokenizer, device, strategy="greedy")
    print(f"Test BLEU Score: {bleu_score:.4f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    start_time = time.time()
    main()
    # result = torch.cuda.is_available()
    # print(f"CUDA Available: {result}")
    end_time = time.time()
    print(f"Execution Time: {end_time - start_time} seconds")
