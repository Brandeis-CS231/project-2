from typing import Callable
import os
import json

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tokenizer import Tokenizer
from model import EncoderDecoder
from dataset import SeqPairDataset, Sample


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
    step = 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # batch is tuple (enc_inp, dec_inp, labels) because dataset returns that
        enc_inp, dec_inp, labels = batch
        enc_inp = enc_inp.to(device)
        dec_inp = dec_inp.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(enc_inp, dec_inp)  # (B, tgt_len, vocab)
        B, T, V = logits.shape
        loss = loss_fn(logits.view(B * T, V), labels.view(B * T))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

    return total_loss / max(1, step)


def test_epoch(
        model: nn.Module,
        dataloader: DataLoader,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        tokenizer: Tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    model.eval()
    model.to(device)
    total_loss = 0.0
    step = 0
    smooth = SmoothingFunction().method1
    total_bleu = 0.0
    n_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            enc_inp, dec_inp, labels = batch
            enc_inp = enc_inp.to(device)
            dec_inp = dec_inp.to(device)
            labels = labels.to(device)

            logits = model(enc_inp, dec_inp)
            B, T, V = logits.shape
            loss = loss_fn(logits.view(B * T, V), labels.view(B * T))
            total_loss += loss.item()
            step += 1

            # generation and BLEU scoring (batch element-wise)
            gen_batches = model.generate(enc_inp, bos_id=tokenizer.bos_id, eos_id=tokenizer.eos_id, max_len=tokenizer.idx2word and T or T)
            # model.generate returns list of lists for each batch item
            for i in range(len(gen_batches)):
                gen_ids = gen_batches[i]
                # decode and strip specials
                pred_tokens = tokenizer.decode([int(x) for x in gen_ids])
                pred_tokens = [t for t in pred_tokens if t not in (tokenizer.config.special_tokens)]
                # reference tokens: labels[i] (ids), remove pads and specials
                ref_ids = labels[i].cpu().tolist()
                ref_ids = [rid for rid in ref_ids if rid != tokenizer.pad_id]
                ref_tokens = tokenizer.decode(ref_ids)
                ref_tokens = [t for t in ref_tokens if t not in (tokenizer.config.special_tokens)]
                # BLEU unigram
                try:
                    bleu = sentence_bleu([ref_tokens], pred_tokens, weights=(1.0, 0, 0, 0), smoothing_function=smooth)
                except Exception:
                    bleu = 0.0
                total_bleu += bleu
                n_samples += 1

    avg_loss = total_loss / max(1, step)
    avg_bleu = total_bleu / max(1, n_samples)
    print(f"Validation Loss={avg_loss:.4f} | BLEU(unigram)={avg_bleu:.4f}")
    return avg_loss


@click.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('dev_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--epochs', default=10, type=int, help='Number of training epochs')
@click.option('--d_model', default=128, type=int, help='Model embedding dimension')
@click.option('--num_layers', default=2, type=int, help='Number of encoder/decoder layers')
@click.option('--num_heads', default=4, type=int, help='Number of attention heads')
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
def main(
        train_file: str,
        dev_file: str,
        test_file: str,
        epochs: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        device: str,
):
    """
    Minimal runnable training script. Adjust hyperparameters inside as needed.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # hyperparams (small by default)
    epochs = 5
    lr = 1e-3
    batch_size = 16
    max_src_len = 50
    max_tgt_len = 50

    # model hyperparams
    d_model = 128
    num_heads = 4
    d_ff = 256
    num_enc_layers = 2
    num_dec_layers = 2
    dropout = 0.1

    # Build tokenizer (from train file)
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)
    print("Vocab size:", len(tokenizer.src_vocab))

    # Datasets / loaders
    train_ds = SeqPairDataset(train_file, tokenizer, max_src_len, max_tgt_len)
    dev_ds = SeqPairDataset(dev_file, tokenizer, max_src_len, max_tgt_len)
    test_ds = SeqPairDataset(test_file, tokenizer, max_src_len, max_tgt_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = EncoderDecoder(
        src_vocab_size=len(tokenizer.src_vocab),
        tgt_vocab_size=len(tokenizer.src_vocab),  # shared vocab
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        max_len=max(max_src_len, max_tgt_len),
        dropout=dropout,
        pad_idx=tokenizer.pad_id
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    best_dev = float("inf")
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        print(f"Train loss: {train_loss:.4f}")
        dev_loss = test_epoch(model, dev_loader, loss_fn, tokenizer, device)

        if dev_loss < best_dev:
            best_dev = dev_loss
            ckpt = "best_model.pt"
            torch.save(model.state_dict(), ckpt)
            print(f"Saved checkpoint to {ckpt}")

    # final test (loads best if available)
    if os.path.exists("best_model.pt"):
        model.load_state_dict(torch.load("best_model.pt", map_location=device))
    print("\nFinal test evaluation:")
    _ = test_epoch(model, test_loader, loss_fn, tokenizer, device)


if __name__ == "__main__":
    main()
