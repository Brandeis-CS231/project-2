from typing import Callable

from dataset import SeqPairDataset
from tokenizer import Tokenizer
from model import EncoderDecoder
import click
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


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

    for enc_ids, dec_ids, label_ids in tqdm(dataloader, desc='Training'):
        enc_ids = enc_ids.to(device)
        dec_ids = dec_ids.to(device)
        label_ids = label_ids.to(device)
        logits = model(enc_ids, dec_ids)
        flat_logits = torch.flatten(logits, start_dim=0, end_dim=1)
        flat_labels = torch.flatten(label_ids)
        loss = loss_fn(flat_logits, flat_labels)
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
        for enc_ids, dec_ids, label_ids in tqdm(dataloader, desc='Testing'):
            enc_ids = enc_ids.to(device)
            dec_ids = dec_ids.to(device)
            label_ids = label_ids.to(device)
            logits = model(enc_ids, dec_ids)
            flat_logits = torch.flatten(logits, start_dim=0, end_dim=1)
            flat_labels = torch.flatten(label_ids)
            loss = loss_fn(flat_logits, flat_labels)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def compute_bleu(
        model: EncoderDecoder,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        max_len: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        strategy: str = "greedy"
):

    model.eval()
    model.to(device)
    total_bleu = 0.0
    num_samples = 0
    total_time = 0.0
    total_seq_len = 0.0
    bos_id = tokenizer.bos_id
    eos_id = tokenizer.eos_id
    smoothing = SmoothingFunction()

    with torch.no_grad():
        for enc_ids, dec_ids, label_ids in tqdm(dataloader, desc='Computing BLEU'):
            enc_ids = enc_ids.to(device)
            dec_ids = dec_ids.to(device)
            label_ids = label_ids.to(device)
            start = time.process_time()
            preds = model.generate(src_ids=enc_ids, bos_id=bos_id, eos_id=eos_id, max_len=max_len, strategy=strategy)
            end = time.process_time()
            total_time += end - start
            for i, pair in enumerate(zip(preds, label_ids.tolist())):
                pred, ref = pair
                pred = tokenizer.decode(pred)
                pred = [p for p in pred if p != '<bos>' and p != '<eos>' and p != '<pad>']
                total_seq_len += len(pred)
                ref = tokenizer.decode(ref)
                ref = [l for l in ref if l != '<bos>' and l != '<eos>' and l != '<pad>']
                bleu = sentence_bleu([ref], pred, weights=(1.0, 0.0, 0.0, 0.0), smoothing_function=smoothing.method4)
                total_bleu += bleu
                num_samples += 1
                if num_samples % 1400 == 0:
                    with open("results/exp3b.txt", "a") as f:
                        f.write(f"pred: {' '.join(pred)}\n")
                        f.write(f"labels: {' '.join(ref)}\n")

    with open("results/exp3b.txt", "a") as f:
        f.write(f"Avg. time per sample: {total_time / num_samples:.5f}\n")
        # f.write(f"Avg. seq len: {total_seq_len / num_samples:.5f}\n")

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
    torch.manual_seed(42)

    epochs = 3
    lr = 1e-3
    batch_size = 64
    max_len = 50

    # d_model = [128, 256]
    d_m = 256
    # num_heads = [2, 4, 8]
    n_h = 4
    # d_ff = [256, 512]
    d_f = 512
    # num_layers = [1, 2, 4]
    # n_l = 4
    dropout = 0.1

    tok = Tokenizer()
    tok.from_file(train_file)

    train_data = SeqPairDataset(train_file, tok, max_len, max_len)
    dev_data = SeqPairDataset(dev_file, tok, max_len, max_len)
    test_data = SeqPairDataset(test_file, tok, max_len, max_len)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    src_size = len(tok.src_vocab)
    tgt_size = len(tok.tgt_vocab)
    pad_id = tok.pad_id

    model1 = EncoderDecoder(
        src_vocab_size=src_size,
        tgt_vocab_size=tgt_size,
        d_model=d_m,
        num_heads=n_h,
        d_ff=d_f,
        num_enc_layers=1,
        num_dec_layers=1,
        max_len=max_len,
        dropout=dropout,
        pad_idx=pad_id
    )

    model2 = EncoderDecoder(
        src_vocab_size=src_size,
        tgt_vocab_size=tgt_size,
        d_model=d_m,
        num_heads=n_h,
        d_ff=d_f,
        num_enc_layers=2,
        num_dec_layers=2,
        max_len=max_len,
        dropout=dropout,
        pad_idx=pad_id
    )

    model3 = EncoderDecoder(
        src_vocab_size=src_size,
        tgt_vocab_size=tgt_size,
        d_model=d_m,
        num_heads=n_h,
        d_ff=d_f,
        num_enc_layers=4,
        num_dec_layers=4,
        max_len=max_len,
        dropout=dropout,
        pad_idx=pad_id
    )

    opt1 = torch.optim.Adam(model1.parameters(), lr=lr)
    opt2 = torch.optim.Adam(model2.parameters(), lr=lr)
    opt3 = torch.optim.Adam(model3.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    m1_train_loss = []
    m2_train_loss = []
    m3_train_loss = []
    m1_dev_loss = []
    m2_dev_loss = []
    m3_dev_loss = []
    m1_params = sum(p.numel() for p in model1.parameters())
    m2_params = sum(p.numel() for p in model2.parameters())
    m3_params = sum(p.numel() for p in model3.parameters())

    with open("results/exp3b.txt", "a") as f:
        f.write(f"{'-' * 30}\n")
        f.write(f"Epochs: {epochs} | LR: {lr} | Batch size: {batch_size} | Max seq. len.: {max_len}\n")
        f.write(f"Model dim.: {d_m} | Heads: {n_h} | FF dim.: {d_f} | Dropout: {dropout}\n")
        f.write(f"M1 params: {m1_params} | M2 params: {m2_params} | M3 params: {m3_params}\n")

    for i in range(epochs):
        print(f"Epoch {i + 1}\n{'-' * 30}", flush=True)
        m1_avg_train_loss = train_epoch(model=model1, dataloader=train_loader, optimizer=opt1, loss_fn=loss_fn)
        m1_train_loss.append(m1_avg_train_loss)
        m2_avg_train_loss = train_epoch(model=model2, dataloader=train_loader, optimizer=opt2, loss_fn=loss_fn)
        m2_train_loss.append(m2_avg_train_loss)
        m3_avg_train_loss = train_epoch(model=model3, dataloader=train_loader, optimizer=opt3, loss_fn=loss_fn)
        m3_train_loss.append(m3_avg_train_loss)
        print(f"M1 train loss: {m1_avg_train_loss:.5f} | M2 train loss: {m2_avg_train_loss:.5f} | M3 train loss: {m3_avg_train_loss:.5f}\n", flush=True)
        with open("results/exp3b.txt", "a") as f:
            f.write(f"Epoch {i + 1} | M1 train loss: {m1_avg_train_loss:.5f} | M2 train loss: {m2_avg_train_loss} | M3 train loss: {m3_avg_train_loss:.5f}\n")

        print('-----\nTesting on dev set...', flush=True)
        m1_avg_dev_loss = test_epoch(model=model1, dataloader=dev_loader, loss_fn=loss_fn)
        m1_dev_loss.append(m1_avg_dev_loss)
        m2_avg_dev_loss = test_epoch(model=model2, dataloader=dev_loader, loss_fn=loss_fn)
        m2_dev_loss.append(m2_avg_dev_loss)
        m3_avg_dev_loss = test_epoch(model=model3, dataloader=dev_loader, loss_fn=loss_fn)
        m3_dev_loss.append(m3_avg_dev_loss)
        print(f"M1 dev loss: {m1_avg_dev_loss:.5f} | M2 dev loss: {m2_avg_dev_loss:.5f} | M3 dev loss: {m3_avg_dev_loss:.5f}\n", flush=True)
        with open("results/exp3b.txt", "a") as f:
            f.write(f"Epoch {i + 1} | M1 dev loss: {m1_avg_dev_loss:.5f} | M2 dev loss: {m2_avg_dev_loss} | M3 dev loss: {m3_avg_dev_loss:.5f}\n")

    # with open("models/exp2", 'wb') as f:
    #     pickle.dump(model1, f)

    plt.plot(m1_train_loss, label="Layers = 1")
    plt.plot(m2_train_loss, label="Layers = 2")
    plt.plot(m3_train_loss, label="Layers = 4")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("results/exp3b_train.png")
    plt.close()

    plt.plot(m1_dev_loss, label="Layers = 1")
    plt.plot(m2_dev_loss, label="Layers = 2")
    plt.plot(m3_dev_loss, label="Layers = 4")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Validation Loss")
    plt.legend()
    plt.savefig("results/exp3b_dev.png")
    plt.close()

    print('-----\nTesting on test set...', flush=True)
    with open("results/exp3b.txt", "a") as f:
        f.write(f"M1 example outputs:\n")
    bleu1 = compute_bleu(model=model1, dataloader=test_loader, tokenizer=tok, max_len=max_len)
    with open("results/exp3b.txt", "a") as f:
        f.write(f"M2 example outputs:\n")
    bleu2 = compute_bleu(model=model2, dataloader=test_loader, tokenizer=tok, max_len=max_len)
    with open("results/exp3b.txt", "a") as f:
        f.write(f"M3 example outputs:\n")
    bleu3 = compute_bleu(model=model3, dataloader=test_loader, tokenizer=tok, max_len=max_len)
    print(f"M1 BLEU score: {bleu1:.5f} | M2 BLEU score: {bleu2:.5f} | M3 BLEU score: {bleu3:.5f}\n", flush=True)
    with open("results/exp3b.txt", "a") as f:
        f.write(f"M1 BLEU score: {bleu1:.5f} | M2 BLEU score: {bleu2:.5f} | M3 BLEU score: {bleu3:.5f}\n")
        f.write(f"{'-' * 30}\n")

    # The code below was used for grid search.

    # for d_m in d_model:
    #     for n_h in num_heads:
    #         for d_f in d_ff:
    #             for n_l in num_layers:
    #                 model = EncoderDecoder(
    #                     src_vocab_size=src_size,
    #                     tgt_vocab_size=tgt_size,
    #                     d_model=d_m,
    #                     num_heads=n_h,
    #                     d_ff=d_f,
    #                     num_enc_layers=n_l,
    #                     num_dec_layers=n_l,
    #                     max_len=max_len,
    #                     dropout=dropout,
    #                     pad_idx=pad_id
    #                 )
    #
    #                 opt = torch.optim.Adam(model.parameters(), lr=lr)
    #                 # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)
    #
    #                 with open("results/grid_search_2.txt", "a") as f:
    #                     f.write(f"{'-' * 30}\n")
    #                     f.write(f"Epochs: {epochs} | LR: {lr} | Batch size: {batch_size} | Max seq. len.: {max_len}\n")
    #                     f.write(f"Model depth: {n_l} | Model dim.: {d_m} | Att. heads: {n_h} | FF dim.: {d_f} | Dropout: {dropout}\n")
    #
    #                 for i in range(epochs):
    #                     print(f"Epoch {i + 1}\n{'-' * 30}")
    #                     avg_train_loss = train_epoch(model=model, dataloader=train_loader, optimizer=opt, loss_fn=loss_fn)
    #                     print(f"Avg. train loss: {avg_train_loss:.5f}")
    #                     with open("results/grid_search_2.txt", "a") as f:
    #                         f.write(f"Epoch {i + 1} train loss: {avg_train_loss:.5f}\n")
    #
    #                     print('-----\nTesting on dev set...')
    #                     avg_dev_loss = test_epoch(model=model, dataloader=dev_loader, loss_fn=loss_fn)
    #                     print(f"Avg. dev loss: {avg_dev_loss:.5f}")
    #                     with open("results/grid_search_2.txt", "a") as f:
    #                         f.write(f"Epoch {i + 1} dev loss: {avg_dev_loss:.5f}\n")

                    # print('-----\nTesting on test set...')
                    # bleu = compute_bleu(model=model, dataloader=test_loader, tokenizer=tok, max_len=max_len)
                    # print(f"BLEU score: {bleu:.5f}")
                    # with open("results/grid_search.txt", "a") as f:
                    #     f.write(f"BLEU score: {bleu:.5f}\n")
                    #     f.write(f"{'-' * 30}\n")

if __name__ == "__main__":
    main()