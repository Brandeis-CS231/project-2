import uuid
from typing import Literal, Self

import timeit
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from project_2.dataset import SeqPairDataset
from project_2.model import EncoderDecoder
from project_2.tokenizer import SPECIAL_TOKENS, Tokenizer, TokenizerConfig


class Trainer:
    device: str
    train_file: str
    dev_file: str
    test_file: str
    epochs: int
    learning_rate: float
    batch_size: int
    max_src_len: int
    max_tgt_len: int
    d_model: int
    num_heads: int
    d_ff: int
    num_enc_layers: int
    num_dec_layers: int
    dropout: float
    strategy: Literal["greedy", "beam_search"]
    beam_width: int

    train_loss: float = torch.inf
    dev_loss: float = torch.inf
    test_loss: float = torch.inf
    test_gen_time: float = torch.inf

    total_bleu: int = 0
    num_samples: int = 0

    def __init__(
        self,
        device: str,
        train_file: str,
        dev_file: str,
        test_file: str,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        max_src_len: int,
        max_tgt_len: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_enc_layers: int,
        num_dec_layers: int,
        dropout: float,
        strategy: Literal["greedy", "beam_search"],
        beam_width: int,
    ) -> None:
        self.device = device
        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dropout = dropout
        self.strategy = strategy
        self.beam_width = beam_width

        self.uuid = uuid.uuid4()
        self.test_gen_time = torch.inf

        self.train_loss_by_epoch: list[float] = []
        self.dev_loss_by_epoch: list[float] = []

        self.tokenizer = Tokenizer(config=TokenizerConfig()).from_file(train_file)

        train_dataset = SeqPairDataset(
            train_file, self.tokenizer, max_src_len, max_tgt_len
        )
        dev_dataset = SeqPairDataset(dev_file, self.tokenizer, max_src_len, max_tgt_len)
        test_dataset = SeqPairDataset(
            test_file, self.tokenizer, max_src_len, max_tgt_len
        )

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        self.dev_dataloader = DataLoader(
            dev_dataset, batch_size=batch_size, shuffle=False
        )
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False
        )

        self.model = EncoderDecoder(
            src_vocab_size=self.tokenizer.src_vocab_size,
            tgt_vocab_size=self.tokenizer.tgt_vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_enc_layers=num_enc_layers,
            num_dec_layers=num_dec_layers,
            dropout=dropout,
        )

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_id)

    def fit(self) -> Self:
        for epoch in range(self.epochs):
            print(f"Epoch: {epoch}")
            self._train()
            self._dev()

            print(f"Train Loss: {self.train_loss}\tDev Loss: {self.dev_loss}")

        return self

    def _train(self) -> None:
        self.model.train()
        self.model.to(self.device)

        total_loss = 0.0
        torch.autograd.set_detect_anomaly(True)

        for batch, (encoder_input_ids, decoder_input_ids, labels) in enumerate(
            tqdm(self.train_dataloader, "Training")
        ):
            encoder_input_ids = encoder_input_ids.to(self.device)
            decoder_input_ids = decoder_input_ids.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(encoder_input_ids, decoder_input_ids)

            batch_size = logits.shape[0]
            tgt_seq_len = logits.shape[1]
            vocab_size = logits.shape[2]

            logits = torch.reshape(logits, (batch_size * tgt_seq_len, vocab_size))
            labels = torch.reshape(labels, (batch_size * tgt_seq_len,))

            loss = self.loss_fn(logits, labels)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            self.train_loss_by_epoch.append(loss.item())

            if batch % 100 == 0:
                print(f"Loss: {loss.item():>7f} [{batch}/{len(self.train_dataloader)}]")
        self.train_loss = total_loss / len(self.train_dataloader)

    def _dev(self) -> None:
        self.model.eval()
        self.model.to(self.device)

        total_loss = 0.0

        with torch.no_grad():
            for encoder_input_ids, decoder_input_ids, labels in tqdm(
                self.dev_dataloader, "Evaluating"
            ):
                encoder_input_ids = encoder_input_ids.to(self.device)
                decoder_input_ids = decoder_input_ids.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(encoder_input_ids, decoder_input_ids)

                batch_size = logits.shape[0]
                tgt_seq_len = logits.shape[1]
                vocab_size = logits.shape[2]

                logits = torch.reshape(logits, (batch_size * tgt_seq_len, vocab_size))
                labels = torch.reshape(labels, (batch_size * tgt_seq_len,))

                loss = self.loss_fn(logits, labels)

                total_loss += loss.item()
                self.dev_loss_by_epoch.append(loss.item())

        self.dev_loss = total_loss / len(self.dev_dataloader)

    def test(self) -> Self:
        def time_test() -> None:
            for encoder_input_ids, _, labels in tqdm(self.test_dataloader, "Testing"):
                sequences = self.model.generate(
                    src_ids=encoder_input_ids,
                    bos_id=self.tokenizer.bos_id,
                    eos_id=self.tokenizer.eos_id,
                    max_len=self.max_src_len,
                    strategy=self.strategy,
                    beam_width=self.beam_width,
                )

                if isinstance(encoder_input_ids, torch.Tensor):
                    encoder_input_ids = encoder_input_ids.tolist()
                if isinstance(labels, torch.Tensor):
                    labels = labels.tolist()
                predictions = [
                    self.tokenizer.decode(sequence) for sequence in sequences
                ]
                cleaned_predictions: list[list[str]] = []
                for prediction in predictions:
                    cleaned_predictions.append(
                        [tok for tok in prediction if tok not in SPECIAL_TOKENS]
                    )

                ground_truth = [self.tokenizer.decode(label) for label in labels]
                cleaned_ground_truth: list[list[str]] = []
                for gold in ground_truth:
                    cleaned_ground_truth.append(
                        [tok for tok in gold if tok not in SPECIAL_TOKENS]
                    )

                laplace = SmoothingFunction()
                for pred, gold in zip(cleaned_predictions, cleaned_ground_truth):
                    bleu = sentence_bleu(
                        [gold],
                        pred,
                        weights=(1.0, 0.0, 0.0, 0.0),
                        smoothing_function=laplace.method2,
                    )
                    print(f"PRED: {pred}\nGOLD: {gold}\nBLEU: {bleu}")
                    self.total_bleu += bleu  # pyright: ignore
                    self.num_samples += 1

        timer = timeit.Timer(lambda: time_test())
        self.test_gen_time = timer.timeit(1)
        print(f"Time taken: {self.test_gen_time:.6f} seconds")

        return self

    def plot(self) -> Self:
        plt.plot(self.train_loss_by_epoch, label="Train Loss")
        plt.plot(self.dev_loss_by_epoch, label="Val Loss")
        plt.legend()
        plt.savefig(f"loss_curve_{self.uuid}.png")

        return self

    def report(self) -> Self:
        print(f"Average BLEU Score: {self.total_bleu / self.num_samples}")
        with open(
            f"E-{self.epochs}_LR-{self.learning_rate:.0e}_B-{self.batch_size}_S-{self.strategy}_{self.uuid}.txt",
            "w",
        ) as file:
            file.write("REPORT\n\n")
            file.write(f"total_bleu: {self.total_bleu}\n")
            file.write(f"num_samples: {self.num_samples}\n")
            file.write(f"average bleu: {self.total_bleu / self.num_samples}\n")
            file.write(f"train_file: {self.train_file}\n")
            file.write(f"dev_file: {self.dev_file}\n")
            file.write(f"test_file: {self.test_file}\n")
            file.write(f"epochs: {self.epochs}\n")
            file.write(f"learning_rate: {self.learning_rate}\n")
            file.write(f"batch_size: {self.batch_size}\n")
            file.write(f"max_src_len: {self.max_src_len}\n")
            file.write(f"max_tgt_len: {self.max_tgt_len}\n")
            file.write(f"d_model: {self.d_model}\n")
            file.write(f"num_heads: {self.num_heads}\n")
            file.write(f"d_ff: {self.d_ff}\n")
            file.write(f"num_enc_layers: {self.num_enc_layers}\n")
            file.write(f"num_dec_layers: {self.num_dec_layers}\n")
            file.write(f"dropout: {self.dropout}\n")
            file.write(f"strategy: {self.strategy}\n")
            file.write(f"beam_width: {self.beam_width}\n")
            file.write(f"train_loss_by_epoch: {self.train_loss_by_epoch}\n")
            file.write(f"dev_loss_by_epoch: {self.dev_loss_by_epoch}\n")
            file.write(f"test_gen_time: {self.test_gen_time}\n")
        return self

    def save_checkpoint(self) -> Self:
        torch.save(
            {
                "epoch": self.epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.train_loss,
            },
            f"checkpoint_{self.uuid}.pt",
        )
        return self

    def inference(self, checkpoint_path: str) -> Self:
        checkpoint = torch.load(checkpoint_path, weights_only=True)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.test()

        return self
