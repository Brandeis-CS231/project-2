from typing import Literal

import click
import torch

from project_2.trainer import Trainer

device: str = "cuda" if torch.cuda.is_available() else "cpu"


@click.group()
def cli():
    pass


@cli.command()
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("dev_file", type=click.Path(exists=True))
@click.argument("test_file", type=click.Path(exists=True))
@click.option("--epochs", type=click.INT)
@click.option("--learning_rate", type=click.FLOAT)
@click.option("--batch_size", type=click.INT)
@click.option("--max_src_len", type=click.INT)
@click.option("--max_tgt_len", type=click.INT)
@click.option("--d_model", type=click.INT)
@click.option("--num_heads", type=click.INT)
@click.option("--d_ff", type=click.INT)
@click.option("--num_enc_layers", type=click.INT)
@click.option("--num_dec_layers", type=click.INT)
@click.option("--dropout", type=click.FLOAT)
@click.option(
    "--strategy", default="greedy", type=click.Choice(["greedy", "beam_search"])
)
@click.option("--beam_width", default=5)
@click.option("--save", is_flag=True, help="Enable verbose output.")
def train(
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
    save: bool,
) -> None:
    trainer = (
        Trainer(
            device=device,
            train_file=train_file,
            dev_file=dev_file,
            test_file=test_file,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_enc_layers=num_enc_layers,
            num_dec_layers=num_dec_layers,
            dropout=dropout,
            strategy=strategy,
            beam_width=beam_width,
        )
        .fit()
        .test()
        .report()
        .plot()
    )

    if save:
        trainer.save_checkpoint()


@cli.command()
@click.option("--checkpoint_path")
@click.option(
    "--strategy", default="greedy", type=click.Choice(["greedy", "beam_search"])
)
@click.option("--beam_width", default=5)
def inference(
    checkpoint_path: str, strategy: Literal["greedy", "beam_search"], beam_width: int
) -> None:
    Trainer(
        device=device,
        train_file="./data/train.json",
        dev_file="./data/dev.json",
        test_file="./data/test.json",
        epochs=4,
        learning_rate=1e-4,
        batch_size=32,
        max_src_len=50,
        max_tgt_len=50,
        d_model=256,
        num_heads=2,
        d_ff=512,
        num_enc_layers=4,
        num_dec_layers=4,
        dropout=0.1,
        strategy=strategy,
        beam_width=beam_width,
    ).inference(checkpoint_path).report()


if __name__ == "__main__":
    cli()
