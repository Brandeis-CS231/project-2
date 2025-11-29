import json
from pathlib import Path
import time
from typing import Callable

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from matplotlib import pyplot as plt
import pandas as pd

from tokenizer import BOS, EOS, PAD, UNK, Tokenizer
from dataset import SeqPairDataset
from model import EncoderDecoder

SPECIAL_TOKENS = [BOS, EOS, PAD, UNK]

BEST_MODEL_PATH = "results/model5_0.001_32_50_50_256_8_256_4_4_0.1/model.pt"
BEST_MODEL_PARAMS = {
    "epochs": 5,
    "lr": 0.001,
    "batch_size": 32,
    "max_src_len": 50,
    "max_tgt_len": 50,
    "d_model": 256,
    "num_heads": 8,
    "d_ff": 256,
    "num_enc_layers": 4,
    "num_dec_layers": 4,
    "dropout": 0.1
}

# uv run main.py data/train.json data/dev.json data/test.json --bleu


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
    tokenizer: Tokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    strategy: str = "greedy",
    beam_width: int = 5,
    save_dir: str | None = None
):

    model.eval()
    model.to(device)

    total_bleu = 0.0
    num_samples = 0

    smoothing_fn = SmoothingFunction().method1

    # TODO: speed this up somehow with parallel processing?
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
                start = time.time()
                pred_tokens = tokenizer.decode(model_outputs[i])
                filtered_pred_tokens = [
                    token for token in pred_tokens if token not in SPECIAL_TOKENS]

                label_tokens = tokenizer.decode(label_ids[i].tolist())
                filtered_label_tokens = [
                    token for token in label_tokens if token not in SPECIAL_TOKENS]
                # TODO: this might be causing issues - need to test why BLEU is low and was getting errors before
                if len(filtered_label_tokens) == 0:
                    continue  # skip empty references
                bleu_score = sentence_bleu(
                    [filtered_label_tokens],
                    filtered_pred_tokens,
                    weights=(1.0, 0.0),
                    smoothing_function=smoothing_fn
                )
                end = time.time()
                if save_dir:
                    with open(Path(f"{save_dir}/bleu_details.txt"), 'a') as f:
                        print(f"Input: {' '.join(tokenizer.decode(enc_inp_ids[i].tolist()))}", file=f)
                        print(f"Reference: {' '.join(filtered_label_tokens)}", file=f)
                        print(f"Prediction: {' '.join(filtered_pred_tokens)}", file=f)
                        print(f"BLEU Score: {bleu_score:.4f}", file=f)
                        print(f"Decoding Strategy: {strategy} (Beam Width: {beam_width})", file=f)
                        print(f"Sequence Length: {len(filtered_label_tokens)}", file=f)
                        print(f"Time Taken: {end - start:.4f} seconds", file=f)
                        print("-" * 50, file=f)

                total_bleu += bleu_score
                num_samples += 1

    if num_samples == 0:
        return 0.0

    return total_bleu / num_samples


@click.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('dev_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--search_grid', is_flag=True, help='Perform grid search over hyperparameters')
@click.option('--exp_one', is_flag=True, help='Run experiment one: Sinusoidal vs Learnable Positional Encodings')
@click.option('--exp_two', is_flag=True, help='Run experiment two: Different Decoding Strategies (Greedy vs Beam Search)')
@click.option('--exp_three', is_flag=True, help='Run experiment three: Model Architecture Variants')
@click.option('--all_experiments', is_flag=True, help='Run all experiments sequentially')
@click.option('--saved_model', type=click.Path(exists=True), default=None)
@click.option('--bleu', is_flag=True, help='Compute BLEU score on test set after training/evaluation')
def main(
    train_file: str,
    dev_file: str,
    test_file: str,
    saved_model: str | None = None,
    bleu: bool = False,
    search_grid: bool = False,
    exp_one: bool = False,
    exp_two: bool = False,
    exp_three: bool = False,
    all_experiments: bool = False
):
    start_time = time.time()
    torch.manual_seed(42)

    if search_grid:
        grid_search(train_file, dev_file, test_file)
        end_time = time.time()
        print(f"Grid search completed in {end_time - start_time:.2f} seconds")
        return
    if exp_one:
        experiment_one(train_file, dev_file, test_file)
        end_time = time.time()
        print(f"Experiment one completed in {end_time - start_time:.2f} seconds")
        return
    if exp_two:
        experiment_two(train_file, dev_file, test_file)
        end_time = time.time()
        print(f"Experiment two completed in {end_time - start_time:.2f} seconds")
        return
    if exp_three:
        experiment_three(train_file, dev_file, test_file)
        end_time = time.time()
        print(f"Experiment three completed in {end_time - start_time:.2f} seconds")
        return
    if all_experiments:
        experiment_one(train_file, dev_file, test_file)
        experiment_two(train_file, dev_file, test_file)
        experiment_three(train_file, dev_file, test_file)
        end_time = time.time()
        print(f"All experiments completed in {end_time - start_time:.2f} seconds")
        return

    hyperparams = {
        "epochs": 2,
        "lr": 0.001,
        "batch_size": 32,
        "max_src_len": 50,
        "max_tgt_len": 50,
        "d_model": 128,
        "num_heads": 2,
        "d_ff": 256,
        "num_enc_layers": 1,
        "num_dec_layers": 1,
        "dropout": 0.1
    }

    hyperparams = BEST_MODEL_PARAMS.copy()
    # to redo experiment 3 heads 8
    hyperparams["num_heads"] = 8

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # initialize tokenizer and build vocab
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)

    # create datasets and dataloaders
    train_dataset, dev_dataset, test_dataset = create_datasets(
        train_file, dev_file, test_file, tokenizer, hyperparams[
            "max_src_len"], hyperparams["max_tgt_len"]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"])

    # initialize model, loss function, optimizer
    model = EncoderDecoder(
        src_vocab_size=len(tokenizer.src_vocab),
        tgt_vocab_size=len(tokenizer.tgt_vocab),
        d_model=hyperparams["d_model"],
        num_heads=hyperparams["num_heads"],
        d_ff=hyperparams["d_ff"],
        num_enc_layers=hyperparams["num_enc_layers"],
        num_dec_layers=hyperparams["num_dec_layers"],
        max_len=hyperparams["max_src_len"],
        dropout=hyperparams["dropout"],
        pad_idx=tokenizer.pad_id
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # training loop
    if saved_model is not None:
        model.load_state_dict(torch.load(saved_model))
    else:
        # train the model
        model, train_losses, dev_losses = train_model(
            model, train_loader, dev_loader, optimizer, loss_fn, hyperparams["epochs"], device
        )
        # final evaluation on test set
        test_loss = test_epoch(
            model, test_loader, loss_fn, device)
        print(f"Test Loss: {test_loss:.4f}")

        # save model and results
        save_dir = save_model_results(
            model, train_losses, dev_losses, hyperparams, folder_name=f"exp_three_results/heads_{hyperparams['num_heads']}")

        evaluate_model(
            model, test_loader, tokenizer, loss_fn, save_dir, device
        )

    # compute BLEU score on test set
    if bleu:
        bleu_score = compute_bleu(
            model, test_loader, tokenizer, device, strategy="greedy")
        print(f"Test BLEU Score: {bleu_score:.4f}")

    end_time = time.time()
    print(f"Execution Time: {end_time - start_time:.2f} seconds")


def create_datasets(
    train_file: str,
    dev_file: str,
    test_file: str,
    tokenizer: Tokenizer,
    max_src_len: int,
    max_tgt_len: int,
) -> tuple[SeqPairDataset, SeqPairDataset, SeqPairDataset]:
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
    return train_dataset, dev_dataset, test_dataset


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    num_epochs: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> tuple[nn.Module, list[float], list[float]]:
    train_losses = []
    dev_losses = []
    # TODO: implement early stopping? - probably no
    # TODO: save checkpoints after each epoch? - probably also no, just at end
    # TODO: print example outputs to assess quality? - part of evaluation function
    # TODO: monitor training and validation loss and create graphs? - part of save function
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device)
        print(f"Train Loss: {train_loss:.4f}")

        dev_loss = test_epoch(
            model, dev_loader, loss_fn, device)
        print(f"Dev Loss: {dev_loss:.4f}")

        train_losses.append(train_loss)
        dev_losses.append(dev_loss)

    return model, train_losses, dev_losses


def save_model_results(
    model: nn.Module,
    train_losses: list[float],
    dev_losses: list[float],
    hyperparams: dict,
    folder_name: str = "results"
):
    # folder structure:
    # results / hyperparams as dir name / saved model, all results, plots

    # model name based on hyperparameters for simplicity and easy identification
    # best-performing model is later renamed to "best_model" for easy loading
    model_name = "model" + \
        "_".join(f"{value}" for value in hyperparams.values())

    save_dir = Path(f"{folder_name}/{model_name}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), Path(f"{save_dir}/model.pt"))

    results = pd.DataFrame({
        "Epoch": list(range(1, len(train_losses) + 1)),
        "Train Loss": train_losses,
        "Dev Loss": dev_losses
    })
    results.to_csv(Path(f"{save_dir}/losses.csv"), index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(dev_losses, label="Dev Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Development Loss Over Epochs")
    plt.legend()
    plt.savefig(Path(f"{save_dir}/loss_plot.png"))
    plt.close()

    return save_dir


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    tokenizer: Tokenizer,
    loss_fn,
    save_dir: str,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    strategy: str = "greedy",
    beam_width: int = 5
):
    # final evaluation on test set
    test_loss = test_epoch(
        model, data_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}")

    # generate example outputs for qualitative assessment
    for batch in tqdm(data_loader, desc="Generating Examples", leave=False):
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
        with open(Path(f"{save_dir}/example_outputs.txt"), 'w') as f:
            for i in range(label_ids.size(0)):
                pred_tokens = tokenizer.decode(model_outputs[i])
                filtered_pred_tokens = [
                    token for token in pred_tokens if token not in SPECIAL_TOKENS]

                label_tokens = tokenizer.decode(label_ids[i].tolist())
                filtered_label_tokens = [
                    token for token in label_tokens if token not in SPECIAL_TOKENS]
                input_tokens = tokenizer.decode(enc_inp_ids[i].tolist())
                filtered_input_tokens = [
                    token for token in input_tokens if token not in SPECIAL_TOKENS]

                print(f"Input: {' '.join(filtered_input_tokens)}", file=f)
                print(f"Reference: {' '.join(filtered_label_tokens)}", file=f)
                print(f"Prediction: {' '.join(filtered_pred_tokens)}", file=f)
                print("-" * 50, file=f)

        break  # only do one batch for examples
    return test_loss


def grid_search(
    train_file: str,
    dev_file: str,
    test_file: str
):
    # should make sure to keep track of final results for each setting and save that result
    depth_options = [1, 2, 4]
    dim_options = [128, 256]
    head_options = [2, 4, 8]
    ff_options = [256, 512]
    batch_size_options = [32, 64]
    num_combinations = (len(depth_options) * len(dim_options) *
                        len(head_options) * len(ff_options) * len(batch_size_options))

    # initialize hyperparams dict with defaults
    hyperparams = {
        "epochs": 5,  # not changing
        "lr": 0.001,  # not changing
        "batch_size": 32,
        "max_src_len": 50,  # not changing
        "max_tgt_len": 50,  # not changing
        "d_model": 128,
        "num_heads": 2,
        "d_ff": 256,
        "num_enc_layers": 1,
        "num_dec_layers": 1,
        "dropout": 0.1  # not changing
    }
    # store hyperparams, final dev loss, final test loss, test loss, BLEU score
    overall_results = []

    # set device, tokenizer, datasets, dataloaders outside loop
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)

    train_dataset, dev_dataset, test_dataset = create_datasets(
        train_file, dev_file, test_file, tokenizer, hyperparams["max_src_len"], hyperparams["max_tgt_len"])

    i = 1
    for batch_size in batch_size_options:
        hyperparams["batch_size"] = batch_size

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        for dim in dim_options:
            hyperparams["d_model"] = dim
            for heads in head_options:
                hyperparams["num_heads"] = heads
                for ff in ff_options:
                    hyperparams["d_ff"] = ff
                    for depth in depth_options:
                        hyperparams["num_enc_layers"] = depth
                        hyperparams["num_dec_layers"] = depth

                        print(f"Starting combination {i}/{num_combinations}")

                        # print("Hyperparameters:")
                        # for key, value in hyperparams.items():
                        #     print(f"{key}: {value}")

                        # continue
                        try:

                            # train and evaluate model with these hyperparams
                            model = EncoderDecoder(
                                src_vocab_size=len(tokenizer.src_vocab),
                                tgt_vocab_size=len(tokenizer.tgt_vocab),
                                d_model=hyperparams["d_model"],
                                num_heads=hyperparams["num_heads"],
                                d_ff=hyperparams["d_ff"],
                                num_enc_layers=hyperparams["num_enc_layers"],
                                num_dec_layers=hyperparams["num_dec_layers"],
                                max_len=hyperparams["max_src_len"],
                                dropout=hyperparams["dropout"],
                                pad_idx=tokenizer.pad_id
                            )
                            optimizer = torch.optim.Adam(
                                model.parameters(), lr=hyperparams["lr"])
                            loss_fn = nn.CrossEntropyLoss(
                                ignore_index=tokenizer.pad_id)

                            model, train_losses, dev_losses = train_model(
                                model, train_loader, dev_loader, optimizer, loss_fn,
                                hyperparams["epochs"], device
                            )

                            save_dir = save_model_results(
                                model, train_losses, dev_losses, hyperparams, folder_name="results")

                            test_loss = evaluate_model(
                                model, test_loader, tokenizer, loss_fn, save_dir, device
                            )

                            try:
                                bleu_score = compute_bleu(
                                    model, test_loader, tokenizer, device, strategy="greedy"
                                )
                            except Exception as e:
                                with open("results/grid_search_errors.txt", 'a') as f:
                                    f.write(
                                        f"Error computing BLEU for combination {i}: {e}\n")
                                bleu_score = None

                            print(
                                f"Completed {i}/{num_combinations} combinations")
                            i += 1
                            print("-" * 100)

                            overall_results.append({
                                "hyperparams": hyperparams.copy(),
                                "final_dev_loss": dev_losses[-1],
                                "final_train_loss": train_losses[-1],
                                "test_loss": test_loss,
                                "BLEU_score": bleu_score
                            })
                        except Exception as e:
                            with open("results/grid_search_errors.txt", 'a') as f:
                                f.write(f"Error with combination {i}: {e}\n")
                            i += 1

                    # continue  # temporary to limit runtime during testing
                    # overall_results_df = pd.DataFrame(overall_results)
                    # expanded_hyperparams = pd.DataFrame(
                    #     overall_results_df['hyperparams'].tolist())
                    # print(expanded_hyperparams.to_string())
                    # overall_results_df = pd.concat([overall_results_df.drop(
                    #     'hyperparams', axis=1), expanded_hyperparams], axis=1)
                    # overall_results_df.to_csv(
                    #     "results/grid_search_overall_results.csv", float_format="%.3f", index=False)
                    # return  # temporary to limit runtime during testing
    overall_results_df = pd.DataFrame(overall_results)
    expanded_hyperparams = pd.DataFrame(
        overall_results_df['hyperparams'].tolist())
    overall_results_df = pd.concat([overall_results_df.drop(
        'hyperparams', axis=1), expanded_hyperparams], axis=1)
    overall_results_df.to_csv(
        "results/grid_search_overall_results.csv", float_format="%.3f", index=False)


# DONE: reporting methods - training/validation loss curves, test set, BLEU score, example outputs
# save models at various checkpoints too

# TODO: experiment 1 - sinusoidal vs learnable positional encodings by modifying model.py
def experiment_one(
    train_file: str,
    dev_file: str,
    test_file: str,
):
    results = {'sinusoidal': {}, 'learnable': {}}
    # train baseline
    hyperparams = BEST_MODEL_PARAMS.copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize tokenizer and build vocab
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)

    # create datasets and dataloaders
    train_dataset, dev_dataset, test_dataset = create_datasets(
        train_file, dev_file, test_file, tokenizer, hyperparams[
            "max_src_len"], hyperparams["max_tgt_len"]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"])

    # initialize model, loss function, optimizer
    model = EncoderDecoder(
        src_vocab_size=len(tokenizer.src_vocab),
        tgt_vocab_size=len(tokenizer.tgt_vocab),
        d_model=hyperparams["d_model"],
        num_heads=hyperparams["num_heads"],
        d_ff=hyperparams["d_ff"],
        num_enc_layers=hyperparams["num_enc_layers"],
        num_dec_layers=hyperparams["num_dec_layers"],
        max_len=hyperparams["max_src_len"],
        dropout=hyperparams["dropout"],
        pad_idx=tokenizer.pad_id,
        learnable_pos_emb=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # training loop
    # train the model
    model, train_losses, dev_losses = train_model(
        model, train_loader, dev_loader, optimizer, loss_fn, hyperparams["epochs"], device
    )
    # final evaluation on test set
    test_loss = test_epoch(
        model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}")
    results['sinusoidal']['test_loss'] = test_loss

    # save model and results
    save_dir = save_model_results(
        model, train_losses, dev_losses, hyperparams, folder_name="exp_one_results/sinusoidal")

    evaluate_model(
        model, test_loader, tokenizer, loss_fn, save_dir, device
    )

    bleu_baseline = compute_bleu(
        model, test_loader, tokenizer, device, strategy="greedy")

    print(f"Number of parameters (Sinusoidal Pos Enc): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    results['sinusoidal']['BLEU'] = bleu_baseline
    results['sinusoidal']['param_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # train variant with learnable positional encodings
    # initialize model, loss function, optimizer
    model = EncoderDecoder(
        src_vocab_size=len(tokenizer.src_vocab),
        tgt_vocab_size=len(tokenizer.tgt_vocab),
        d_model=hyperparams["d_model"],
        num_heads=hyperparams["num_heads"],
        d_ff=hyperparams["d_ff"],
        num_enc_layers=hyperparams["num_enc_layers"],
        num_dec_layers=hyperparams["num_dec_layers"],
        max_len=hyperparams["max_src_len"],
        dropout=hyperparams["dropout"],
        pad_idx=tokenizer.pad_id,
        learnable_pos_emb=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # training loop
    # train the model
    model, train_losses, dev_losses = train_model(
        model, train_loader, dev_loader, optimizer, loss_fn, hyperparams["epochs"], device
    )
    # final evaluation on test set
    test_loss = test_epoch(
        model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}")
    results['learnable']['test_loss'] = test_loss

    # save model and results
    save_dir = save_model_results(
        model, train_losses, dev_losses, hyperparams, folder_name="exp_one_results/learnable")

    evaluate_model(
        model, test_loader, tokenizer, loss_fn, save_dir, device
    )

    bleu_learnable_pos = compute_bleu(
        model, test_loader, tokenizer, device, strategy="greedy")

    print(f"BLEU Baseline (Sinusoidal Pos Enc): {bleu_baseline:.4f}")
    print(f"BLEU Learnable Pos Enc: {bleu_learnable_pos:.4f}")
    results['learnable']['BLEU'] = bleu_learnable_pos
    results['learnable']['param_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Number of parameters (Learnable Pos Enc): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # save results to file
    json_save_path = Path("exp_one_results/pos_enc_comparison_results.json")
    with open(json_save_path, 'w') as f:
        json.dump(results, f, indent=4)

# TODO: experiment 2 - different decoding strategies (greedy vs beam search) in main.py
# only actually need to run the greedy one b/c beam takes too long
# report BLEU score, generation time, avg seq length


def experiment_two(
    train_file: str,
    dev_file: str,
    test_file: str,

):
    results = {}
    hyperparams = BEST_MODEL_PARAMS.copy()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize tokenizer and build vocab
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)

    # create datasets and dataloaders
    train_dataset, dev_dataset, test_dataset = create_datasets(
        train_file, dev_file, test_file, tokenizer, hyperparams[
            "max_src_len"], hyperparams["max_tgt_len"]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"])

    # initialize model, loss function, optimizer
    model = EncoderDecoder(
        src_vocab_size=len(tokenizer.src_vocab),
        tgt_vocab_size=len(tokenizer.tgt_vocab),
        d_model=hyperparams["d_model"],
        num_heads=hyperparams["num_heads"],
        d_ff=hyperparams["d_ff"],
        num_enc_layers=hyperparams["num_enc_layers"],
        num_dec_layers=hyperparams["num_dec_layers"],
        max_len=hyperparams["max_src_len"],
        dropout=hyperparams["dropout"],
        pad_idx=tokenizer.pad_id,
        learnable_pos_emb=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # training loop
    # train the model
    model, train_losses, dev_losses = train_model(
        model, train_loader, dev_loader, optimizer, loss_fn, hyperparams["epochs"], device
    )
    # # final evaluation on test set
    # test_loss = test_epoch(
    #     model, test_loader, loss_fn, device)
    # print(f"Test Loss: {test_loss:.4f}")

    # save model and results
    save_dir = save_model_results(
        model, train_losses, dev_losses, hyperparams, folder_name="exp_two_results")

    test_loss = evaluate_model(
        model, test_loader, tokenizer, loss_fn, save_dir, device
    )
    print(f"Test Loss: {test_loss:.4f}")
    results['test_loss'] = test_loss

    greedy_start = time.time()
    greedy_bleu = compute_bleu(
        model, test_loader, tokenizer, device, strategy="greedy", save_dir=save_dir)
    greedy_end = time.time()
    # TODO: need to get average sequence length
    # TODO: need to save some example outputs for both methods to compare quality
    # needs to output source, ground truth, greedy, and beam outputs

    print(f"BLEU Greedy: {greedy_bleu:.4f}")
    print(f"Total Greedy Decoding Time: {greedy_end - greedy_start:.4f} seconds")
    results['greedy'] = {
        'BLEU': greedy_bleu,
        'decoding_time': greedy_end - greedy_start
    }

    beam3_start = time.time()
    beam3_bleu = compute_bleu(
        model, test_loader, tokenizer, device, strategy="beam_search", beam_width=3, save_dir=save_dir)
    beam3_end = time.time()

    print(f"BLEU Beam Search (width=3): {beam3_bleu:.4f}")
    print(f"Beam Search Decoding Time (width=3): {beam3_end - beam3_start:.4f} seconds")
    results['beam_search_width_3'] = {
        'BLEU': beam3_bleu,
        'decoding_time': beam3_end - beam3_start
    }

    with open("exp_two_results/decoding_strategy_comparison_results.json", 'w') as f:
        json.dump(results, f, indent=4)

# TODO: experiment 3 - model architecture variants
# number attention heads
# encoder/decoder depth


def experiment_three(
    train_file: str,
    dev_file: str,
    test_file: str,
):
    # TODO: I've already mostly done this experiment with the grid search
    # really just need to analyze those results and write them up
    # could also do the residual connection & attention variants experiments, but not at all necessary
    
    # TODO: need to do this one to have correct BLEU scores for these

    hyperparams = BEST_MODEL_PARAMS.copy()

    num_heads = [2, 4, 8]
    depths = [1, 2, 4]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # initialize tokenizer and build vocab
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)

    # create datasets and dataloaders
    train_dataset, dev_dataset, test_dataset = create_datasets(
        train_file, dev_file, test_file, tokenizer, hyperparams[
            "max_src_len"], hyperparams["max_tgt_len"]
    )
    train_loader = DataLoader(
        train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=hyperparams["batch_size"])
    test_loader = DataLoader(
        test_dataset, batch_size=hyperparams["batch_size"])
    
    results = []

    for heads in num_heads:
        ind_results = {}
        hyperparams["num_heads"] = heads
        model = EncoderDecoder(
            src_vocab_size=len(tokenizer.src_vocab),
            tgt_vocab_size=len(tokenizer.tgt_vocab),
            d_model=hyperparams["d_model"],
            num_heads=hyperparams["num_heads"],
            d_ff=hyperparams["d_ff"],
            num_enc_layers=hyperparams["num_enc_layers"],
            num_dec_layers=hyperparams["num_dec_layers"],
            max_len=hyperparams["max_src_len"],
            dropout=hyperparams["dropout"],
            pad_idx=tokenizer.pad_id
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

        model, train_losses, dev_losses = train_model(
                model, train_loader, dev_loader, optimizer, loss_fn, hyperparams["epochs"], device
            )
        
        # save model and results
        save_dir = save_model_results(
            model, train_losses, dev_losses, hyperparams, folder_name=f"exp_three_results/heads_{heads}")

        test_loss = evaluate_model(
            model, test_loader, tokenizer, loss_fn, save_dir, device
        )
        print(f"Test Loss (Heads={heads}): {test_loss:.4f}")

        # compute BLEU score on test set
        bleu_score = compute_bleu(
            model, test_loader, tokenizer, device, strategy="greedy")
        print(f"Test BLEU Score: {bleu_score:.4f}")

        ind_results['num_heads'] = heads
        ind_results['test_loss'] = test_loss
        ind_results['bleu_score'] = bleu_score
        results.append(ind_results)


    with open("exp_three_results/num_heads_experiment_results.json", 'w') as f:
        results.append(hyperparams)
        json.dump(results, f, indent=4)

    results = []
    for depth in depths:
        ind_results = {}
        hyperparams["num_enc_layers"] = depth
        hyperparams["num_dec_layers"] = depth
        model = EncoderDecoder(
            src_vocab_size=len(tokenizer.src_vocab),
            tgt_vocab_size=len(tokenizer.tgt_vocab),
            d_model=hyperparams["d_model"],
            num_heads=hyperparams["num_heads"],
            d_ff=hyperparams["d_ff"],
            num_enc_layers=hyperparams["num_enc_layers"],
            num_dec_layers=hyperparams["num_dec_layers"],
            max_len=hyperparams["max_src_len"],
            dropout=hyperparams["dropout"],
            pad_idx=tokenizer.pad_id
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
        loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

        model, train_losses, dev_losses = train_model(
                model, train_loader, dev_loader, optimizer, loss_fn, hyperparams["epochs"], device
            )
        
        # save model and results
        save_dir = save_model_results(
            model, train_losses, dev_losses, hyperparams, folder_name=f"exp_three_results/depth_{depth}")

        test_loss = evaluate_model(
            model, test_loader, tokenizer, loss_fn, save_dir, device
        )
        print(f"Test Loss (Depth={depth}): {test_loss:.4f}")

        # compute BLEU score on test set
        bleu_score = compute_bleu(
            model, test_loader, tokenizer, device, strategy="greedy")
        print(f"Test BLEU Score: {bleu_score:.4f}")

        ind_results['depth'] = depth
        ind_results['test_loss'] = test_loss
        ind_results['bleu_score'] = bleu_score
        results.append(ind_results)

    with open("exp_three_results/depth_experiment_results.json", 'w') as f:
        results.append(hyperparams)
        json.dump(results, f, indent=4)    


if __name__ == "__main__":
    main()
