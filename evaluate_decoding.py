"""
Evaluation script for comparing different decoding strategies (Experiment 2)
"""
import time
import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tokenizer import Tokenizer
from dataset import SeqPairDataset
from model import EncoderDecoder


def evaluate_with_strategy(
    model,
    dataloader,
    tokenizer,
    strategy="greedy",
    beam_width=5,
    max_gen_len=50,
    device="cpu"
):
    """
    Evaluate model with a specific decoding strategy.

    Returns:
        bleu_score: Average BLEU score
        avg_time: Average generation time per sample (seconds)
        avg_length: Average generated sequence length
    """
    model.eval()
    model.to(device)

    total_bleu = 0.0
    total_time = 0.0
    total_length = 0
    num_samples = 0

    smoothing = SmoothingFunction().method1

    with torch.no_grad():
        for encoder_input_ids, _, label_ids in tqdm(dataloader, desc=f'{strategy} decoding'):
            encoder_input_ids = encoder_input_ids.to(device)
            label_ids = label_ids.to(device)

            batch_size = encoder_input_ids.size(0)

            # Time generation for this batch
            start_time = time.time()

            generated_sequences = model.generate(
                src_ids=encoder_input_ids,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_len=max_gen_len,
                strategy=strategy,
                beam_width=beam_width
            )

            batch_time = time.time() - start_time
            total_time += batch_time

            # Process each sample
            for i in range(batch_size):
                pred_ids = generated_sequences[i]
                pred_tokens = tokenizer.decode(pred_ids)
                pred_tokens = [t for t in pred_tokens if t not in ['<bos>', '<eos>', '<pad>']]

                ref_ids = label_ids[i].tolist()
                ref_tokens = tokenizer.decode(ref_ids)
                ref_tokens = [t for t in ref_tokens if t not in ['<bos>', '<eos>', '<pad>']]

                # Compute BLEU
                if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                    bleu = sentence_bleu(
                        [ref_tokens],
                        pred_tokens,
                        weights=(1.0, 0, 0, 0),
                        smoothing_function=smoothing
                    )
                    total_bleu += bleu

                # Track length
                total_length += len(pred_tokens)
                num_samples += 1

    avg_bleu = total_bleu / num_samples if num_samples > 0 else 0.0
    avg_time = total_time / num_samples if num_samples > 0 else 0.0
    avg_length = total_length / num_samples if num_samples > 0 else 0.0

    return avg_bleu, avg_time, avg_length


def show_examples(model, dataset, tokenizer, strategies, device="cpu", num_examples=5):
    """Show example outputs for different decoding strategies."""
    model.eval()

    print("\n" + "="*80)
    print("EXAMPLE OUTPUTS")
    print("="*80 + "\n")

    for idx in range(min(num_examples, len(dataset))):
        enc_inp, _, labels = dataset[idx]
        src_ids = enc_inp.unsqueeze(0).to(device)

        # Decode source
        src_tokens = tokenizer.decode(enc_inp.tolist())
        src_text = ' '.join([t for t in src_tokens if t not in ['<bos>', '<eos>', '<pad>']])

        # Decode reference
        ref_tokens = tokenizer.decode(labels.tolist())
        ref_text = ' '.join([t for t in ref_tokens if t not in ['<bos>', '<eos>', '<pad>']])

        print(f"Example {idx + 1}:")
        print(f"  Source: {src_text}")
        print(f"  Target: {ref_text}")

        # Generate with each strategy
        for strategy_name, strategy, beam_width in strategies:
            with torch.no_grad():
                generated = model.generate(
                    src_ids=src_ids,
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id,
                    max_len=50,
                    strategy=strategy,
                    beam_width=beam_width
                )[0]

            pred_tokens = tokenizer.decode(generated)
            pred_text = ' '.join([t for t in pred_tokens if t not in ['<bos>', '<eos>', '<pad>']])
            print(f"  {strategy_name}: {pred_text}")

        print()


@click.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.argument('train_file', type=click.Path(exists=True))
@click.option('--batch-size', default=32, help='Batch size for evaluation')
@click.option('--d-model', default=128, help='Model dimension')
@click.option('--num-heads', default=4, help='Number of attention heads')
@click.option('--d-ff', default=512, help='Feedforward dimension')
@click.option('--num-enc-layers', default=2, help='Number of encoder layers')
@click.option('--num-dec-layers', default=2, help='Number of decoder layers')
@click.option('--dropout', default=0.1, help='Dropout rate')
@click.option('--pos-encoding', default='sinusoidal', help='Positional encoding type')
def main(
    model_path,
    test_file,
    train_file,
    batch_size,
    d_model,
    num_heads,
    d_ff,
    num_enc_layers,
    num_dec_layers,
    dropout,
    pos_encoding
):
    """Evaluate different decoding strategies on a trained model."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # Build tokenizer from training data
    print("Building tokenizer...")
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)
    vocab_size = len(tokenizer.word2idx)
    print(f"Vocabulary size: {vocab_size}\n")

    # Load test dataset
    print("Loading test dataset...")
    test_dataset = SeqPairDataset(test_file, tokenizer, 50, 50)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Test samples: {len(test_dataset)}\n")

    # Initialize model
    print("Initializing model...")
    model = EncoderDecoder(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        max_len=100,
        dropout=dropout,
        pad_idx=tokenizer.pad_id,
        pos_encoding_type=pos_encoding
    )

    # Load model weights
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Model loaded!\n")

    # Define decoding strategies to test
    strategies = [
        # ("Greedy", "greedy", 1),
        # ("Beam Search (width=3)", "beam_search", 3),
        ("Beam Search (width=5)", "beam_search", 5),
        ("Beam Search (width=10)", "beam_search", 10),
    ]

    # Evaluate each strategy
    print("="*80)
    print("EVALUATING DECODING STRATEGIES")
    print("="*80 + "\n")

    results = []
    for strategy_name, strategy, beam_width in strategies:
        print(f"Evaluating: {strategy_name}")
        bleu, avg_time, avg_length = evaluate_with_strategy(
            model=model,
            dataloader=test_loader,
            tokenizer=tokenizer,
            strategy=strategy,
            beam_width=beam_width,
            max_gen_len=50,
            device=device
        )
        results.append((strategy_name, bleu, avg_time, avg_length))
        print(f"  BLEU: {bleu:.4f}")
        print(f"  Avg time per sample: {avg_time:.4f}s")
        print(f"  Avg sequence length: {avg_length:.2f}")
        print()

    # Summary table
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Strategy':<30} {'BLEU':>10} {'Time (s)':>12} {'Avg Length':>12}")
    print("-"*80)
    for strategy_name, bleu, avg_time, avg_length in results:
        print(f"{strategy_name:<30} {bleu:>10.4f} {avg_time:>12.4f} {avg_length:>12.2f}")
    print()

    # Show example outputs
    show_examples(model, test_dataset, tokenizer, strategies, device, num_examples=5)

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
