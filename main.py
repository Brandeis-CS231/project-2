from typing import Callable

import click
import torch
import torch.nn as nn
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
    """
    Train model for one epoch using teacher forcing.

    Args:
        model: The encoder-decoder model
        dataloader: DataLoader for training data
        optimizer: Optimizer for updating model parameters
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        device: Device to run training on ('cuda' or 'cpu')

    Returns:
        Average loss over the epoch
    """
    # Set model to training mode
    model.train()
    model.to(device)

    total_loss = 0.0

    # Iterate through batches with progress bar
    for encoder_input_ids, decoder_input_ids, label_ids in tqdm(dataloader, desc='Training'):
        # Move tensors to device
        encoder_input_ids = encoder_input_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        label_ids = label_ids.to(device)

        # Forward pass
        logits = model(encoder_input_ids, decoder_input_ids)  # (batch_size, tgt_seq_len, vocab_size)

        # Reshape for loss computation
        batch_size, tgt_seq_len, vocab_size = logits.shape
        logits_flat = logits.reshape(batch_size * tgt_seq_len, vocab_size)  # (batch_size * tgt_seq_len, vocab_size)
        labels_flat = label_ids.reshape(batch_size * tgt_seq_len)  # (batch_size * tgt_seq_len,)

        # Compute loss (automatically ignores padding if ignore_index is set)
        loss = loss_fn(logits_flat, labels_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    # Return average loss
    return total_loss / len(dataloader)


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """
    Evaluate model on validation or test set without updating parameters.

    Args:
        model: The encoder-decoder model
        dataloader: DataLoader for validation/test data
        loss_fn: Loss function (e.g., CrossEntropyLoss)
        device: Device to run evaluation on ('cuda' or 'cpu')

    Returns:
        Average loss over the dataset
    """
    # Set model to evaluation mode
    model.eval()
    model.to(device)

    total_loss = 0.0

    # Disable gradient computation
    with torch.no_grad():
        # Iterate through batches with progress bar
        for encoder_input_ids, decoder_input_ids, label_ids in tqdm(dataloader, desc='Evaluating'):
            # Move tensors to device
            encoder_input_ids = encoder_input_ids.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            label_ids = label_ids.to(device)

            # Forward pass
            logits = model(encoder_input_ids, decoder_input_ids)  # (batch_size, tgt_seq_len, vocab_size)

            # Reshape for loss computation
            batch_size, tgt_seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(batch_size * tgt_seq_len, vocab_size)
            labels_flat = label_ids.reshape(batch_size * tgt_seq_len)

            # Compute loss
            loss = loss_fn(logits_flat, labels_flat)

            # Accumulate loss
            total_loss += loss.item()

    # Return average loss
    return total_loss / len(dataloader)


def compute_bleu(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    strategy: str = "greedy",
    beam_width: int = 5,
    max_gen_len: int = 100
) -> float:
    """
    Compute BLEU score on test set using generated sequences.

    Args:
        model: The encoder-decoder model
        dataloader: DataLoader for test data
        tokenizer: Tokenizer instance for decoding
        device: Device to run generation on ('cuda' or 'cpu')
        strategy: Decoding strategy ('greedy' or 'beam_search')
        beam_width: Beam width for beam search
        max_gen_len: Maximum generation length

    Returns:
        Average BLEU score over the dataset
    """
    # Set model to evaluation mode
    model.eval()
    model.to(device)

    total_bleu = 0.0
    num_samples = 0

    # Smoothing function for BLEU
    smoothing = SmoothingFunction().method1

    with torch.no_grad():
        for encoder_input_ids, decoder_input_ids, label_ids in tqdm(dataloader, desc='Computing BLEU'):
            # Move tensors to device
            encoder_input_ids = encoder_input_ids.to(device)
            label_ids = label_ids.to(device)

            # Generate sequences
            generated_sequences = model.generate(
                src_ids=encoder_input_ids,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_len=max_gen_len,
                strategy=strategy,
                beam_width=beam_width
            )

            # Process each sample in the batch
            batch_size = encoder_input_ids.size(0)
            for i in range(batch_size):
                # Decode prediction
                pred_ids = generated_sequences[i]
                pred_tokens = tokenizer.decode(pred_ids)

                # Remove special tokens from prediction
                pred_tokens = [t for t in pred_tokens if t not in ['<bos>', '<eos>', '<pad>']]

                # Decode ground truth
                ref_ids = label_ids[i].tolist()
                ref_tokens = tokenizer.decode(ref_ids)

                # Remove padding from ground truth
                ref_tokens = [t for t in ref_tokens if t not in ['<bos>', '<eos>', '<pad>']]

                # Compute BLEU score for this sample
                # Using unigram weights (1.0, 0, 0, 0) as suggested in README
                if len(pred_tokens) > 0 and len(ref_tokens) > 0:
                    bleu = sentence_bleu(
                        [ref_tokens],  # Reference as list of sequences
                        pred_tokens,   # Hypothesis
                        weights=(1.0, 0, 0, 0),  # Unigram only
                        smoothing_function=smoothing
                    )
                    total_bleu += bleu

                num_samples += 1

    # Return average BLEU score
    return total_bleu / num_samples if num_samples > 0 else 0.0


@click.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('dev_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
@click.option('--epochs', default=10, help='Number of training epochs')
@click.option('--lr', default=1e-3, help='Learning rate')
@click.option('--batch-size', default=64, help='Batch size')
@click.option('--d-model', default=128, help='Model dimension')
@click.option('--num-heads', default=4, help='Number of attention heads')
@click.option('--d-ff', default=512, help='Feedforward dimension')
@click.option('--num-enc-layers', default=2, help='Number of encoder layers')
@click.option('--num-dec-layers', default=2, help='Number of decoder layers')
@click.option('--dropout', default=0.1, help='Dropout rate')
@click.option('--pos-encoding', default='sinusoidal', type=click.Choice(['sinusoidal', 'learnable']),
              help='Positional encoding type')
@click.option('--model-save-path', default='best_model.pt', help='Path to save best model')
@click.option('--seed', default=42, help='Random seed for reproducibility')
def main(
    train_file: str,
    dev_file: str,
    test_file: str,
    epochs: int,
    lr: float,
    batch_size: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_enc_layers: int,
    num_dec_layers: int,
    dropout: float,
    pos_encoding: str,
    model_save_path: str,
    seed: int
):
    """
    Main training script for sequence-to-sequence Transformer model.

    Args:
        train_file: Path to training data JSON file
        dev_file: Path to development/validation data JSON file
        test_file: Path to test data JSON file
    """
    # Import required modules
    from tokenizer import Tokenizer
    from dataset import SeqPairDataset
    from model import EncoderDecoder

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Random seed: {seed}")

    # ==================== Hyperparameters ====================
    # Sequence length hyperparameters
    max_src_len = 50
    max_tgt_len = 50
    max_len = 100  # Maximum sequence length for positional encoding

    print("\n" + "="*50)
    print("HYPERPARAMETERS")
    print("="*50)
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {lr}")
    print(f"Batch Size: {batch_size}")
    print(f"Max Src/Tgt Length: {max_src_len}/{max_tgt_len}")
    print(f"d_model: {d_model}, num_heads: {num_heads}, d_ff: {d_ff}")
    print(f"Encoder Layers: {num_enc_layers}, Decoder Layers: {num_dec_layers}")
    print(f"Dropout: {dropout}")
    print(f"Positional Encoding: {pos_encoding}")
    print(f"Model Save Path: {model_save_path}")
    print("="*50 + "\n")

    # ==================== Build Tokenizer ====================
    print("Building tokenizer from training data...")
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)
    vocab_size = len(tokenizer.word2idx)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens - BOS: {tokenizer.bos_id}, EOS: {tokenizer.eos_id}, "
          f"PAD: {tokenizer.pad_id}, UNK: {tokenizer.unk_id}\n")

    # ==================== Create Datasets ====================
    print("Creating datasets...")
    train_dataset = SeqPairDataset(train_file, tokenizer, max_src_len, max_tgt_len)
    dev_dataset = SeqPairDataset(dev_file, tokenizer, max_src_len, max_tgt_len)
    test_dataset = SeqPairDataset(test_file, tokenizer, max_src_len, max_tgt_len)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    print(f"Test samples: {len(test_dataset)}\n")

    # ==================== Create DataLoaders ====================
    print("Creating dataloaders...")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    print(f"Test batches: {len(test_loader)}\n")

    # ==================== Initialize Model ====================
    print("Initializing model...")
    model = EncoderDecoder(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        max_len=max_len,
        dropout=dropout,
        pad_idx=tokenizer.pad_id,
        pos_encoding_type=pos_encoding
    )
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")

    # ==================== Initialize Optimizer and Loss ====================
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # ==================== Training Loop ====================
    print("Starting training...\n")
    best_dev_loss = float('inf')

    for epoch in range(1, epochs + 1):
        print(f"{'='*50}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*50}")

        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Evaluate on dev set
        dev_loss = test_epoch(model, dev_loader, loss_fn, device)

        print(f"\nEpoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Dev Loss:   {dev_loss:.4f}")

        # Save best model
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved! (Dev Loss: {dev_loss:.4f})")

        print()

    # ==================== Final Evaluation ====================
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST SET")
    print("="*50 + "\n")

    # Load best model
    print("Loading best model...")
    model.load_state_dict(torch.load(model_save_path))

    # Compute test loss
    test_loss = test_epoch(model, test_loader, loss_fn, device)
    print(f"\nTest Loss: {test_loss:.4f}\n")

    # Compute BLEU score
    print("Computing BLEU score on test set...")
    bleu_score = compute_bleu(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        device=device,
        strategy="greedy",
        max_gen_len=max_tgt_len
    )
    print(f"\nTest BLEU Score: {bleu_score:.4f}\n")

    # ==================== Example Predictions ====================
    print("="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50 + "\n")

    model.eval()
    with torch.no_grad():
        # Get a few examples from test set
        examples = []
        for i in range(min(5, len(test_dataset))):
            enc_inp, dec_inp, labels = test_dataset[i]
            examples.append((enc_inp, dec_inp, labels))

        for idx, (enc_inp, dec_inp, labels) in enumerate(examples, 1):
            # Prepare input
            src_ids = enc_inp.unsqueeze(0).to(device)

            # Generate prediction
            generated = model.generate(
                src_ids=src_ids,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_len=max_tgt_len,
                strategy="greedy"
            )[0]

            # Decode sequences
            src_tokens = tokenizer.decode(enc_inp.tolist())
            src_text = ' '.join([t for t in src_tokens if t not in ['<bos>', '<eos>', '<pad>']])

            pred_tokens = tokenizer.decode(generated)
            pred_text = ' '.join([t for t in pred_tokens if t not in ['<bos>', '<eos>', '<pad>']])

            ref_tokens = tokenizer.decode(labels.tolist())
            ref_text = ' '.join([t for t in ref_tokens if t not in ['<bos>', '<eos>', '<pad>']])

            print(f"Example {idx}:")
            print(f"  Source:     {src_text}")
            print(f"  Target:     {ref_text}")
            print(f"  Prediction: {pred_text}")
            print()

    print("Training complete!")


if __name__ == "__main__":
    main()
