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


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The model to train
        dataloader: DataLoader for training data
        optimizer: Optimizer for updating parameters
        loss_fn: Loss function
        device: Device to use (cuda or cpu)

    Returns:
        Average loss over the epoch
    """
    # Set model to training mode
    model.train()
    model.to(device)

    total_loss = 0.0

    # Iterate through batches with progress bar
    for batch in tqdm(dataloader, desc='Training'):
        # print("this is running")
        # Unpack batch
        encoder_input_ids, decoder_input_ids, label_ids = batch

        # Move tensors to device
        encoder_input_ids = encoder_input_ids.to(device)
        decoder_input_ids = decoder_input_ids.to(device)
        label_ids = label_ids.to(device)

        # Forward pass
        logits = model(encoder_input_ids, decoder_input_ids)
        # logits shape: (batch_size, tgt_seq_len, vocab_size)

        # Reshape for loss computation
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(batch_size * seq_len, vocab_size)
        labels_flat = label_ids.view(batch_size * seq_len)

        # Compute loss
        loss = loss_fn(logits_flat, labels_flat)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

    # Return average loss
    average = total_loss / len(dataloader)
    return average


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    """
        Evaluate the model on validation or test set.

        Args:
            model: The model to evaluate
            dataloader: DataLoader for validation/test data
            loss_fn: Loss function
            device: Device to use (cuda or cpu)

        Returns:
            Average loss over the dataset
        """
    # Set model to evaluation mode
    model.eval()
    model.to(device)

    total_loss = 0.0

    # Disable gradient computation
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Unpack batch
            encoder_input_ids, decoder_input_ids, label_ids = batch

            # Move tensors to device
            encoder_input_ids = encoder_input_ids.to(device)
            decoder_input_ids = decoder_input_ids.to(device)
            label_ids = label_ids.to(device)

            # Forward pass
            logits = model(encoder_input_ids, decoder_input_ids)

            # Reshape for loss computation
            batch_size, seq_len, vocab_size = logits.shape
            logits_flat = logits.view(batch_size * seq_len, vocab_size)
            labels_flat = label_ids.view(batch_size * seq_len)

            # Compute loss
            loss = loss_fn(logits_flat, labels_flat)

            # Accumulate loss
            total_loss += loss.item()

    # Return average loss
    average_loss = total_loss / len(dataloader)
    return average_loss


def compute_bleu(
        model: nn.Module,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_len: int = 50,
        strategy: str = "greedy"
) -> float:
    """
    Compute BLEU score on test set.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for test data
        tokenizer: Tokenizer for decoding
        device: Device to use
        max_len: Maximum generation length
        strategy: Decoding strategy ('greedy' or 'beam_search')

    Returns:
        Average BLEU score
    """
    model.eval()
    model.to(device)

    total_bleu = 0.0
    num_samples = 0

    # Smoothing function for BLEU
    smoothing = SmoothingFunction().method1

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Computing BLEU'):
            encoder_input_ids, _, label_ids = batch

            encoder_input_ids = encoder_input_ids.to(device)

            # Generate sequences
            generated_ids = model.generate(
                encoder_input_ids,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_len=max_len,
                strategy=strategy
            )

            # Process each sample in batch
            for i in range(len(encoder_input_ids)):
                # Decode prediction
                pred_ids = generated_ids[i]
                pred_tokens = tokenizer.decode(pred_ids)

                # Remove special tokens from prediction
                pred_tokens = [t for t in pred_tokens
                               if t not in ['<bos>', '<eos>', '<pad>']]

                # Decode ground truth
                label_ids_list = label_ids[i].tolist()
                gt_tokens = tokenizer.decode(label_ids_list)

                # Remove padding from ground truth
                gt_tokens = [t for t in gt_tokens if t != '<pad>']

                # Compute BLEU score
                # Reference should be a list of reference translations
                # We use weights favoring unigrams for short sequences
                if len(pred_tokens) > 0 and len(gt_tokens) > 0:
                    bleu = sentence_bleu(
                        [gt_tokens],  # reference as list
                        pred_tokens,  # hypothesis
                        weights=(1.0, 0, 0, 0),  # unigram only
                        smoothing_function=smoothing
                    )
                    total_bleu += bleu
                    num_samples += 1

    return total_bleu / num_samples if num_samples > 0 else 0.0


@click.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('dev_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
def main(
    train_file: str,
    dev_file: str,
    test_file: str
):
    # ========== Hyperparameters ==========
    # Training hyperparameters
    epochs = 3
    learning_rate = 0.001
    batch_size = 32

    # Sequence length hyperparameters
    max_src_len = 50
    max_tgt_len = 50

    # Model hyperparameters
    d_model = 128
    num_heads = 4
    d_ff = 512
    num_enc_layers = 2
    num_dec_layers = 2
    dropout = 0.1
    max_len = 50

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Build Tokenizer
    print("Building tokenizer from training data...")
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)
    print(f"Vocabulary size: {len(tokenizer.src_vocab)}")

    # Create Datasets
    print("Loading datasets...")
    train_dataset = SeqPairDataset(train_file, tokenizer, max_src_len, max_tgt_len)
    dev_dataset = SeqPairDataset(dev_file, tokenizer, max_src_len, max_tgt_len)
    test_dataset = SeqPairDataset(test_file, tokenizer, max_src_len, max_tgt_len)

    print(f"Train size: {len(train_dataset)}")
    print(f"Dev size: {len(dev_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Model
    print("Initializing model...")
    model = EncoderDecoder(
        src_vocab_size=len(tokenizer.src_vocab),
        tgt_vocab_size=len(tokenizer.src_vocab),  # Same vocab for src and tgt
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_enc_layers=num_enc_layers,
        num_dec_layers=num_dec_layers,
        max_len=max_len,
        dropout=dropout,
        pad_idx=tokenizer.pad_id
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")

    # Initialize Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    # Training Loop
    print("\nStarting training...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'=' * 50}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)

        # Validate
        val_loss = test_epoch(model, dev_loader, loss_fn, device)

        # Print results
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Training Loss:   {train_loss:.4f}")
        print(f"  Validation Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  ✓ New best model saved!")

    # Final Evaluation
    print(f"\n{'=' * 50}")
    print("Final Evaluation on Test Set")
    print(f"{'=' * 50}")

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))

    # Compute test loss
    test_loss = test_epoch(model, test_loader, loss_fn, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Compute BLEU score
    print("\nComputing BLEU score...")
    bleu_score = compute_bleu(model, test_loader, tokenizer, device, max_len, "greedy")
    print(f"BLEU Score: {bleu_score:.4f}")

    # Show Example Output
    print(f"\n{'=' * 50}")
    print("Example Predictions")
    print(f"{'=' * 50}")

    model.eval()
    with torch.no_grad():
        # Get first batch from test set
        batch = next(iter(test_loader))
        encoder_input_ids, _, label_ids = batch
        encoder_input_ids = encoder_input_ids.to(device)

        # Generate predictions
        generated_ids = model.generate(
            encoder_input_ids[:5],  # First 5 samples
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_len=max_len,
            strategy="greedy"
        )

        # Display examples
        for i in range(min(5, len(generated_ids))):
            # Source
            src_ids = encoder_input_ids[i].tolist()
            src_tokens = tokenizer.decode(src_ids)
            src_text = ' '.join([t for t in src_tokens if t not in ['<bos>', '<eos>', '<pad>']])

            # Prediction
            pred_ids = generated_ids[i]
            pred_tokens = tokenizer.decode(pred_ids)
            pred_text = ' '.join([t for t in pred_tokens if t not in ['<bos>', '<eos>', '<pad>']])

            # Ground truth
            gt_ids = label_ids[i].tolist()
            gt_tokens = tokenizer.decode(gt_ids)
            gt_text = ' '.join([t for t in gt_tokens if t != '<pad>'])

            print(f"\nExample {i + 1}:")
            print(f"  Source:     {src_text}")
            print(f"  Prediction: {pred_text}")
            print(f"  Ground Truth: {gt_text}")


if __name__ == "__main__":
    main()

