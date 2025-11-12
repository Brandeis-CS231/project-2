from typing import Callable
import click
import torch
import torch.nn as nn
from dataset import SeqPairDataset
from model import EncoderDecoder
from tokenizer import Tokenizer
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
    
    model = model
    model.train()
    model.to(device=device)
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Forward pass:
        
        batch[0].to(device)
        batch[1].to(device)
        batch[2].to(device)
        logits = model(batch[0], batch[1])
        labels = batch[2]
        # batch_size = logits[0]
        # seq_len = logits[1]
        # vocab_size = logits[2]
        
        logits = logits.flatten(start_dim=0,end_dim=1)
        labels = labels.flatten()

        loss = loss_fn(logits, labels)
        
        # Backward pass:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    average_loss = total_loss/len(dataloader)
    print(f"Average training loss: {average_loss}")
    return average_loss


def test_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> float:
    
    model.eval()
    model.to(device)
    
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            logits = model.forward(batch[0], batch[1])
            labels = batch[2]
            logits = logits.flatten(start_dim=0,end_dim=1)
            labels = labels.flatten()

            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
    average_loss =total_loss/ len(dataloader)
    print(f"Average evaluation loss: {average_loss}")
    
    return average_loss

@click.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('dev_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
def main(
    train_file: str,
    dev_file: str,
    test_file: str
    
    
):
    # 1. Define hyperparameters:
    epochs = 10
    learning_rate = 1e-3
    batch_size = 64
    max_src_length = max_tgt_length = max_length= 50
    d_model = 512
    num_heads = 8
    d_ff = 1024
    num_enc_layers = num_dec_layers = num_layers = 4
    dropout = 0.2
    strategy = 'greedy'
    beam_width = 5
    
    # 2. Build tokenizer:
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)
    
    # 3. Create Datasets:
    train_data = SeqPairDataset(data_file=train_file,
                                tokenizer=tokenizer,
                                max_src_len=20,
                                max_tgt_len=20)
    
    dev_data = SeqPairDataset(data_file=dev_file,
                                tokenizer=tokenizer,
                                max_src_len=20,
                                max_tgt_len=20)
    
    test_data = SeqPairDataset(data_file=test_file,
                                tokenizer=tokenizer,
                                max_src_len=20,
                                max_tgt_len=20)
    
    # 4. Create dataloaders:
    
    train_dataloader = DataLoader(dataset=train_data,
                                    batch_size=batch_size,
                                    shuffle=True
                                    )
    
    dev_dataloader = DataLoader(dataset=dev_data,
                                    batch_size=batch_size,
                                    shuffle=False
                                    )
    
    test_dataloader = DataLoader(dataset=test_data,
                                    batch_size=batch_size,
                                    shuffle=False
                                    )
    
    # 5. Initialize model:
    model = EncoderDecoder(src_vocab_size= train_data.vocab_size,
                            tgt_vocab_size=train_data.vocab_size,
                            d_model=d_model,
                            num_heads=num_heads,
                            d_ff=d_ff,
                            num_enc_layers= num_enc_layers,
                            max_len=max_length,
                            dropout=dropout,
                            pad_idx=tokenizer.pad_id
                            )

    # 6. Initialize optimizer and loss function: 
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    optimizer = torch.optim.Adam(params=model.parameters(), lr = learning_rate)
    
    #7. Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_epoch(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer)
        test_epoch(model=model, dataloader=dev_dataloader, loss_fn= loss_fn)
        
    # Final Evaluation:
    model.eval()
        
    bleu_weights = (1.0,0,0,0)
    scores = []
    special_tokens = ['<bos>','eos','<pad>']
    for batch in tqdm(test_dataloader, desc="Evaluating BLEU Score on test data"):
        references = batch[2]
        hypotheses = model.generate(src_ids=batch[0],
                            bos_id=tokenizer.bos_id,
                            eos_id=tokenizer.eos_id,
                            max_len=max_length,
                            strategy=strategy,
                            beam_width=beam_width
                            )
        
        
        chencherry = SmoothingFunction()
        
        for index in range(len(references)):
            decoded_hypothesis = tokenizer.decode(hypotheses[index])
            hypothesis = [tok for tok in decoded_hypothesis if tok not in special_tokens]
            
            decoded_labels = tokenizer.decode(references[index].tolist())
            reference = [tok for tok in decoded_labels if tok not in special_tokens]


            if index ==0:
                print(f"Sample hypothesis: {hypothesis}")
                print(f"Sample reference: {reference}")
                        
        
            score = sentence_bleu(references=reference,
                                hypothesis=hypothesis,
                                weights=bleu_weights,
                                smoothing_function=chencherry.method7
                                )
            scores.append(score)
        
    print(f'Avg BLEU score: {sum(scores)/len(test_data)}')
        
    
    
    
if __name__ == "__main__":
    main()
