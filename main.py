from typing import Callable
import click
import random
import torch
import torch.nn as nn
from dataset import SeqPairDataset
from model import EncoderDecoder
from tokenizer import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction



learning_rate = 1e-3
batch_size = 64
strategy = 'greedy'
beam_width = 5

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> float:
    
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
        for batch in tqdm(dataloader, desc = f'Evaluating hyperparameters'):
            batch[0].to(device)
            batch[1].to(device)
            logits = model.forward(batch[0], batch[1])
            labels = batch[2]
            logits = logits.flatten(start_dim=0,end_dim=1)
            labels = labels.flatten()

            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
    average_loss =total_loss/ len(dataloader)
    print(f"Average evaluation loss: {average_loss}")
    
    return average_loss

def grid_search(tokenizer,
                train_dataloader,
                dev_dataloader,
                test_dataloader
                ):
    
    #grid search params:
    epochs = 1
    model_depths = [1,2,4]
    model_dimensions = [128,256]
    attention_heads = [2,4,8]
    feedfwd_dimensions = [256,512]
    dropout = 0.2
    batch_size =64
    max_length = 100
    src_vocab_size = len(tokenizer.src_vocab)
    tgt_vocab_size = len(tokenizer.tgt_vocab)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    test_number = 1
    total_tests = len(model_depths)*len(model_dimensions)*len(attention_heads)*len(feedfwd_dimensions)
    
    # Begin grid search
    for depth in model_depths:
        for dim in model_dimensions:
            for heads in attention_heads:
                for ff in feedfwd_dimensions:
                    
                    print(f"Running test {test_number} of {total_tests}:")
                    # create dict to log results 
                    results = {'model_depth':depth,
                                'model_dimension':dim,
                                'attention_heads':heads,
                                'ff_dimension':ff,
                                'dropout':dropout,
                                'avg_losses':{'average_train_losses':[],
                                            'average_eval_losses':[]
                                            },
                                'BLEU':0
                                }
                    
                    model = EncoderDecoder(pos_embed='enc',
                                            src_vocab_size= src_vocab_size,
                                                tgt_vocab_size=tgt_vocab_size,
                                                d_model=dim,
                                                num_heads=heads,
                                                d_ff=ff,
                                                num_enc_layers= depth,
                                                max_len=max_length,
                                                dropout=dropout,
                                                pad_idx=tokenizer.pad_id
                                                )
                    optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
                    
                    # Training Loop:
                    for epoch in range(epochs):
                        print(f"\tEpoch {epoch+1}")
                        avg_train_loss = train_epoch(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer)
                        avg_eval_loss = test_epoch(model=model, dataloader=dev_dataloader, loss_fn= loss_fn)
                        results['avg_losses']['average_train_losses'].append(avg_train_loss)
                        results['avg_losses']['average_eval_losses'].append(avg_eval_loss)
                        
                        
                    # Final Evaluation:
                    model.eval()

                    bleu_weights = (1.0,0,0,0)
                    scores = []
                    examples = 0
                    special_tokens = ['<bos>','<eos>','<pad>']
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
                    qa = random.randrange(0,len(references))
                    for index in range(len(references)):
                        decoded_hypothesis = tokenizer.decode(hypotheses[index])
                        hypothesis = [tok for tok in decoded_hypothesis if tok not in special_tokens]
                        
                        decoded_labels = tokenizer.decode(references[index].tolist())
                        reference = [tok for tok in decoded_labels if tok not in special_tokens]

                        # Random qualitative assessment 
                        if index == qa:
                            print(f"Sample hypothesis: {hypothesis}")
                            print(f"Sample reference: {reference}")
            

                        score = sentence_bleu(references=reference,
                                            hypothesis=hypothesis,
                                            weights=bleu_weights,
                                            smoothing_function=chencherry.method7
                                            )
                        
                        scores.append(score)
                        examples += 1
                    results['BLEU'] = sum(scores)/examples
                    print(f'Avg BLEU score: {sum(scores)/examples}')
                    print(results)
                    test_number += 1
                    
                    results_log = 'grid_search_results.txt'
                    with open(results_log, 'a') as file:
                        file.write(str(results)+'\n')
                        
def experiment1(tokenizer,
                train_dataloader,
                dev_dataloader,
                test_dataloader
                ):
    '''compare positional encodings vs learnable pos embeddings'''
    
    epochs = 10
    model_depth = 4
    model_dim = 128
    attention_heads = 2
    ff_dimension = 512
    dropout = 0.2
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    '''BASELINE: POSITIONAL ENCODING WITH GRIDSEARCH HYPERPARAMETERS'''
    
    model = EncoderDecoder(pos_embed='enc',
                            src_vocab_size=len(tokenizer.src_vocab),
                            tgt_vocab_size=len(tokenizer.tgt_vocab),
                            d_model=model_dim,
                            num_heads=attention_heads,
                            d_ff=ff_dimension,
                            num_enc_layers= model_depth,
                            max_len=100,
                            dropout=dropout,
                            pad_idx=tokenizer.pad_id
                            )
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
    
    # Training Loop:
    training_losses=[] # record losses to generate loss curve for report
    for epoch in range(epochs):
        print(f"\tEpoch {epoch+1}")
        avg_train_loss = train_epoch(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer)
        avg_eval_loss = test_epoch(model=model, dataloader=dev_dataloader, loss_fn= loss_fn)
        training_losses.append((avg_train_loss,avg_eval_loss))
        print(f'Average training loss: {avg_train_loss}\nAverage eval loss: {avg_eval_loss}')
        
        
    # Final Evaluation:
    model.eval()

    bleu_weights = (1.0,0,0,0)
    scores = []
    examples = 0
    special_tokens = ['<bos>','<eos>','<pad>']
    for batch in tqdm(test_dataloader, desc="Evaluating BLEU Score on test data"):
        references = batch[2]
        hypotheses = model.generate(src_ids=batch[0],
                            bos_id=tokenizer.bos_id,
                            eos_id=tokenizer.eos_id,
                            max_len=100,
                            strategy=strategy,
                            beam_width=beam_width
                            )


    chencherry = SmoothingFunction()
    qa = random.randrange(0,len(references))
    for index in range(len(references)):
        decoded_hypothesis = tokenizer.decode(hypotheses[index])
        hypothesis = [tok for tok in decoded_hypothesis if tok not in special_tokens]
        
        decoded_labels = tokenizer.decode(references[index].tolist())
        reference = [tok for tok in decoded_labels if tok not in special_tokens]

        # Random qualitative assessment 
        if index == qa:
            print(f"Sample hypothesis: {hypothesis}")
            print(f"Sample reference: {reference}")


        score = sentence_bleu(references=reference,
                            hypothesis=hypothesis,
                            weights=bleu_weights,
                            smoothing_function=chencherry.method7
                            )
        
        scores.append(score)
        examples += 1
    bleu = sum(scores)/examples
    print(f'Avg BLEU score: {bleu}')
    
    results_log = 'experiment1.txt'
    with open(results_log, 'a') as file:
        file.write('Positional Encoding results:\n'+str(training_losses)+f'\n{bleu}')
    
    '''EXPERIMENT: LEARNABLE POSITIONAL EMBEDDINGS:'''
    
    model = EncoderDecoder(pos_embed='emb',
                            src_vocab_size=len(tokenizer.src_vocab),
                            tgt_vocab_size=len(tokenizer.tgt_vocab),
                            d_model=model_dim,
                            num_heads=attention_heads,
                            d_ff=ff_dimension,
                            num_enc_layers= model_depth,
                            max_len=100,
                            dropout=dropout,
                            pad_idx=tokenizer.pad_id
                            )
    
    optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
    
    # Training Loop:
    training_losses=[] # record losses to generate loss curve for report
    for epoch in range(epochs):
        print(f"\tEpoch {epoch+1}")
        avg_train_loss = train_epoch(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer)
        avg_eval_loss = test_epoch(model=model, dataloader=dev_dataloader, loss_fn= loss_fn)
        training_losses.append((avg_train_loss,avg_eval_loss))
        print(f'Average training loss: {avg_train_loss}\nAverage eval loss: {avg_eval_loss}')
        
        
    # Final Evaluation:
    model.eval()

    bleu_weights = (1.0,0,0,0)
    scores = []
    examples = 0
    special_tokens = ['<bos>','<eos>','<pad>']
    for batch in tqdm(test_dataloader, desc="Evaluating BLEU Score on test data"):
        references = batch[2]
        hypotheses = model.generate(src_ids=batch[0],
                            bos_id=tokenizer.bos_id,
                            eos_id=tokenizer.eos_id,
                            max_len=100,
                            strategy=strategy,
                            beam_width=beam_width
                            )


    chencherry = SmoothingFunction()
    qa = random.randrange(0,len(references))
    for index in range(len(references)):
        decoded_hypothesis = tokenizer.decode(hypotheses[index])
        hypothesis = [tok for tok in decoded_hypothesis if tok not in special_tokens]
        
        decoded_labels = tokenizer.decode(references[index].tolist())
        reference = [tok for tok in decoded_labels if tok not in special_tokens]

        # Random qualitative assessment 
        if index == qa:
            print(f"Sample hypothesis: {hypothesis}")
            print(f"Sample reference: {reference}")


        score = sentence_bleu(references=reference,
                            hypothesis=hypothesis,
                            weights=bleu_weights,
                            smoothing_function=chencherry.method7
                            )
        
        scores.append(score)
        examples += 1
    bleu = sum(scores)/examples
    print(f'Avg BLEU score: {bleu}')
    
    results_log = 'experiment1.txt'
    with open(results_log, 'a') as file:
        file.write('Positional Embeddings Results:\n'+str(training_losses)+f'\n{bleu}')
        
def experiment2():
    pass

def experiment3(tokenizer,
                train_dataloader,
                dev_dataloader,
                test_dataloader
):
    
    with open('experiment3.txt', 'a') as file:
        file.write("Experiment 3: Model Architecture Variants\n")
        
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)
    
    '''A) NUM ATTENTION HEADS'''
    with open('experiment3.txt', 'a') as file:
        file.write("Experiment 3a: Number of Attention Heads:\n")
        
    epochs = 10
    model_depth = 4
    model_dim = 128
    ff_dimension = 512
    dropout = 0.2
    
    attention_heads = [2,4,8]
    
    for attn in attention_heads:
        model = EncoderDecoder(pos_embed='enc',
                                src_vocab_size=len(tokenizer.src_vocab),
                                tgt_vocab_size=len(tokenizer.tgt_vocab),
                                d_model=model_dim,
                                num_heads=attn,
                                d_ff=ff_dimension,
                                num_enc_layers= model_depth,
                                max_len=100,
                                dropout=dropout,
                                pad_idx=tokenizer.pad_id
                                )
    
        optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
    
        # Training Loop:
        training_losses=[] # record losses to generate loss curve for report
        for epoch in range(epochs):
            print(f"\tEpoch {epoch+1}")
            avg_train_loss = train_epoch(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer)
            avg_eval_loss = test_epoch(model=model, dataloader=dev_dataloader, loss_fn= loss_fn)
            training_losses.append((avg_train_loss,avg_eval_loss))
            print(f'Average training loss: {avg_train_loss}\nAverage eval loss: {avg_eval_loss}')
            
        
        # Final Evaluation:
        model.eval()

        bleu_weights = (1.0,0,0,0)
        scores = []
        examples = 0
        special_tokens = ['<bos>','<eos>','<pad>']
        for batch in tqdm(test_dataloader, desc="Evaluating BLEU Score on test data"):
            references = batch[2]
            hypotheses = model.generate(src_ids=batch[0],
                                bos_id=tokenizer.bos_id,
                                eos_id=tokenizer.eos_id,
                                max_len=100,
                                strategy=strategy,
                                beam_width=beam_width
                                )

        chencherry = SmoothingFunction()
        qa = random.randrange(0,len(references))
        for index in range(len(references)):
            decoded_hypothesis = tokenizer.decode(hypotheses[index])
            hypothesis = [tok for tok in decoded_hypothesis if tok not in special_tokens]
            
            decoded_labels = tokenizer.decode(references[index].tolist())
            reference = [tok for tok in decoded_labels if tok not in special_tokens]

            # Random qualitative assessment 
            if index == qa:
                print(f"Sample hypothesis: {hypothesis}")
                print(f"Sample reference: {reference}")


            score = sentence_bleu(references=reference,
                                hypothesis=hypothesis,
                                weights=bleu_weights,
                                smoothing_function=chencherry.method7
                                )
            
            scores.append(score)
            examples += 1
        bleu = sum(scores)/examples
        print(f'Avg BLEU score: {bleu}')
        
        results_log = 'experiment3.txt'
        with open(results_log, 'a') as file:
            file.write(f'{attn} Attention Heads:\n'+str(training_losses)+f'\nBLEU: {bleu}\n')
    
    '''B) MODEL ENCODER/DECODER DEPTH'''
    with open('experiment3.txt', 'a') as file:
        file.write("Experiment 3b: Encoder/Decoder Depth:\n")
        
    epochs = 10
    model_dim = 128
    attention_heads = 2
    ff_dimension = 512
    dropout = 0.2
    
    model_depths = [1,2,4]
    
    for depth in model_depths:
        model = EncoderDecoder(pos_embed='enc',
                                src_vocab_size=len(tokenizer.src_vocab),
                                tgt_vocab_size=len(tokenizer.tgt_vocab),
                                d_model=model_dim,
                                num_heads=attention_heads,
                                d_ff=ff_dimension,
                                num_enc_layers= depth,
                                max_len=100,
                                dropout=dropout,
                                pad_idx=tokenizer.pad_id
                                )
    
        optimizer = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
    
        # Training Loop:
        training_losses=[] # record losses to generate loss curve for report
        for epoch in range(epochs):
            print(f"\tEpoch {epoch+1}")
            avg_train_loss = train_epoch(model=model,dataloader=train_dataloader,loss_fn=loss_fn,optimizer=optimizer)
            avg_eval_loss = test_epoch(model=model, dataloader=dev_dataloader, loss_fn= loss_fn)
            training_losses.append((avg_train_loss,avg_eval_loss))
            print(f'Average training loss: {avg_train_loss}\nAverage eval loss: {avg_eval_loss}')
            
        
        # Final Evaluation:
        model.eval()

        bleu_weights = (1.0,0,0,0)
        scores = []
        examples = 0
        special_tokens = ['<bos>','<eos>','<pad>']
        for batch in tqdm(test_dataloader, desc="Evaluating BLEU Score on test data"):
            references = batch[2]
            hypotheses = model.generate(src_ids=batch[0],
                                bos_id=tokenizer.bos_id,
                                eos_id=tokenizer.eos_id,
                                max_len=100,
                                strategy=strategy,
                                beam_width=beam_width
                                )

        chencherry = SmoothingFunction()
        qa = random.randrange(0,len(references))
        for index in range(len(references)):
            decoded_hypothesis = tokenizer.decode(hypotheses[index])
            hypothesis = [tok for tok in decoded_hypothesis if tok not in special_tokens]
            
            decoded_labels = tokenizer.decode(references[index].tolist())
            reference = [tok for tok in decoded_labels if tok not in special_tokens]

            # Random qualitative assessment 
            if index == qa:
                print(f"Sample hypothesis: {hypothesis}")
                print(f"Sample reference: {reference}")


            score = sentence_bleu(references=reference,
                                hypothesis=hypothesis,
                                weights=bleu_weights,
                                smoothing_function=chencherry.method7
                                )
            
            scores.append(score)
            examples += 1
        bleu = sum(scores)/examples
        print(f'Avg BLEU score: {bleu}')
        
        results_log = 'experiment3.txt'
        with open(results_log, 'a') as file:
            file.write(f'{depth} Encoding/Decoding layers:\n'+str(training_losses)+f'\nBLEU: {bleu}\n')
    


@click.command()
@click.argument('train_file', type=click.Path(exists=True))
@click.argument('dev_file', type=click.Path(exists=True))
@click.argument('test_file', type=click.Path(exists=True))
def main(
    train_file: str,
    dev_file: str,
    test_file: str
    
    
):

    # Build tokenizer:
    tokenizer = Tokenizer()
    tokenizer.from_file(train_file)
    
    # Create Datasets:
    
    batch_size = 64
    max_len = 100
    
    train_data = SeqPairDataset(data_file=train_file,
                                tokenizer=tokenizer,
                                max_src_len=max_len,
                                max_tgt_len=max_len)
    
    train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True
                                )
    
    dev_data = SeqPairDataset(data_file=dev_file,
                                tokenizer=tokenizer,
                                max_src_len=max_len,
                                max_tgt_len=max_len)
    
    dev_dataloader = DataLoader(dataset=dev_data,
                                    batch_size=batch_size,
                                    shuffle=False
                                    )
    
    test_data = SeqPairDataset(data_file=test_file,
                                tokenizer=tokenizer,
                                max_src_len=max_len,
                                max_tgt_len=max_len)

    test_dataloader = DataLoader(dataset=test_data,
                                    batch_size=batch_size,
                                    shuffle=False
                                    )
    
    experiment3(tokenizer=tokenizer,
                train_dataloader=train_dataloader,
                dev_dataloader=dev_dataloader,
                test_dataloader=test_dataloader
                )
    
if __name__ == "__main__":
    main()
