"""
Experiments runner supporting:
 - exp1: positional encoding comparison (sinusoidal vs learnable)
 - exp2: decoding algorithms (greedy vs beam search)
 - exp3: architecture sweep (heads x depth)
Designed to be MPS/CUDA/CPU aware and to avoid state_dict loading mismatches.
"""
import argparse
import json
import os
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tokenizer import Tokenizer
from dataset import SeqPairDataset
from model import EncoderDecoder

# Optional plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except Exception:
    MATPLOTLIB = False

# -------------------------
# Config (tweak for full runs)
# -------------------------
CFG = {
    "seed": 2025,
    "batch_size": 64,
    "exp1_epochs": 4,     # increase for final experiments
    "exp2_examples": 8,
    "exp3_epochs": 2,     # quick checks
    "d_model": 256,
    "num_heads": 4,
    "d_ff": 512,
    "num_layers": 2,
    "dropout": 0.1,
    "max_len": 100,
    "save_dir": "results",
    "progress_bar": True,  # show batch progress during epochs
}
Path(CFG["save_dir"]).mkdir(exist_ok=True)


# -------------------------
# Device utilities
# -------------------------
def get_device():
    # Prefer MPS (Apple Silicon) -> CUDA -> CPU
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = get_device()


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


# -------------------------
# Basic helpers
# -------------------------
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def compute_unigram_bleu(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    ch = SmoothingFunction()
    try:
        return sentence_bleu([ref_tokens], pred_tokens, weights=(1.0, 0, 0, 0), smoothing_function=ch.method1)
    except Exception:
        return 0.0


# -------------------------
# Data loaders builder
# -------------------------
def build_dataloaders(batch_size: int = None, max_src_len: int = None, max_tgt_len: int = None):
    batch_size = batch_size or CFG["batch_size"]
    max_src_len = max_src_len or CFG["max_len"]
    max_tgt_len = max_tgt_len or CFG["max_len"]

    tok = Tokenizer()
    tok.from_file("data/train.json")

    train_ds = SeqPairDataset("data/train.json", tok, max_src_len, max_tgt_len)
    dev_ds = SeqPairDataset("data/dev.json", tok, max_src_len, max_tgt_len)
    test_ds = SeqPairDataset("data/test.json", tok, max_src_len, max_tgt_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
# experiments.py
"""
Experiments runner supporting:
 - exp1: positional encoding comparison (sinusoidal vs learnable)
 - exp2: decoding algorithms (greedy vs beam search)
 - exp3: architecture sweep (heads x depth)
Designed to be MPS/CUDA/CPU aware and to avoid state_dict loading mismatches.
"""
import argparse
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from tokenizer import Tokenizer
from dataset import SeqPairDataset
from model import EncoderDecoder

# Optional plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB = True
except Exception:
    MATPLOTLIB = False

# -------------------------
# Config
# -------------------------
CFG = {
    "seed": 2025,
    "batch_size": 64,
    "exp1_epochs": 4,
    "exp2_examples": 8,
    "exp3_epochs": 2,
    "d_model": 256,
    "num_heads": 4,
    "d_ff": 512,
    "num_layers": 2,
    "dropout": 0.1,
    "max_len": 100,
    "save_dir": "results",
    "progress_bar": True,
}
Path(CFG["save_dir"]).mkdir(exist_ok=True)

# -------------------------
# Device utilities
# -------------------------
def get_device():
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)

# -------------------------
# Basic helpers
# -------------------------
def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def compute_unigram_bleu(pred_tokens: List[str], ref_tokens: List[str]) -> float:
    ch = SmoothingFunction()
    try:
        return sentence_bleu([ref_tokens], pred_tokens, weights=(1.0, 0, 0, 0), smoothing_function=ch.method1)
    except Exception:
        return 0.0

# -------------------------
# Data loaders builder
# -------------------------
def build_dataloaders(batch_size: int = None, max_src_len: int = None, max_tgt_len: int = None):
    batch_size = batch_size or CFG["batch_size"]
    max_src_len = max_src_len or CFG["max_len"]
    max_tgt_len = max_tgt_len or CFG["max_len"]

    tok = Tokenizer()
    tok.from_file("data/train.json")

    train_ds = SeqPairDataset("data/train.json", tok, max_src_len, max_tgt_len)
    dev_ds = SeqPairDataset("data/dev.json", tok, max_src_len, max_tgt_len)
    test_ds = SeqPairDataset("data/test.json", tok, max_src_len, max_tgt_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return tok, train_loader, dev_loader, test_loader

# -------------------------
# Train / Eval loops
# -------------------------
def train_one_epoch(model: nn.Module, dataloader: DataLoader, optimizer, loss_fn, device: str) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    iterator = dataloader
    if CFG["progress_bar"]:
        iterator = tqdm(dataloader, desc="Training", leave=False)
    for enc_inp, dec_inp, labels in iterator:
        enc_inp, dec_inp, labels = enc_inp.to(device), dec_inp.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(enc_inp, dec_inp)
        B, T, V = logits.size()
        loss = loss_fn(logits.view(B * T, V), labels.view(B * T))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item())
        batches += 1
    return total_loss / max(1, batches)

def eval_loss(model: nn.Module, dataloader: DataLoader, loss_fn, device: str) -> float:
    model.eval()
    total_loss = 0.0
    batches = 0
    with torch.no_grad():
        for enc_inp, dec_inp, labels in dataloader:
            enc_inp, dec_inp, labels = enc_inp.to(device), dec_inp.to(device), labels.to(device)
            logits = model(enc_inp, dec_inp)
            total_loss += float(loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1)).item())
            batches += 1
    return total_loss / max(1, batches)

def eval_unigram_bleu(model: nn.Module, tok: Tokenizer, dataloader: DataLoader, device: str) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for enc_inp, dec_inp, labels in dataloader:
            enc_inp = enc_inp.to(device)
            gen_list = model.generate(enc_inp, bos_id=tok.bos_id, eos_id=tok.eos_id, max_len=CFG["max_len"], strategy="greedy")
            for i, gen_ids in enumerate(gen_list):
                pred_tokens = [t for t in tok.decode(gen_ids) if t not in tok.config.special_tokens]
                ref_ids = labels[i].cpu().tolist()
                ref_tokens = [t for t in tok.decode([rid for rid in ref_ids if rid != tok.pad_id]) if t not in tok.config.special_tokens]
                total += compute_unigram_bleu(pred_tokens, ref_tokens)
                n += 1
    return total / max(1, n)

# -------------------------
# Learnable positional encoding
# -------------------------
class LearnablePosEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int):
        super().__init__()
        self.embedding = nn.Embedding(seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        pos_ids = torch.arange(0, T, device=x.device).unsqueeze(0)
        return x + self.embedding(pos_ids)

# -------------------------
# Experiment 1 - Positional encoding comparison
# -------------------------
def run_exp1(epochs: int = None, fast: bool = False):
    print("\n=== Experiment 1: Sinusoidal vs Learnable Positional Encoding ===")
    epochs = epochs if epochs is not None else (4 if fast else CFG["exp1_epochs"])
    set_seed(CFG["seed"])
    device = DEVICE

    # Reduce model size for fast mode
    d_model, num_heads, num_layers = (128, 2, 1) if fast else (CFG["d_model"], CFG["num_heads"], CFG["num_layers"])

    tok, train_loader, dev_loader, test_loader = build_dataloaders(batch_size=32 if fast else CFG["batch_size"])

    def make_model():
        return EncoderDecoder(
            src_vocab_size=len(tok.src_vocab),
            tgt_vocab_size=len(tok.src_vocab),
            d_model=d_model,
            num_heads=num_heads,
            d_ff=CFG["d_ff"],
            num_enc_layers=num_layers,
            num_dec_layers=num_layers,
            max_len=CFG["max_len"],
            dropout=CFG["dropout"],
            pad_idx=tok.pad_id
        ).to(device)

    baseline = make_model()
    alt = make_model()
    emb_dim = alt.encoder.token_emb.embedding_dim
    seq_len = CFG["max_len"]
    alt.encoder.pos_emb = LearnablePosEncoding(emb_dim, seq_len).to(device)
    alt.decoder.pos_emb = LearnablePosEncoding(emb_dim, seq_len).to(device)

    def train_model(m: nn.Module, name: str):
        optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
        loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
        best_state, best_bleu = None, -1.0
        history = {"train": [], "val_loss": [], "val_bleu": []}
        for ep in range(1, epochs + 1):
            start = time.time()
            tr_loss = train_one_epoch(m, train_loader, optimizer, loss_fn, device)
            val_loss = eval_loss(m, dev_loader, loss_fn, device)
            val_bleu = eval_unigram_bleu(m, tok, dev_loader, device)
            elapsed = time.time() - start
            history["train"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["val_bleu"].append(val_bleu)
            print(f"[{name}] Epoch {ep}/{epochs} train={tr_loss:.4f} val_loss={val_loss:.4f} val_bleu={val_bleu:.4f} time={elapsed:.1f}s")
            if val_bleu > best_bleu:
                best_bleu = val_bleu
                best_state = deepcopy(m.state_dict())
        if best_state:
            m.load_state_dict(best_state)
        return m, history, best_bleu

    print("Training baseline (sinusoidal)...")
    baseline_trained, baseline_hist, baseline_best = train_model(baseline, "sinusoidal")
    print("\nTraining alternative (learnable)...")
    alt_trained, alt_hist, alt_best = train_model(alt, "learnable")

    baseline_test = eval_unigram_bleu(baseline_trained, tok, test_loader, device)
    alt_test = eval_unigram_bleu(alt_trained, tok, test_loader, device)

    out_dir = Path(CFG["save_dir"])
    saved = {}
    baseline_path = out_dir / "exp1_sinusoidal.pt"
    alt_path = out_dir / "exp1_learnable.pt"
    torch.save(baseline_trained.state_dict(), baseline_path)
    torch.save(alt_trained.state_dict(), alt_path)
    saved["sinusoidal"] = str(baseline_path)
    saved["learnable"] = str(alt_path)
    chosen = "learnable" if alt_test >= baseline_test else "sinusoidal"

    res = {
        "tok_vocab_size": len(tok.src_vocab),
        "baseline": {"params": count_params(baseline_trained), "history": baseline_hist, "test_bleu": baseline_test, "checkpoint": str(baseline_path)},
        "learnable": {"params": count_params(alt_trained), "history": alt_hist, "test_bleu": alt_test, "checkpoint": str(alt_path)},
        "chosen": chosen,
        "saved_models": saved
    }

    with open(out_dir / "exp1_posencoding.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"Saved Exp1 results to {out_dir / 'exp1_posencoding.json'} (best: {chosen})")

    # Small test outputs
    examples = []
    with torch.no_grad():
        for i, (enc_inp, dec_inp, labels) in enumerate(test_loader):
            if i >= CFG["exp2_examples"]:
                break
            enc_inp = enc_inp.to(device)
            bgen = baseline_trained.generate(enc_inp, bos_id=tok.bos_id, eos_id=tok.eos_id, max_len=CFG["max_len"], strategy="greedy")[0]
            agen = alt_trained.generate(enc_inp, bos_id=tok.bos_id, eos_id=tok.eos_id, max_len=CFG["max_len"], strategy="greedy")[0]
            src = " ".join(tok.decode(enc_inp[0].cpu().tolist()))
            ref = " ".join([w for w in tok.decode([int(x) for x in labels[0].tolist()]) if w not in tok.config.special_tokens])
            examples.append({
                "src": src,
                "ref": ref,
                "baseline": " ".join([w for w in tok.decode(bgen) if w not in tok.config.special_tokens]),
                "learnable": " ".join([w for w in tok.decode(agen) if w not in tok.config.special_tokens])
            })
    with open(out_dir / "exp1_examples.json", "w", encoding="utf-8") as fh:
        json.dump(examples, fh, indent=2)

    # Optional plot
    if MATPLOTLIB:
        epochs_range = list(range(1, epochs + 1))
        plt.figure(figsize=(8, 4))
        plt.plot(epochs_range, baseline_hist["train"], 'o-', label="baseline train")
        plt.plot(epochs_range, baseline_hist["val_loss"], 'o--', label="baseline val")
        plt.plot(epochs_range, alt_hist["train"], 's-', label="learnable train")
        plt.plot(epochs_range, alt_hist["val_loss"], 's--', label="learnable val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.xticks(epochs_range)
        plt.legend()
        plt.title("Exp1 Loss Curves (Fast Mode)" if fast else "Exp1 Loss Curves")
        plt.tight_layout()
        plt.savefig(out_dir / "exp1_loss_curves.png")
        print("Saved loss curves")

    return res

# -------------------------
# Experiment 2 - Decoding algorithms
# -------------------------
def run_exp2():
    print("\n=== Experiment 2: Decoding Algorithms ===")
    set_seed(CFG["seed"])
    device = DEVICE

    tok, _, _, test_loader = build_dataloaders(batch_size=1)

    # Load Exp1 results
    exp1_json = Path(CFG["save_dir"]) / "exp1_posencoding.json"
    if not exp1_json.exists():
        raise FileNotFoundError(f"{exp1_json} not found. Run exp1 first.")
    with open(exp1_json, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    chosen = meta.get("chosen")
    best_model_path = meta["saved_models"][chosen]
    print(f"Exp1 chosen positional encoding: {chosen}")
    print(f"Loading checkpoint from: {best_model_path}")

    # Build model matching Exp1
    model_inst = EncoderDecoder(
        src_vocab_size=len(tok.src_vocab),
        tgt_vocab_size=len(tok.src_vocab),
        d_model=CFG["d_model"],
        num_heads=CFG["num_heads"],
        d_ff=CFG["d_ff"],
        num_enc_layers=CFG["num_layers"],
        num_dec_layers=CFG["num_layers"],
        max_len=CFG["max_len"],
        dropout=CFG["dropout"],
        pad_idx=tok.pad_id
    ).to(device)

    if chosen == "learnable":
        emb_dim = model_inst.encoder.token_emb.embedding_dim
        seq_len = CFG["max_len"]
        model_inst.encoder.pos_emb = LearnablePosEncoding(emb_dim, seq_len).to(device)
        model_inst.decoder.pos_emb = LearnablePosEncoding(emb_dim, seq_len).to(device)

    model_inst.load_state_dict(torch.load(best_model_path, map_location=device))
    model_inst.eval()

    strategies = [("greedy", None), ("beam_search", 3), ("beam_search", 5), ("beam_search", 10)]
    results, examples = {}, {}

    for strategy, beam in strategies:
        total_bleu, total_time, total_len, n = 0.0, 0.0, 0, 0
        example_list = []
        for i, (enc_inp, dec_inp, labels) in enumerate(test_loader):
            enc_inp = enc_inp.to(device)
            t0 = time.time()
            gen_list = model_inst.generate(enc_inp, bos_id=tok.bos_id, eos_id=tok.eos_id, max_len=CFG["max_len"], strategy=strategy, beam_width=beam)
            t1 = time.time()
            pred_ids = gen_list[0]
            pred_tokens = [t for t in tok.decode(pred_ids) if t not in tok.config.special_tokens]
            ref_ids = labels[0].cpu().tolist()
            ref_tokens = [t for t in tok.decode([rid for rid in ref_ids if rid != tok.pad_id]) if t not in tok.config.special_tokens]
            total_bleu += compute_unigram_bleu(pred_tokens, ref_tokens)
            total_time += t1 - t0
            total_len += len(pred_tokens)
            n += 1

            if i < CFG["exp2_examples"]:
                src = " ".join(tok.decode(enc_inp[0].cpu().tolist()))
                example_list.append({"src": src, "ref": " ".join(ref_tokens), "pred": " ".join(pred_tokens)})

        key = f"{strategy}" + (f"_b{beam}" if beam else "")
        results[key] = {"avg_bleu": total_bleu / max(1, n), "time_per_sample": total_time / max(1, n), "avg_len": total_len / max(1, n), "n": n}
        examples[key] = example_list
        print(f"{key} -> BLEU={results[key]['avg_bleu']:.4f}, time/sample={results[key]['time_per_sample']:.4f}s")

    out = Path(CFG["save_dir"]) / "exp2_decoding.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"results": results, "examples": examples}, fh, indent=2)
    print(f"Saved Exp2 results to {out}")
    return results

# -------------------------
# Experiment 3 - Architecture sweep
# -------------------------
def run_exp3():
    print("\n=== Experiment 3: Architecture Variants (heads x depth) ===")
    set_seed(CFG["seed"])
    device = DEVICE

    tok, train_loader, dev_loader, test_loader = build_dataloaders(batch_size=CFG["batch_size"])

    heads_list = [2, 4, 8]
    depth_list = [1, 2, 4]
    results = {}

    for heads in heads_list:
        for depth in depth_list:
            print(f"Training model with {heads} heads and {depth} layers")
            model_inst = EncoderDecoder(
                src_vocab_size=len(tok.src_vocab),
                tgt_vocab_size=len(tok.src_vocab),
                d_model=CFG["d_model"],
                num_heads=heads,
                d_ff=CFG["d_ff"],
                num_enc_layers=depth,
                num_dec_layers=depth,
                max_len=CFG["max_len"],
                dropout=CFG["dropout"],
                pad_idx=tok.pad_id
            ).to(device)
            optimizer = torch.optim.Adam(model_inst.parameters(), lr=1e-4)
            loss_fn = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
            best_state, best_loss = None, float("inf")
            for ep in range(CFG["exp3_epochs"]):
                tr_loss = train_one_epoch(model_inst, train_loader, optimizer, loss_fn, device)
                val_loss = eval_loss(model_inst, dev_loader, loss_fn, device)
                print(f"Epoch {ep+1}: train={tr_loss:.4f} val={val_loss:.4f}")
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_state = deepcopy(model_inst.state_dict())
            if best_state:
                model_inst.load_state_dict(best_state)
            results[f"h{heads}_d{depth}"] = {"val_loss": best_loss, "params": count_params(model_inst)}

    out = Path(CFG["save_dir"]) / "exp3_arch.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"Saved Exp3 results to {out}")
    return results

# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3], help="Which experiment to run")
    parser.add_argument("--fast", action="store_true", help="Run fast mode (Exp1 only)")
    args = parser.parse_args()

    if args.exp == 1:
        run_exp1(fast=args.fast)
    elif args.exp == 2:
        run_exp2()
    elif args.exp == 3:
        run_exp3()

if __name__ == "__main__":
    main()