"""
calc_bleu.py

Parse a file where each sample is a small block like:

Input: ...
Reference: ...
Prediction: ...
BLEU Score: 0.8000
Decoding Strategy: greedy (Beam Width: 5)
Sequence Length: 20
Time Taken: ...
--------------------------------------------------

Compute average BLEU and average sequence length for greedy vs beam.
"""
# find and count tokens in the "Prediction:" field (handles multi-line predictions)

import os
import re
import sys
from statistics import mean

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "exp_two_results", "bleu_details.txt")
INPUT_PATH = os.path.abspath("C:\\Users\\peter\\coding\\brandeis\\cosi231\\project-2\\exp_two_results\\model5_0.001_32_50_50_256_8_256_4_4_0.1\\bleu_details.txt")
RE_BLEU = re.compile(r'BLEU\s*Score[:\s]*([0-9]+(?:\.[0-9]+)?)', re.I)
RE_LEN = re.compile(r'Sequence\s*Length[:\s]*([0-9]+)', re.I)
RE_DEC = re.compile(r'Decoding\s*Strategy[:\s]*(.*)', re.I)

RE_PRED = re.compile(r'^\s*Prediction[:\s]*(.*)$', re.I)
_RE_FIELD_START = re.compile(r'^\s*(Input:|Reference:|BLEU\s*Score|Decoding\s*Strategy|Sequence\s*Length|Time\s*Taken)\b', re.I)

def count_prediction_tokens(block_lines):
    """
    Return the number of whitespace-separated tokens in the Prediction entry for a block,
    or None if no prediction was found.
    Handles predictions that span multiple lines until the next recognized field.
    """
    pred_parts = []
    in_pred = False
    for ln in block_lines:
        if not in_pred:
            m = RE_PRED.search(ln)
            if m:
                in_pred = True
                part = m.group(1).strip()
                if part:
                    pred_parts.append(part)
                continue
        else:
            # already inside a Prediction; stop if we hit another known field
            if _RE_FIELD_START.search(ln) or (ln.strip() and all(ch == '-' for ch in ln.strip())):
                break
            pred_parts.append(ln.strip())

    if not pred_parts:
        return None
    pred_text = " ".join(pred_parts).strip()
    # simple whitespace tokenization
    tokens = re.findall(r'\S+', pred_text)
    return len(tokens)

def split_blocks(lines):
    """Yield blocks split by a line of dashes (or any line of >=4 dashes)."""
    current = []
    for ln in lines:
        if ln.strip() and all(ch == '-' for ch in ln.strip()) and len(ln.strip()) >= 4:
            if current:
                yield current
                current = []
        else:
            # keep original line content
            current.append(ln.rstrip('\n'))
    if current:
        yield current

def classify_decoding(text):
    """Return 'greedy', 'beam', or None based on text."""
    if not text:
        return None
    lower = text.lower()
    # prefer explicit tags: if 'beam' appears anywhere, treat as beam
    if 'greedy' in lower:
        return 'greedy'
    if 'beam' in lower:
        return 'beam'
    return None

def parse_block(block_lines):
    """Extract bleu (float), seq_len (int), and decoding strategy text from a block."""
    bleu = None
    seq_len = None
    dec_text = None

    for ln in block_lines:
        if bleu is None:
            m = RE_BLEU.search(ln)
            if m:
                try:
                    bleu = float(m.group(1))
                except Exception:
                    bleu = None
        if seq_len is None:
            try:
                tok_count = count_prediction_tokens(block_lines)
                if tok_count is not None:
                    seq_len = int(tok_count)
            except Exception:
                seq_len = None
        if dec_text is None:
            m = RE_DEC.search(ln)
            if m:
                dec_text = m.group(1).strip()

    # fallback: try to find any standalone number lines if missing
    if bleu is None or seq_len is None:
        all_nums = []
        for ln in block_lines:
            all_nums.extend(re.findall(r'([0-9]+(?:\.[0-9]+)?)', ln))
        if bleu is None and all_nums:
            # prefer a float-like number for BLEU
            dot_nums = [n for n in all_nums if '.' in n]
            candidate = dot_nums[0] if dot_nums else all_nums[0]
            try:
                bleu = float(candidate)
            except Exception:
                bleu = None
        if seq_len is None and all_nums:
            int_candidates = [n for n in all_nums if '.' not in n]
            candidate = int_candidates[0] if int_candidates else (all_nums[1] if len(all_nums) > 1 else all_nums[0])
            try:
                seq_len = int(float(candidate))
            except Exception:
                seq_len = None

    method = classify_decoding(dec_text)
    return method, bleu, seq_len

def safe_mean(lst):
    return mean(lst) if lst else None

def print_stats(name, bleus, lens):
    a_bleu = safe_mean(bleus)
    a_len = safe_mean(lens)
    count_bleu = len(bleus)
    count_len = len(lens)
    if a_bleu is not None:
        bstr = f"avg_BLEU={a_bleu:.4f}"
    else:
        bstr = "avg_BLEU=N/A"
    if a_len is not None:
        lstr = f"avg_len={a_len:.2f}"
    else:
        lstr = "avg_len=N/A"
    print(f"{name}  count={count_bleu}/{count_len}  {bstr}  {lstr}")

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"Input not found: {INPUT_PATH}", file=sys.stderr)
        sys.exit(2)

    greedy_bleus = []
    greedy_lens = []
    beam_bleus = []
    beam_lens = []

    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        blocks = list(split_blocks(f))
    for block in blocks:
        method, bleu, seq_len = parse_block(block)
        if method == 'greedy':
            if bleu is not None:
                greedy_bleus.append(bleu)
            if seq_len is not None:
                greedy_lens.append(seq_len)
        elif method == 'beam':
            if bleu is not None:
                beam_bleus.append(bleu)
            if seq_len is not None:
                beam_lens.append(seq_len)
        else:
            # unknown method: try to guess by scanning the block text
            joined = "\n".join(block).lower()
            if 'greedy' in joined and 'beam' not in joined:
                if bleu is not None:
                    greedy_bleus.append(bleu)
                if seq_len is not None:
                    greedy_lens.append(seq_len)
            elif 'beam' in joined:
                if bleu is not None:
                    beam_bleus.append(bleu)
                if seq_len is not None:
                    beam_lens.append(seq_len)
            else:
                # skip ambiguous
                continue

    print_stats("Greedy", greedy_bleus, greedy_lens)
    print_stats("Beam", beam_bleus, beam_lens)

if __name__ == "__main__":
    main()