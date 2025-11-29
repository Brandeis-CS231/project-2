from pathlib import Path
import pandas as pd

from calc_bleu import split_blocks

def main():
    input = Path("exp_two_results/model5_0.001_32_50_50_256_8_256_4_4_0.1/bleu_details.txt")

    # dict with input as key, then greedy and beam outputs as values
    outputs = {}


    with open(input, "r", encoding="utf-8") as f:
        blocks = list(split_blocks(f))
        for block in blocks:
            strat = None
            pred = None
            for line in block:
                if line.startswith("Input: "):
                    inp = line[len("Input: "):].strip()
                    inp = inp.replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("<unk>", "").strip()
                    if outputs.get(inp) is None:
                        outputs[inp] = {}
                elif line.startswith("Reference: "):
                    ref = line[len("Reference: "):].strip()
                    outputs[inp]["reference"] = ref
                elif line.startswith("Prediction: "):
                    pred = line[len("Prediction: "):].strip()
                elif line.startswith("Decoding Strategy: "):
                    if "greedy" in line.lower():
                        strat = "greedy"
                    elif "beam" in line.lower():
                        strat = "beam"
            if strat and pred is not None:
                outputs[inp][strat] = pred
            else:
                print(f"Warning: could not classify block:\n{block}\n")

    # convert to dataframe
    df = pd.DataFrame.from_dict(outputs, orient="index")
    df.index.name = "input"

    # save to csv
    outpath = input.parent.parent / "collected_outputs.csv"
    df.to_csv(outpath, encoding="utf-8", index=True)
    print(f"Saved collected outputs to {outpath}")

if __name__ == "__main__":
    main()