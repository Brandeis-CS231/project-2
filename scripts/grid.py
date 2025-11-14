import subprocess
from itertools import product


def grid_search() -> None:
    batch_size = [32, 64, 128]
    num_enc_dec_layers = [1, 2, 4]
    model_dimensions = [128, 256]
    num_heads = [2, 4, 8]
    d_ff = [256, 512]

    grid = product(batch_size, num_enc_dec_layers, model_dimensions, num_heads, d_ff)

    for combination in grid:
        batch_size, num_enc_dec_layers, model_dimensions, num_heads, d_ff = combination
        subprocess.run(
            [
                "sbatch",
                "scripts/grid.sh",
                str(batch_size),
                str(num_enc_dec_layers),
                str(model_dimensions),
                str(num_heads),
                str(d_ff),
            ]
        )


if __name__ == "__main__":
    grid_search()
