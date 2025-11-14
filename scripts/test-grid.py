import subprocess


def test_grid() -> None:
    batch_size = 32
    num_enc_dec_layers = 1
    model_dimensions = 128
    num_heads = 2
    d_ff = 256

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
    test_grid()
