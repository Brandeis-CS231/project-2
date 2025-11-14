# Project 2

This project uses uv and runs as a cli. To configure the
environment, run `uv sync` from the root.

## CLI Interface

```shell
Usage: project-2 [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  inference
  train
```

### Train Command

```shell
Usage: project-2 train [OPTIONS] TRAIN_FILE DEV_FILE TEST_FILE

Options:
  --epochs INTEGER
  --learning_rate FLOAT
  --batch_size INTEGER
  --max_src_len INTEGER
  --max_tgt_len INTEGER
  --d_model INTEGER
  --num_heads INTEGER
  --d_ff INTEGER
  --num_enc_layers INTEGER
  --num_dec_layers INTEGER
  --dropout FLOAT
  --strategy [greedy|beam_search]
  --beam_width INTEGER
  --save                          Enable verbose output.
  --help                          Show this message and exit.
```

### Inference Command

```shell
Usage: project-2 inference [OPTIONS]

Options:
  --checkpoint_path TEXT
  --strategy [greedy|beam_search]
  --beam_width INTEGER
  --help                          Show this message and exit.
```
