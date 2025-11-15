#!/bin/bash
# Experiment 1: Positional Encoding Strategies
# Compare sinusoidal vs learnable positional encodings

echo "=================================================="
echo "EXPERIMENT 1: Positional Encoding Strategies"
echo "=================================================="
echo ""

# Shared hyperparameters
EPOCHS=15
LR=0.001
BATCH_SIZE=64
D_MODEL=128
NUM_HEADS=4
D_FF=512
NUM_ENC_LAYERS=2
NUM_DEC_LAYERS=2
DROPOUT=0.1
SEED=42

DATA_DIR="data"
TRAIN_FILE="${DATA_DIR}/train.json"
DEV_FILE="${DATA_DIR}/dev.json"
TEST_FILE="${DATA_DIR}/test.json"

echo "Shared Hyperparameters:"
echo "  Epochs: ${EPOCHS}"
echo "  Learning Rate: ${LR}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  d_model: ${D_MODEL}, num_heads: ${NUM_HEADS}"
echo "  Encoder/Decoder Layers: ${NUM_ENC_LAYERS}/${NUM_DEC_LAYERS}"
echo "  Seed: ${SEED}"
echo ""

# Run 1: Sinusoidal Positional Encoding (Baseline)
echo "=================================================="
echo "Training Model 1: Sinusoidal Positional Encoding"
echo "=================================================="
echo ""

uv run python main.py \
    ${TRAIN_FILE} ${DEV_FILE} ${TEST_FILE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --d-model ${D_MODEL} \
    --num-heads ${NUM_HEADS} \
    --d-ff ${D_FF} \
    --num-enc-layers ${NUM_ENC_LAYERS} \
    --num-dec-layers ${NUM_DEC_LAYERS} \
    --dropout ${DROPOUT} \
    --pos-encoding sinusoidal \
    --model-save-path exp1_sinusoidal.pt \
    --seed ${SEED}

echo ""
echo "Model 1 training complete!"
echo ""

# Run 2: Learnable Positional Encoding
echo "=================================================="
echo "Training Model 2: Learnable Positional Encoding"
echo "=================================================="
echo ""

uv run python main.py \
    ${TRAIN_FILE} ${DEV_FILE} ${TEST_FILE} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --d-model ${D_MODEL} \
    --num-heads ${NUM_HEADS} \
    --d-ff ${D_FF} \
    --num-enc-layers ${NUM_ENC_LAYERS} \
    --num-dec-layers ${NUM_DEC_LAYERS} \
    --dropout ${DROPOUT} \
    --pos-encoding learnable \
    --model-save-path exp1_learnable.pt \
    --seed ${SEED}

echo ""
echo "Model 2 training complete!"
echo ""

echo "=================================================="
echo "EXPERIMENT 1 COMPLETE"
echo "=================================================="
echo "Results saved to:"
echo "  - exp1_sinusoidal.pt"
echo "  - exp1_learnable.pt"
echo ""
echo "Compare the BLEU scores and loss curves from both runs."
