#!/bin/bash
# Experiment 3: Model Architecture Variants
# Compare different architectural choices: attention heads and model depth

echo "=================================================="
echo "EXPERIMENT 3: Architecture Variants"
echo "=================================================="
echo ""
echo "This experiment will test:"
echo "  Dimension 1: Number of attention heads (2, 4, 8)"
echo "  Dimension 2: Model depth (1, 2, 4 layers)"
echo ""

# Shared hyperparameters
EPOCHS=15
LR=0.001
BATCH_SIZE=64
D_MODEL=128
D_FF=512
DROPOUT=0.1
POS_ENCODING="sinusoidal"
SEED=42

DATA_DIR="data"
TRAIN_FILE="${DATA_DIR}/train.json"
DEV_FILE="${DATA_DIR}/dev.json"
TEST_FILE="${DATA_DIR}/test.json"

echo "Shared Hyperparameters:"
echo "  Epochs: ${EPOCHS}"
echo "  Learning Rate: ${LR}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  d_model: ${D_MODEL}, d_ff: ${D_FF}"
echo "  Dropout: ${DROPOUT}"
echo "  Seed: ${SEED}"
echo ""

# ==================== Dimension 1: Number of Attention Heads ====================
echo "=================================================="
echo "DIMENSION 1: Number of Attention Heads"
echo "=================================================="
echo ""

# Fixed depth for this dimension
FIXED_LAYERS=2

# Experiment with 2, 4, 8 heads
for NUM_HEADS in 2 4 8; do
    echo "--------------------------------------------------"
    echo "Training: ${NUM_HEADS} attention heads, ${FIXED_LAYERS} layers"
    echo "--------------------------------------------------"
    echo ""

    MODEL_PATH="exp3_heads${NUM_HEADS}_layers${FIXED_LAYERS}.pt"

    uv run python main.py \
        ${TRAIN_FILE} ${DEV_FILE} ${TEST_FILE} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --batch-size ${BATCH_SIZE} \
        --d-model ${D_MODEL} \
        --num-heads ${NUM_HEADS} \
        --d-ff ${D_FF} \
        --num-enc-layers ${FIXED_LAYERS} \
        --num-dec-layers ${FIXED_LAYERS} \
        --dropout ${DROPOUT} \
        --pos-encoding ${POS_ENCODING} \
        --model-save-path ${MODEL_PATH} \
        --seed ${SEED}

    echo ""
    echo "Model saved to: ${MODEL_PATH}"
    echo ""
done

# ==================== Dimension 2: Model Depth ====================
echo "=================================================="
echo "DIMENSION 2: Model Depth (Encoder/Decoder Layers)"
echo "=================================================="
echo ""

# Fixed heads for this dimension
FIXED_HEADS=4

# Experiment with 1, 2, 4 layers
for NUM_LAYERS in 1 2 4; do
    # Skip 2 layers since we already ran it above
    if [ ${NUM_LAYERS} -eq 2 ]; then
        echo "Skipping ${NUM_LAYERS} layers (already trained with ${FIXED_HEADS} heads)"
        echo ""
        continue
    fi

    echo "--------------------------------------------------"
    echo "Training: ${FIXED_HEADS} attention heads, ${NUM_LAYERS} layers"
    echo "--------------------------------------------------"
    echo ""

    MODEL_PATH="exp3_heads${FIXED_HEADS}_layers${NUM_LAYERS}.pt"

    uv run python main.py \
        ${TRAIN_FILE} ${DEV_FILE} ${TEST_FILE} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --batch-size ${BATCH_SIZE} \
        --d-model ${D_MODEL} \
        --num-heads ${FIXED_HEADS} \
        --d-ff ${D_FF} \
        --num-enc-layers ${NUM_LAYERS} \
        --num-dec-layers ${NUM_LAYERS} \
        --dropout ${DROPOUT} \
        --pos-encoding ${POS_ENCODING} \
        --model-save-path ${MODEL_PATH} \
        --seed ${SEED}

    echo ""
    echo "Model saved to: ${MODEL_PATH}"
    echo ""
done

echo "=================================================="
echo "EXPERIMENT 3 COMPLETE"
echo "=================================================="
echo ""
echo "Models trained:"
echo ""
echo "Attention Heads Experiments (2 layers):"
echo "  - exp3_heads2_layers2.pt  (2 heads)"
echo "  - exp3_heads4_layers2.pt  (4 heads)"
echo "  - exp3_heads8_layers2.pt  (8 heads)"
echo ""
echo "Model Depth Experiments (4 heads):"
echo "  - exp3_heads4_layers1.pt  (1 layer)"
echo "  - exp3_heads4_layers2.pt  (2 layers) [shared with above]"
echo "  - exp3_heads4_layers4.pt  (4 layers)"
echo ""
echo "Compare BLEU scores and parameter counts from training logs."
