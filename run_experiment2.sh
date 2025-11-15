#!/bin/bash
# Experiment 2: Decoding Algorithms
# Train one strong model, then compare greedy vs beam search decoding

echo "=================================================="
echo "EXPERIMENT 2: Decoding Algorithms"
echo "=================================================="
echo ""

# Hyperparameters
EPOCHS=20
LR=0.001
BATCH_SIZE=64
D_MODEL=128
NUM_HEADS=4
D_FF=512
NUM_ENC_LAYERS=2
NUM_DEC_LAYERS=2
DROPOUT=0.1
POS_ENCODING="sinusoidal"
SEED=42

DATA_DIR="data"
TRAIN_FILE="${DATA_DIR}/train.json"
DEV_FILE="${DATA_DIR}/dev.json"
TEST_FILE="${DATA_DIR}/test.json"
MODEL_PATH="exp2_model.pt"

echo "Hyperparameters:"
echo "  Epochs: ${EPOCHS}"
echo "  Learning Rate: ${LR}"
echo "  Batch Size: ${BATCH_SIZE}"
echo "  d_model: ${D_MODEL}, num_heads: ${NUM_HEADS}"
echo "  Encoder/Decoder Layers: ${NUM_ENC_LAYERS}/${NUM_DEC_LAYERS}"
echo "  Positional Encoding: ${POS_ENCODING}"
echo "  Seed: ${SEED}"
echo ""

# Step 1: Train model to convergence
echo "=================================================="
echo "Step 1: Training Model to Convergence"
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
    --pos-encoding ${POS_ENCODING} \
    --model-save-path ${MODEL_PATH} \
    --seed ${SEED}

echo ""
echo "Model training complete!"
echo ""

# Step 2: Evaluate with different decoding strategies
echo "=================================================="
echo "Step 2: Evaluating Decoding Strategies"
echo "=================================================="
echo ""

uv run python evaluate_decoding.py \
    ${MODEL_PATH} \
    ${TEST_FILE} \
    ${TRAIN_FILE} \
    --batch-size ${BATCH_SIZE} \
    --d-model ${D_MODEL} \
    --num-heads ${NUM_HEADS} \
    --d-ff ${D_FF} \
    --num-enc-layers ${NUM_ENC_LAYERS} \
    --num-dec-layers ${NUM_DEC_LAYERS} \
    --dropout ${DROPOUT} \
    --pos-encoding ${POS_ENCODING}

echo ""
echo "=================================================="
echo "EXPERIMENT 2 COMPLETE"
echo "=================================================="
echo "Model saved to: ${MODEL_PATH}"
echo ""
echo "Results show BLEU scores and timing for:"
echo "  - Greedy decoding"
echo "  - Beam search (width=3)"
echo "  - Beam search (width=5)"
echo "  - Beam search (width=10)"
