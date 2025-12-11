#!/bin/bash

# Proto-Tokens Feasibility Verification Experiment
# Quick Run Script

echo "========================================"
echo "Proto-Tokens Feasibility Verification Experiment"
echo "========================================"
echo ""

# Set parameters
MODEL_PATH="Daniel0724/SimpleAR-1.5B-RL"
VQ_MODEL="nvidia/Cosmos-1.0-Tokenizer-DV8x16x16"
IMAGE_SIZE=256  # Use smaller size to speed up experiment
NUM_STEPS=1000
PROMPT="a red apple on a wooden table"
SAVE_DIR="./proto_tokens_exp"
USE_WANDB=true  # Set to true to enable WandB

echo "Configuration:"
echo "  Model: $MODEL_PATH"
echo "  Image Size: ${IMAGE_SIZE}x${IMAGE_SIZE}"
echo "  Optimization Steps: $NUM_STEPS"
echo "  Prompt: $PROMPT"
echo "  Save Directory: $SAVE_DIR"
echo "  WandB: $USE_WANDB"
echo ""

# Build command
CMD="python verify_proto_tokens.py \
    --model-path \"$MODEL_PATH\" \
    --vq-model-ckpt \"$VQ_MODEL\" \
    --prompt \"$PROMPT\" \
    --image-size $IMAGE_SIZE \
    --num-steps $NUM_STEPS \
    --save-dir \"$SAVE_DIR\" \
    --device \"cuda:0\""

# If WandB is enabled, add argument
if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use-wandb"
    echo "WandB enabled, experiment results will be uploaded to wandb.ai"
    echo ""
fi

# Run verification script
eval $CMD

echo ""
echo "========================================"
echo "Experiment Completed!"
echo "========================================"
echo "Results saved in: $SAVE_DIR"
echo ""
echo "View Results:"
echo "  1. Training Curves: $SAVE_DIR/training_curves.png"
echo "  2. Reference Image: $SAVE_DIR/reference_image.png"
echo "  3. Reconstructed Image: $SAVE_DIR/reconstructed_image.png"
echo "  4. Comparison Image: $SAVE_DIR/comparison.png"
echo "  5. Proto-tokens: $SAVE_DIR/proto_tokens.pt"

if [ "$USE_WANDB" = true ]; then
    echo ""
    echo "WandB Dashboard: https://wandb.ai/your-username/simplear-proto-tokens"
fi
