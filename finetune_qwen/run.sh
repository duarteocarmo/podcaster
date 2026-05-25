#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

EXPERIMENT_NAME="qwen_tts_finetune"
SPEAKER_NAME="duarte"

DATASET_DIR="dataset"
TRAIN_DIR="$DATASET_DIR/samples"
REF_DIR="$DATASET_DIR/reference"

NUM_SAMPLES=200
SEED=42
BATCH_SIZE=4
LR=2e-6
NUM_EPOCHS=3
GRAD_ACCUM=4

INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-0.6B-Base"

: "${HF_TOKEN:?Set HF_TOKEN to sync the dataset from Hugging Face}"
uvx --from "huggingface-hub[cli]" hf sync --ignore-times hf://buckets/duarteocarmo/voice ./dataset

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python train.py \
    --experiment_name "$EXPERIMENT_NAME" \
    --speaker_name    "$SPEAKER_NAME" \
    --train_dir       "$TRAIN_DIR" \
    --ref_dir         "$REF_DIR" \
    --num_samples     "$NUM_SAMPLES" \
    --seed            "$SEED" \
    --batch_size      "$BATCH_SIZE" \
    --lr              "$LR" \
    --num_epochs      "$NUM_EPOCHS" \
    --grad_accum      "$GRAD_ACCUM" \
    --init_model_path "$INIT_MODEL_PATH"
