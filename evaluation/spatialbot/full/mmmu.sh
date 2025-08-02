#!/bin/bash

SPLIT="test"
MODEL_TYPE=Qwen2.5-VL-7B-SIF-50K-SFT-GRPO-SIF
TARGET_DIR=bunny-Qwen2.5-VL-7B-SIF-50K-SFT-GRPO-SIF

python -m bunny.eval.model_vqa_mmmu \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-type $MODEL_TYPE \
    --data-path ./eval/mmmu/MMMU \
    --config-path ./eval/mmmu/config.yaml \
    --output-path ./eval/mmmu/answers_upload/$SPLIT/$TARGET_DIR.json \
    --split $SPLIT \
    --conv-mode bunny
