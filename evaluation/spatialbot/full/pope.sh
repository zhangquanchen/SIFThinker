#!/bin/bash

MODEL_TYPE=Qwen2.5-VL-7B-SIF-50K-SFT-GRPO-SIF
TARGET_DIR=bunny-Qwen2.5-VL-7B-SIF-50K-SFT-GRPO-SIF

python -m bunny.eval.model_vqa_loader \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-type $MODEL_TYPE \
    --question-file ./eval/pope/bunny_pope_test.jsonl \
    --image-folder SpatialBot/eval/val2014 \
    --answers-file ./eval/pope/answers/$TARGET_DIR.jsonl \
    --temperature 0 \
    --conv-mode bunny

python eval/pope/eval_pope.py \
    --annotation-dir ./eval/pope/coco \
    --question-file ./eval/pope/bunny_pope_test.jsonl \
    --result-file ./eval/pope/answers/$TARGET_DIR.jsonl
