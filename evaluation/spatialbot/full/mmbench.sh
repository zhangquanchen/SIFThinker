#!/bin/bash

SPLIT="MMBench_TEST_EN_legacy"
LANG=en
MODEL_TYPE=Qwen2.5-VL-7B-SIF-50K-SFT-GRPO-SIF
TARGET_DIR=bunny-Qwen2.5-VL-7B-SIF-50K-SFT-GRPO-SIF


python -m bunny.eval.model_vqa_mmbench \
    --model-path ./checkpoints-$MODEL_TYPE/$TARGET_DIR \
    --model-type $MODEL_TYPE \
    --question-file ./eval/mmbench/$SPLIT.tsv \
    --answers-file ./eval/mmbench/answers/$SPLIT/$TARGET_DIR.jsonl \
    --lang $LANG \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode bunny

mkdir -p eval/mmbench/answers_upload/$SPLIT

python eval/mmbench/convert_mmbench_for_submission.py \
    --annotation-file ./eval/mmbench/$SPLIT.tsv \
    --result-dir ./eval/mmbench/answers/$SPLIT \
    --upload-dir ./eval/mmbench/answers_upload/$SPLIT \
    --experiment $TARGET_DIR
