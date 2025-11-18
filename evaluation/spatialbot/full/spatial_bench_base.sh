#!/bin/bash
# positional, existence, counting, size
python -m bunny.eval.eval_spatialbench_base \
    --model-path Qwen2.5-VL-7B-SIF-50K-SFT-GRPO-SIF \
    --data-path SpatialBot/eval/spatial_bench/spatial_bench \
    --conv-mode bunny \
    --question positional.json \
    --depth
