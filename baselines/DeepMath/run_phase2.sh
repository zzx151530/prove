#!/bin/bash
###############################################################################
# DeepMath Phase 2: 10轮反馈循环
# 等待Phase 1完成后自动启动
###############################################################################
BASE="/home/he3090/users/zhaozhongxu/prover"
export PROVER_BASE="$BASE"
source activate pytorch 2>/dev/null || conda activate pytorch 2>/dev/null

# 等Phase 1完成
while [ ! -f "$BASE/DeepMath/eval_cascade_deepmath/cascade_metrics.json" ]; do
    echo "[$(date)] 等待Phase 1完成..."
    sleep 60
done

echo "[$(date)] Phase 1完成:"
cat "$BASE/DeepMath/eval_cascade_deepmath/cascade_metrics.json"
echo ""

# Phase 2: 反馈循环, 用Phase 1的模型作为init_model
echo "[$(date)] === DeepMath Phase 2: 10轮反馈循环 ==="
cd "$BASE"
CUDA_VISIBLE_DEVICES=0 python3 feedback_loop.py \
    --model_type DEEPMATH --iterations 10 \
    --dim 1024 --layers 1 \
    --lr 0.001 --batch_size 64 --epochs 50 \
    --device cuda:0 --n_shards 6 \
    --init_model "$BASE/DeepMath/models/deepmath_D0/best.pt" \
    --data_dir "$BASE/DeepMath/feedback_loop" \
    --logfile "$BASE/DeepMath/feedback_loop.log"

echo "[$(date)] === DeepMath Phase 2完成 ==="
cat "$BASE/DeepMath/feedback_loop/loop_results.json"
