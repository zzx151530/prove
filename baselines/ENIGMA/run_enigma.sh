#!/bin/bash
###############################################################################
# ENIGMA 实验流程
# 使用 ENIGMA-enhanced E 2.1 直接证明 280 个测试问题
# 10轮反馈循环: 运行E→提取训练数据→训练LIBLINEAR→ENIGMA增强运行→重复
###############################################################################
set -o pipefail

BASE="/home/he3090/users/zhaozhongxu/prover"
ENIGMA_E="$BASE/enigma-eprover/PROVER/eprover"
ENIGMA_FEATURES="$BASE/enigma-eprover/SIMPLE_APPS/enigma-features"
CHAINY="$BASE/MPTP2078-master/chainy"
WORK="$BASE/ENIGMA"
TIME_LIMIT=60    # 60秒/题 (与级联评估总时间一致)
ITERATIONS=10

mkdir -p "$WORK/logs" "$WORK/models" "$WORK/results"

ts() { date "+%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$WORK/enigma.log"; }

# 加载测试问题列表
TEST_PROBLEMS=$(python3 -c "import json; print(' '.join(json.load(open('$BASE/LAH_TWGNN/test_problems_provable.json'))))")
TRAIN_PROBLEMS=$(python3 -c "
import json, os, glob
test = set(json.load(open('$BASE/LAH_TWGNN/test_problems_provable.json')))
all_probs = set()
for f in glob.glob('$CHAINY/*__*.p'):
    name = os.path.basename(f).split('__')[1].replace('.p','')
    all_probs.add(name)
train = all_probs - test
print(' '.join(sorted(train)[:1000]))
")

log "============================================================"
log "ENIGMA 实验 (E 2.1 + LIBLINEAR)"
log "测试问题: $(echo $TEST_PROBLEMS | wc -w)"
log "时间限制: ${TIME_LIMIT}s/题"
log "============================================================"

# ================================================================
# 函数: 运行E-prover并统计结果
# ================================================================
run_eprover() {
    local eprover_bin="$1"
    local extra_args="$2"
    local output_dir="$3"
    local label="$4"
    local limit="${5:-$TIME_LIMIT}"

    mkdir -p "$output_dir"
    local proved=0
    local total=0

    for prob in $TEST_PROBLEMS; do
        total=$((total + 1))
        # 找对应的chainy文件
        local pfile=$(find "$CHAINY" -name "*__${prob}.p" | head -1)
        [ -z "$pfile" ] && continue

        local out="$output_dir/${prob}.out"
        if [ -f "$out" ]; then
            grep -q "Theorem" "$out" && proved=$((proved + 1))
            continue
        fi

        timeout $((limit + 5)) "$eprover_bin" \
            --auto-schedule --free-numbers -s -R -p \
            --cpu-limit=$limit --memory-limit=4000 \
            --tstp-format $extra_args \
            "$pfile" > "$out" 2>/dev/null

        grep -q "Theorem" "$out" && proved=$((proved + 1))
    done

    log "  $label: $proved/$total"
    echo $proved
}

# ================================================================
# 函数: 提取训练数据
# ================================================================
extract_training_data() {
    local round=$1
    local train_dir="$WORK/train_data/round_$round"
    mkdir -p "$train_dir"

    log "  提取训练数据 (轮 $round)..."
    local count=0

    # 从训练问题的证明中提取正/负子句
    for prob in $TRAIN_PROBLEMS; do
        local pfile=$(find "$CHAINY" -name "*__${prob}.p" | head -1)
        [ -z "$pfile" ] && continue

        local pos_file="$train_dir/${prob}.pos"
        local neg_file="$train_dir/${prob}.neg"
        [ -f "$pos_file" ] && { count=$((count + 1)); continue; }

        # 运行E-prover并导出训练样本
        timeout 15 "$ENIGMA_E" \
            --training-examples=3 \
            --free-numbers -s -R \
            --cpu-limit=10 --memory-limit=4000 \
            --tstp-format \
            ${2:+$2} \
            "$pfile" > /dev/null 2>/dev/null

        # training-examples输出到当前目录
        [ -f "*.pos" ] && mv *.pos "$pos_file" 2>/dev/null
        [ -f "*.neg" ] && mv *.neg "$neg_file" 2>/dev/null
        count=$((count + 1))

        [ $((count % 100)) -eq 0 ] && log "    $count 个问题已处理"
    done
    log "  训练数据: $count 个问题"
}

# ================================================================
# Round 0: Vanilla E 2.1 (无ENIGMA增强)
# ================================================================
log ""
log "=== Round 0: Vanilla E 2.1 ==="
ROUND0_DIR="$WORK/results/round_0"
proved_0=$(run_eprover "$ENIGMA_E" "" "$ROUND0_DIR" "Vanilla E 2.1")

# 记录结果
RESULTS="$WORK/loop_results.json"
python3 -c "import json; json.dump({'round_0': $proved_0}, open('$RESULTS','w'), indent=2)"

# 累计proved集合
ALL_PROVED="$WORK/all_proved.json"
python3 -c "
import os, json
proved = set()
rd = '$ROUND0_DIR'
for f in os.listdir(rd):
    if f.endswith('.out'):
        with open(os.path.join(rd, f)) as fh:
            if 'Theorem' in fh.read():
                proved.add(f.replace('.out',''))
json.dump(sorted(proved), open('$ALL_PROVED','w'))
print(f'Round 0 累计: {len(proved)}')
" | tee -a "$WORK/enigma.log"

log ""
log "============================================================"
log "ENIGMA Round 0 完成: $proved_0/280"
log "(注: ENIGMA训练+反馈循环需要enigmatic框架, 待实现)"
log "============================================================"

# 汇总
log ""
log "=== 结果 ==="
cat "$RESULTS"
