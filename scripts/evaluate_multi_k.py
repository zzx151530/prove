#!/usr/bin/env python3
"""
Multi-K 工作流评估
按顺序用不同 K 值尝试证明，每个 K 只处理尚未证出的题。

用法:
  python3 evaluate_multi_k.py \
    --premise_rankings ../models/model_D4_d512_l1/premise_rankings_1024.json \
    --test_problems ../data/mptp2078_custom/test_problems_provable.json \
    --output_dir ../eval_D4_d512_l1_multiK/ \
    --k_sequence 256,64,128,512,1024 \
    --time_limit 60
"""

import os
import sys
import json
import argparse
import subprocess
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate_fixed_k import evaluate_fixed_k_single
from evaluate_simple import parse_problem_file
from run import find_problem_file, split_list, parse_split


def main():
    parser = argparse.ArgumentParser(description='Multi-K 工作流评估')
    parser.add_argument('--premise_rankings', type=str, required=True)
    parser.add_argument('--test_problems', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--k_sequence', type=str, default='256,64,128,512,1024',
                       help='K 值序列，逗号分隔 [默认: 256,64,128,512,1024]')
    parser.add_argument('--time_limit', type=int, default=60)
    parser.add_argument('--eprover_path', type=str,
                       default='/home/stu1/zhaozhongxu/prove/eprover/PROVER/eprover')
    parser.add_argument('--chainy_dir', type=str,
                       default='/home/stu1/zhaozhongxu/prove/MPTP2078-master/chainy/')
    parser.add_argument('--strategy', type=str, default='satauto')
    parser.add_argument('--parallel', type=int, default=8, help='并行进程数')
    args = parser.parse_args()

    k_sequence = [int(k) for k in args.k_sequence.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    with open(args.premise_rankings) as f:
        rankings = json.load(f)

    with open(args.test_problems) as f:
        test_probs = json.load(f)
    if isinstance(test_probs, dict):
        prob_names = sorted(test_probs.keys())
    else:
        prob_names = sorted(test_probs)

    print("=" * 60)
    print(f"Multi-K 工作流评估")
    print(f"K 序列: {k_sequence}")
    print(f"测试题数: {len(prob_names)}")
    print(f"时间限制: {args.time_limit}s/K/题")
    print(f"并行: {args.parallel}")
    print("=" * 60)

    # Track results
    all_results = {}
    proved_set = set()
    step_stats = []

    for step_idx, K in enumerate(k_sequence):
        remaining = [p for p in prob_names if p not in proved_set]
        if not remaining:
            print(f"\n所有题已证出！")
            break

        print(f"\n--- Step {step_idx+1}: K={K}, 剩余 {len(remaining)} 题 ---")
        start = time.time()

        # Write remaining problems to temp file for parallel processing
        step_dir = os.path.join(args.output_dir, f"step_{step_idx+1}_K{K}")
        os.makedirs(step_dir, exist_ok=True)

        # Save remaining problems list
        remaining_file = os.path.join(step_dir, "remaining.json")
        with open(remaining_file, 'w') as f:
            json.dump({p: [] for p in remaining}, f)

        # Run parallel evaluation using subprocess
        processes = []
        n_splits = min(args.parallel, len(remaining))
        for i in range(1, n_splits + 1):
            cmd = [
                sys.executable, os.path.join(os.path.dirname(__file__), 'evaluate_fixed_k.py'),
                'run', '--split', f'{i}/{n_splits}', '--K', str(K),
                '--premise_rankings', args.premise_rankings,
                '--test_problems', remaining_file,
                '--output_dir', step_dir,
                '--time_limit', str(args.time_limit),
                '--strategy', args.strategy,
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            processes.append(p)

        # Wait for all
        for p in processes:
            p.wait()

        # Merge results
        merge_cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), 'evaluate_fixed_k.py'),
            'merge', '--output_dir', step_dir,
        ]
        subprocess.run(merge_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load merged results
        merged_file = os.path.join(step_dir, "fixedk_merged.json")
        if os.path.exists(merged_file):
            with open(merged_file) as f:
                step_results = json.load(f)

            new_proved = 0
            for prob, result in step_results.items():
                if result.get('proved'):
                    proved_set.add(prob)
                    new_proved += 1
                    result['proved_at_K'] = K
                    result['proved_at_step'] = step_idx + 1
                all_results[prob] = result

            elapsed = time.time() - start
            step_stats.append({
                'step': step_idx + 1,
                'K': K,
                'remaining': len(remaining),
                'new_proved': new_proved,
                'cumulative': len(proved_set),
                'time': round(elapsed, 1)
            })
            print(f"  新证明: {new_proved}, 累计: {len(proved_set)}/{len(prob_names)}, 耗时: {elapsed:.0f}s")
        else:
            print(f"  [WARNING] 合并失败")

    # Save final results
    final_file = os.path.join(args.output_dir, "multi_k_results.json")
    with open(final_file, 'w') as f:
        json.dump({
            'k_sequence': k_sequence,
            'total_problems': len(prob_names),
            'total_proved': len(proved_set),
            'step_stats': step_stats,
            'results': all_results,
        }, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Multi-K 最终结果: {len(proved_set)}/{len(prob_names)} ({len(proved_set)/len(prob_names)*100:.1f}%)")
    print(f"{'=' * 60}")
    for s in step_stats:
        print(f"  Step {s['step']}: K={s['K']:>4}, +{s['new_proved']:>3} 新证明, 累计 {s['cumulative']}, {s['time']:.0f}s")
    print(f"结果: {final_file}")


if __name__ == '__main__':
    main()
