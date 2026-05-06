#!/usr/bin/env python3
"""
E-SInE基准评估脚本

使用E-prover内置的SInE前提选择方法（--sine=Auto）运行60秒评估。
带-p标志提取证明使用的公理，用于计算Spre/Sel。

使用方式:
  # 单分片运行
  python3 evaluate_sine.py run --split 1/8 --output_dir results_sine

  # 一键后台启动所有分片
  python3 evaluate_sine.py run --split all/8 --output_dir results_sine

  # 合并结果
  python3 evaluate_sine.py merge --output_dir results_sine
"""

import os
import sys
import re
import json
import time
import subprocess
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_simple import parse_problem_file, extract_used_axioms
from run import find_problem_file, split_list, parse_split


def run_eprover_sine(problem_file, eprover_path, time_limit=60,
                     memory_limit=10000, strategy='satauto'):
    """
    运行E-prover（带SInE前提选择和-p证明输出）
    """
    cmd = [eprover_path]

    if strategy == 'auto':
        cmd.append('--auto-schedule')
    elif strategy == 'satauto':
        cmd.append('--satauto-schedule')

    cmd.extend([
        '--sine=Auto',
        '--free-numbers',
        '-s', '-R', '-p',
        f'--cpu-limit={time_limit}',
        f'--memory-limit={memory_limit}',
        '--tstp-format',
        problem_file
    ])

    start = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=time_limit + 30
        )
        elapsed = time.time() - start
        success = '# Proof found!' in result.stdout or '% Proof found!' in result.stdout

        m = re.search(r'[#%] Total time\s*:\s*([\d.]+)', result.stdout)
        cpu_time = float(m.group(1)) if m else elapsed

        return success, result.stdout, cpu_time

    except subprocess.TimeoutExpired:
        return False, 'Timeout', time.time() - start
    except Exception as e:
        return False, f'Error: {e}', 0.0


def cmd_run(args):
    """运行E-SInE评估"""
    split_id, total_splits = parse_split(args.split)

    if split_id == 'all':
        launch_all_splits(args, total_splits)
        return

    if not os.path.exists(args.eprover_path):
        print(f"错误: E-prover不存在: {args.eprover_path}")
        sys.exit(1)

    # 加载测试问题列表
    with open(args.test_problems, 'r') as f:
        test_problems = json.load(f)
        if isinstance(test_problems, dict):
            test_problems = sorted(test_problems.keys())

    all_problems = sorted(test_problems)
    my_problems = split_list(all_problems, split_id, total_splits)

    output_file = os.path.join(args.output_dir, f'sine_split_{split_id}of{total_splits}.json')

    print("=" * 60)
    print(f"E-SInE评估 - 分片 {split_id}/{total_splits}")
    print("=" * 60)
    print(f"总问题数: {len(all_problems)} | 本分片: {len(my_problems)}")
    print(f"时间限制: {args.time_limit}s | 策略: {args.strategy}")
    print("=" * 60)

    # 断点续传
    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    done = set(results.keys())
    print(f"已完成: {len(done)}")

    proved = sum(1 for v in results.values() if v.get('proved'))
    count = len(done)

    for prob_name in my_problems:
        if prob_name in done:
            continue

        prob_file = find_problem_file(prob_name, args.chainy_dir)
        if not prob_file:
            count += 1
            continue

        # 解析获取公理总数
        conj_name, conj_str, local_axioms = parse_problem_file(prob_file)
        num_total_axioms = len(local_axioms) if local_axioms else 0

        # 运行E-SInE
        success, output, proof_time = run_eprover_sine(
            prob_file, args.eprover_path, args.time_limit,
            args.memory_limit, args.strategy
        )

        used_axioms = extract_used_axioms(output) if success else []

        results[prob_name] = {
            'proved': success,
            'proof_time': round(proof_time, 2),
            'used_axioms': used_axioms,
            'num_used': len(used_axioms),
            'num_total_axioms': num_total_axioms
        }

        count += 1
        if success:
            proved += 1

        print(f"\r[{count}/{len(my_problems)}] 已证明: {proved} | "
              f"{prob_name} {'OK' if success else 'FAIL'} ({proof_time:.1f}s)     ",
              end='', flush=True)

        if count % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    print()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n分片 {split_id}/{total_splits} 完成!")
    print(f"处理: {len(results)} | 已证明: {proved}")


def cmd_merge(args):
    """合并分片结果并计算指标"""
    print("=" * 60)
    print("合并E-SInE结果")
    print("=" * 60)

    split_files = sorted([
        f for f in os.listdir(args.output_dir)
        if re.match(r'sine_split_\d+of\d+\.json', f)
    ])

    if not split_files:
        print("未找到分片文件（sine_split_*of*.json）")
        return

    print(f"找到 {len(split_files)} 个分片文件:")

    merged = {}
    for filename in split_files:
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        merged.update(data)
        print(f"  {filename}: {len(data)} 个问题")

    print(f"\n合并后总计: {len(merged)} 个问题")

    # 计算指标
    proved_problems = {k: v for k, v in merged.items() if v.get('proved')}
    proved_count = len(proved_problems)
    total_count = len(merged)

    # SInE: NSel = NAxP (SInE内部选择，我们把输入总数作为NSel)
    spre_sum = 0.0
    sel_sum = 0.0
    total_proof_time = 0.0

    for v in proved_problems.values():
        n_used = v.get('num_used', 0)
        n_axp = v.get('num_total_axioms', 1)

        # SInE: NSel = NAxP
        if n_axp > 0:
            spre_sum += n_used / n_axp
            sel_sum += 1.0  # NSel/NAxP = 1.0

        total_proof_time += v.get('proof_time', 0.0)

    spre = spre_sum / proved_count if proved_count > 0 else 0.0
    sel = sel_sum / proved_count if proved_count > 0 else 0.0
    avg_proof_time = total_proof_time / proved_count if proved_count > 0 else 0.0

    metrics = {
        'total_problems': total_count,
        'proved_count': proved_count,
        'proved_rate': round(proved_count / max(total_count, 1) * 100, 2),
        'avg_proof_time': round(avg_proof_time, 2),
        'Spre': round(spre, 4),
        'Sel': round(sel, 4)
    }

    merged_file = os.path.join(args.output_dir, 'sine_merged.json')
    with open(merged_file, 'w') as f:
        json.dump(merged, f, indent=2)

    metrics_file = os.path.join(args.output_dir, 'sine_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"E-SInE评估结果")
    print(f"{'='*60}")
    print(f"总问题数: {total_count}")
    print(f"已证明:   {proved_count} ({metrics['proved_rate']:.1f}%)")
    print(f"平均证明时间: {avg_proof_time:.2f}s")
    print(f"Spre: {spre:.4f}")
    print(f"Sel:  {sel:.4f}")
    print(f"\n合并结果: {merged_file}")
    print(f"指标: {metrics_file}")


def launch_all_splits(args, total_splits):
    """后台启动所有分片"""
    print("=" * 60)
    print(f"一键启动 {total_splits} 个分片")
    print("=" * 60)

    script_path = os.path.abspath(__file__)
    os.makedirs(args.output_dir, exist_ok=True)

    for i in range(1, total_splits + 1):
        cmd = [sys.executable, script_path, 'run', '--split', f'{i}/{total_splits}']

        for key, value in vars(args).items():
            if key in ('command', 'split'):
                continue
            if value is None:
                continue
            if isinstance(value, bool):
                if value:
                    cmd.append(f'--{key}')
                continue
            if isinstance(value, list):
                cmd.append(f'--{key}')
                cmd.extend(str(v) for v in value)
                continue
            cmd.extend([f'--{key}', str(value)])

        log_file = os.path.join(args.output_dir, f'log_split_{i}of{total_splits}.txt')
        print(f"  启动分片 {i}/{total_splits} -> {log_file}")

        with open(log_file, 'w') as log:
            subprocess.Popen(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True
            )

    print(f"\n全部 {total_splits} 个分片已在后台启动!")
    print(f"查看进度: tail -f {args.output_dir}/log_split_*")
    print(f"全部完成后运行: python3 {script_path} merge --output_dir {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='E-SInE基准评估',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    # === run ===
    p1 = subparsers.add_parser('run', help='运行E-SInE评估')
    p1.add_argument('--split', type=str, default='1/1')
    p1.add_argument('--chainy_dir', type=str,
                    default='/home/stu1/zhaozhongxu/prove/MPTP2078-master/chainy')
    p1.add_argument('--test_problems', type=str,
                    default='/home/stu1/zhaozhongxu/prove/TWGNN/data/mptp2078_custom/test_problems.json')
    p1.add_argument('--eprover_path', type=str,
                    default='/home/stu1/zhaozhongxu/prove/eprover/PROVER/eprover')
    p1.add_argument('--output_dir', type=str, required=True)
    p1.add_argument('--time_limit', type=int, default=60)
    p1.add_argument('--memory_limit', type=int, default=10000)
    p1.add_argument('--strategy', type=str, default='satauto',
                    choices=['auto', 'satauto', 'none'])

    # === merge ===
    p2 = subparsers.add_parser('merge', help='合并分片结果')
    p2.add_argument('--output_dir', type=str, required=True)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == 'run':
        os.makedirs(args.output_dir, exist_ok=True)

    start = datetime.now()
    print(f"开始: {start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.command == 'run':
        cmd_run(args)
    elif args.command == 'merge':
        cmd_merge(args)

    print(f"\n总用时: {datetime.now() - start}")


if __name__ == '__main__':
    main()
