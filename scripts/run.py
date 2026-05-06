#!/usr/bin/env python3
"""
分批并行E-prover测试脚本

核心思路: 将问题分成K份，在K个终端中同时运行，最后合并结果。

使用示例:

  # === 1. 筛选可证明问题（bushy），分4批运行 ===

  # 终端1:
  python3 run.py provable --split 1/4 --problem_dir .../bushy --output_dir .../results

  # 终端2:
  python3 run.py provable --split 2/4 --problem_dir .../bushy --output_dir .../results

  # 终端3:
  python3 run.py provable --split 3/4 --problem_dir .../bushy --output_dir .../results

  # 终端4:
  python3 run.py provable --split 4/4 --problem_dir .../bushy --output_dir .../results

  # 全部完成后合并:
  python3 run.py merge --output_dir .../results --train_deps .../train_deps

  # === 2. Baseline测试（chainy），分4批 ===
  python3 run.py baseline --split 1/4 --problem_dir .../chainy --output_dir .../results
  ...
  python3 run.py merge --output_dir .../results

  # === 3. 前提选择评估，分4批 ===
  python3 run.py evaluate --split 1/4 --chainy_dir .../chainy --output_dir .../results ...
  ...
  python3 run.py merge --output_dir .../results --mode evaluate --k_values 64 128 256 512

  # === 快捷方式: 一键启动所有分片（后台运行） ===
  python3 run.py provable --split all/4 --problem_dir .../bushy --output_dir .../results
"""

import os
import sys
import re
import json
import time
import subprocess
import argparse
from datetime import datetime


# ============================================================
# E-prover 运行
# ============================================================

def run_eprover(problem_file, eprover_path, time_limit=60, memory_limit=4000, strategy='none'):
    """运行E-prover证明单个问题"""
    cmd = [eprover_path]

    if strategy == 'auto':
        cmd.append('--auto-schedule')
    elif strategy == 'satauto':
        cmd.append('--satauto-schedule')

    cmd.extend([
        '--free-numbers',
        '-s', '-R',
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

        return success, cpu_time, result.stdout

    except subprocess.TimeoutExpired:
        return False, time.time() - start, 'Timeout'
    except FileNotFoundError:
        return False, 0.0, f'Error: eprover not found at {eprover_path}'
    except Exception as e:
        return False, 0.0, f'Error: {e}'


# ============================================================
# 工具函数
# ============================================================

def check_eprover(eprover_path):
    """验证E-prover路径是否有效"""
    if not os.path.exists(eprover_path):
        print(f"错误: E-prover不存在: {eprover_path}")
        print("请使用 --eprover_path 指定正确路径")
        sys.exit(1)
    if not os.access(eprover_path, os.X_OK):
        print(f"错误: E-prover无执行权限: {eprover_path}")
        sys.exit(1)
    print(f"E-prover: {eprover_path} [OK]")


def extract_problem_name(filename):
    """compts_1__t10_compts_1.p -> t10_compts_1"""
    name = filename[:-2] if filename.endswith('.p') else filename
    if '__' in name:
        return name.split('__', 1)[1]
    return name


def find_problem_file(problem_name, problem_dir):
    """根据问题名找到文件路径"""
    candidate = os.path.join(problem_dir, f"{problem_name}.p")
    if os.path.exists(candidate):
        return candidate

    m = re.match(r't\d+_(.+)', problem_name)
    if m:
        theory = m.group(1)
        candidate = os.path.join(problem_dir, f"{theory}__{problem_name}.p")
        if os.path.exists(candidate):
            return candidate
    return None


def parse_problem_file(file_path):
    """解析问题文件，提取猜想和公理"""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    conj_name = conj_str = None
    local_axioms = {}
    buffer = []

    for line in lines:
        if '%' in line:
            line = line.split('%')[0]
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        buffer.append(line)

        if line.endswith('.'):
            full_fof = " ".join(buffer)
            buffer = []
            if not full_fof.startswith('fof'):
                continue
            try:
                ns = full_fof.find('(') + 1
                ne = full_fof.find(',', ns)
                name = full_fof[ns:ne].strip()
                clean = full_fof.replace(' ', '')
                if ',conjecture,' in clean:
                    conj_name, conj_str = name, full_fof
                elif ',axiom,' in clean:
                    local_axioms[name] = full_fof
            except Exception:
                continue

    return conj_name, conj_str, local_axioms


def split_list(lst, split_id, total_splits):
    """将列表分成total_splits份，返回第split_id份（1-based）"""
    n = len(lst)
    chunk_size = n // total_splits
    remainder = n % total_splits

    # 前remainder份多一个元素
    start = 0
    for i in range(1, split_id):
        start += chunk_size + (1 if i <= remainder else 0)
    end = start + chunk_size + (1 if split_id <= remainder else 0)

    return lst[start:end]


def parse_split(split_str):
    """解析 '1/4' 或 'all/4' 格式"""
    parts = split_str.split('/')
    if len(parts) != 2:
        raise ValueError(f"分片格式错误: {split_str}，应为 '1/4' 或 'all/4'")
    total = int(parts[1])
    if parts[0] == 'all':
        return 'all', total
    split_id = int(parts[0])
    if split_id < 1 or split_id > total:
        raise ValueError(f"分片ID必须在1到{total}之间")
    return split_id, total


# ============================================================
# 任务1: provable - 筛选可证明问题
# ============================================================

def cmd_provable(args):
    split_id, total_splits = parse_split(args.split)

    # 一键启动所有分片
    if split_id == 'all':
        check_eprover(args.eprover_path)
        launch_all_splits('provable', args, total_splits)
        return

    check_eprover(args.eprover_path)

    # 收集所有问题文件（排序保证每个分片一致）
    all_files = sorted([f for f in os.listdir(args.problem_dir) if f.endswith('.p')])
    my_files = split_list(all_files, split_id, total_splits)

    output_file = os.path.join(args.output_dir, f'provable_split_{split_id}of{total_splits}.json')

    print("=" * 60)
    print(f"筛选可证明问题 - 分片 {split_id}/{total_splits}")
    print("=" * 60)
    print(f"总问题数: {len(all_files)}")
    print(f"本分片: {len(my_files)} 个问题")
    print(f"时间限制: {args.time_limit}s | 内存: {args.memory_limit}MB | 策略: {args.strategy}")
    print(f"输出: {output_file}")
    print("=" * 60)

    # 加载已有结果（断点续传）
    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    done = set(results.keys())
    print(f"已完成: {len(done)}")

    proved = sum(1 for v in results.values() if v['success'])
    count = len(done)

    for i, filename in enumerate(my_files):
        prob_name = extract_problem_name(filename)
        if prob_name in done:
            continue

        prob_file = os.path.join(args.problem_dir, filename)
        success, cpu_time, output = run_eprover(
            prob_file, args.eprover_path, args.time_limit,
            args.memory_limit, args.strategy
        )

        results[prob_name] = {
            'success': success,
            'time': round(cpu_time, 2)
        }

        count += 1
        if success:
            proved += 1

        # 首次失败时显示详细错误信息，帮助诊断
        if not success and count == 1 and output.startswith('Error:'):
            print(f"\n首个问题失败: {output}")
            print("请检查eprover路径和问题文件路径是否正确\n")

        total = len(my_files)
        print(f"\r[{count}/{total}] 已证明: {proved} | "
              f"{prob_name} {'OK' if success else 'FAIL'} ({cpu_time:.1f}s)     ",
              end='', flush=True)

        # 每20个保存一次
        if count % 20 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    print()

    # 最终保存
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n分片 {split_id}/{total_splits} 完成!")
    print(f"处理: {len(results)} | 可证明: {proved}")
    print(f"结果: {output_file}")


# ============================================================
# 任务2: baseline - Chainy baseline测试
# ============================================================

def cmd_baseline(args):
    split_id, total_splits = parse_split(args.split)

    if split_id == 'all':
        check_eprover(args.eprover_path)
        launch_all_splits('baseline', args, total_splits)
        return

    check_eprover(args.eprover_path)

    # 加载问题列表
    if args.problem_list:
        with open(args.problem_list, 'r') as f:
            all_problems = json.load(f)
            if isinstance(all_problems, dict):
                all_problems = sorted(all_problems.keys())
    else:
        all_problems = sorted([extract_problem_name(f)
                               for f in os.listdir(args.problem_dir) if f.endswith('.p')])

    my_problems = split_list(all_problems, split_id, total_splits)
    output_file = os.path.join(args.output_dir, f'baseline_split_{split_id}of{total_splits}.json')

    print("=" * 60)
    print(f"Chainy Baseline - 分片 {split_id}/{total_splits}")
    print("=" * 60)
    print(f"总问题数: {len(all_problems)} | 本分片: {len(my_problems)}")
    print(f"时间限制: {args.time_limit}s | 内存: {args.memory_limit}MB | 策略: {args.strategy}")
    print("=" * 60)

    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    done = set(results.keys())
    print(f"已完成: {len(done)}")

    proved = sum(1 for v in results.values() if v['success'])
    count = len(done)

    for prob_name in my_problems:
        if prob_name in done:
            continue

        prob_file = find_problem_file(prob_name, args.problem_dir)
        if not prob_file:
            results[prob_name] = {'success': False, 'time': 0.0, 'error': 'file not found'}
            count += 1
            continue

        success, cpu_time, output = run_eprover(
            prob_file, args.eprover_path, args.time_limit,
            args.memory_limit, args.strategy
        )

        results[prob_name] = {'success': success, 'time': round(cpu_time, 2)}
        count += 1
        if success:
            proved += 1

        print(f"\r[{count}/{len(my_problems)}] 已证明: {proved} | "
              f"{prob_name} {'OK' if success else 'FAIL'} ({cpu_time:.1f}s)     ",
              end='', flush=True)

        if count % 20 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    print()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n分片 {split_id}/{total_splits} 完成! 处理: {len(results)} | 可证明: {proved}")


# ============================================================
# 任务3: evaluate - 前提选择评估
# ============================================================

def cmd_evaluate(args):
    split_id, total_splits = parse_split(args.split)

    if split_id == 'all':
        check_eprover(args.eprover_path)
        launch_all_splits('evaluate', args, total_splits)
        return

    check_eprover(args.eprover_path)

    # 加载排序
    with open(args.premise_rankings, 'r') as f:
        rankings = json.load(f)

    # 加载问题列表
    if args.test_problems:
        with open(args.test_problems, 'r') as f:
            all_problems = json.load(f)
            if isinstance(all_problems, dict):
                all_problems = sorted(all_problems.keys())
    else:
        all_problems = sorted(rankings.keys())

    # 过滤无排序的问题
    all_problems = [p for p in all_problems if p in rankings]

    my_problems = split_list(all_problems, split_id, total_splits)
    output_file = os.path.join(args.output_dir, f'evaluate_split_{split_id}of{total_splits}.json')
    temp_dir = os.path.join(args.output_dir, f'temp_{split_id}')
    os.makedirs(temp_dir, exist_ok=True)

    print("=" * 60)
    print(f"前提选择评估 - 分片 {split_id}/{total_splits}")
    print("=" * 60)
    print(f"总问题数: {len(all_problems)} | 本分片: {len(my_problems)}")
    print(f"K值: {args.k_values}")
    print(f"时间限制: {args.time_limit}s | 策略: {args.strategy}")
    print("=" * 60)

    results = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            results = json.load(f)
    done = set(results.keys())
    print(f"已完成: {len(done)}")

    count = len(done)
    success_counts = {k: 0 for k in args.k_values}
    for v in results.values():
        for k in args.k_values:
            if v.get(str(k), {}).get('success'):
                success_counts[k] += 1

    for prob_name in my_problems:
        if prob_name in done:
            continue

        prob_file = find_problem_file(prob_name, args.chainy_dir)
        if not prob_file:
            count += 1
            continue

        conj_name, conj_str, local_axioms = parse_problem_file(prob_file)
        if not conj_str or not local_axioms:
            count += 1
            continue

        ranked = [p for p in rankings[prob_name] if p in local_axioms]
        if not ranked:
            count += 1
            continue

        prob_results = {}
        proved = False

        for k in sorted(args.k_values):
            if proved:
                prob_results[str(k)] = prob_results[str(prev_k)].copy()
                continue

            top_k = ranked[:min(k, len(ranked))]

            temp_file = os.path.join(temp_dir, f"{prob_name}_k{k}.p")
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    f.write(conj_str + "\n")
                    for prem in top_k:
                        f.write(local_axioms[prem] + "\n")

                success, cpu_time, output = run_eprover(
                    temp_file, args.eprover_path, args.time_limit,
                    args.memory_limit, args.strategy
                )

                prob_results[str(k)] = {
                    'success': success,
                    'time': round(cpu_time, 2),
                    'num_premises': len(top_k)
                }

                if success:
                    proved = True
                    prev_k = k
                    success_counts[k] += 1

            except Exception as e:
                prob_results[str(k)] = {'success': False, 'time': 0.0}
            finally:
                try:
                    os.remove(temp_file)
                except:
                    pass

        results[prob_name] = prob_results
        count += 1

        status = ' '.join(f"K{k}:{'Y' if prob_results.get(str(k),{}).get('success') else 'N'}"
                         for k in args.k_values[:4])
        print(f"\r[{count}/{len(my_problems)}] {status} | {prob_name}     ",
              end='', flush=True)

        if count % 10 == 0:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)

    print()

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    # 清理
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass

    print(f"\n分片 {split_id}/{total_splits} 完成!")
    for k in args.k_values:
        print(f"  K={k}: {success_counts[k]}/{len(results)}")


# ============================================================
# merge - 合并所有分片结果
# ============================================================

def cmd_merge(args):
    print("=" * 60)
    print("合并分片结果")
    print("=" * 60)

    # 找到所有分片文件
    split_files = sorted([
        f for f in os.listdir(args.output_dir)
        if re.match(rf'{args.mode}_split_\d+of\d+\.json', f)
    ])

    if not split_files:
        print(f"未找到分片文件（模式: {args.mode}_split_*of*.json）")
        return

    print(f"找到 {len(split_files)} 个分片文件:")
    for f in split_files:
        print(f"  {f}")

    # 合并
    merged = {}
    for filename in split_files:
        filepath = os.path.join(args.output_dir, filename)
        with open(filepath, 'r') as f:
            data = json.load(f)
        merged.update(data)
        print(f"  {filename}: {len(data)} 个问题")

    print(f"\n合并后总计: {len(merged)} 个问题")

    # 统计
    if args.mode in ('provable', 'baseline'):
        proved = sum(1 for v in merged.values() if v.get('success'))
        total_time = sum(v.get('time', 0) for v in merged.values())
        print(f"可证明: {proved} ({proved/max(len(merged),1)*100:.1f}%)")
        print(f"总用时: {total_time:.1f}秒 ({total_time/3600:.1f}小时)")

    elif args.mode == 'evaluate':
        for k in args.k_values:
            count = sum(1 for v in merged.values()
                       if v.get(str(k), {}).get('success'))
            rate = count / max(len(merged), 1) * 100
            print(f"  K={k:4d}: {count:4d}/{len(merged)} ({rate:5.1f}%)")

    # 保存合并结果
    output_file = os.path.join(args.output_dir, f'{args.mode}_merged.json')
    with open(output_file, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"\n合并结果: {output_file}")

    # provable模式: 生成测试集
    if args.mode == 'provable' and args.train_deps:
        proved_names = sorted([p for p, v in merged.items() if v.get('success')])
        print(f"\n可证明问题: {len(proved_names)}")

        # 读取训练集
        train_set = set()
        with open(args.train_deps, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    train_set.add(line.split(':')[0])

        # 测试集 = 可证明 - 训练集
        test_problems = sorted([p for p in proved_names if p not in train_set])

        test_file = os.path.join(args.output_dir, 'test_problems_provable.json')
        with open(test_file, 'w') as f:
            json.dump(test_problems, f, indent=2)

        provable_file = os.path.join(args.output_dir, 'provable_problems.json')
        with open(provable_file, 'w') as f:
            json.dump(proved_names, f, indent=2)

        print(f"训练集: {len(train_set)} | 测试集: {len(test_problems)}")
        print(f"测试集文件: {test_file}")
        print(f"可证明列表: {provable_file}")


# ============================================================
# all分片: 一键后台启动所有分片
# ============================================================

def launch_all_splits(command, args, total_splits):
    """后台启动所有分片"""
    print("=" * 60)
    print(f"一键启动 {total_splits} 个分片")
    print("=" * 60)

    script_path = os.path.abspath(__file__)

    for i in range(1, total_splits + 1):
        # 重建命令行
        cmd = [sys.executable, script_path, command, '--split', f'{i}/{total_splits}']

        # 复制其他参数
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
    print(f"全部完成后运行: python3 {script_path} merge --output_dir {args.output_dir} --mode {command}")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='TWGNN分批并行E-prover测试工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    # === provable ===
    p1 = subparsers.add_parser('provable', help='筛选可证明问题（bushy）')
    p1.add_argument('--split', type=str, default='1/1',
                   help='分片: "1/4"=第1份共4份, "all/4"=后台启动全部')
    p1.add_argument('--problem_dir', type=str, required=True)
    p1.add_argument('--eprover_path', type=str, default='/home/stu1/zhaozhongxu/prove/eprover/PROVER/eprover')
    p1.add_argument('--output_dir', type=str, required=True)
    p1.add_argument('--train_deps', type=str, default=None,
                   help='训练集deps文件（仅merge时用）')
    p1.add_argument('--time_limit', type=int, default=60)
    p1.add_argument('--memory_limit', type=int, default=4000)
    p1.add_argument('--strategy', type=str, default='none',
                   choices=['auto', 'satauto', 'none'])

    # === baseline ===
    p2 = subparsers.add_parser('baseline', help='Chainy baseline测试')
    p2.add_argument('--split', type=str, default='1/1')
    p2.add_argument('--problem_dir', type=str, required=True)
    p2.add_argument('--problem_list', type=str, default=None)
    p2.add_argument('--eprover_path', type=str, default='/home/stu1/zhaozhongxu/prove/eprover/PROVER/eprover')
    p2.add_argument('--output_dir', type=str, required=True)
    p2.add_argument('--time_limit', type=int, default=60)
    p2.add_argument('--memory_limit', type=int, default=10000)
    p2.add_argument('--strategy', type=str, default='none',
                   choices=['auto', 'satauto', 'none'])

    # === evaluate ===
    p3 = subparsers.add_parser('evaluate', help='前提选择评估')
    p3.add_argument('--split', type=str, default='1/1')
    p3.add_argument('--chainy_dir', type=str, required=True)
    p3.add_argument('--premise_rankings', type=str, required=True)
    p3.add_argument('--test_problems', type=str, default=None)
    p3.add_argument('--eprover_path', type=str, default='/home/stu1/zhaozhongxu/prove/eprover/PROVER/eprover')
    p3.add_argument('--output_dir', type=str, required=True)
    p3.add_argument('--k_values', type=int, nargs='+', default=[64, 128, 256, 512])
    p3.add_argument('--time_limit', type=int, default=60)
    p3.add_argument('--memory_limit', type=int, default=10000)
    p3.add_argument('--strategy', type=str, default='none',
                   choices=['auto', 'satauto', 'none'])

    # === merge ===
    p4 = subparsers.add_parser('merge', help='合并分片结果')
    p4.add_argument('--output_dir', type=str, required=True)
    p4.add_argument('--mode', type=str, default='provable',
                   choices=['provable', 'baseline', 'evaluate'],
                   help='合并哪种结果')
    p4.add_argument('--train_deps', type=str, default=None,
                   help='训练集deps文件（provable模式用于生成测试集）')
    p4.add_argument('--k_values', type=int, nargs='+', default=[64, 128, 256, 512])

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command != 'merge':
        os.makedirs(args.output_dir, exist_ok=True)

    start = datetime.now()
    print(f"开始: {start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.command == 'provable':
        cmd_provable(args)
    elif args.command == 'baseline':
        cmd_baseline(args)
    elif args.command == 'evaluate':
        cmd_evaluate(args)
    elif args.command == 'merge':
        cmd_merge(args)

    print(f"\n总用时: {datetime.now() - start}")


if __name__ == '__main__':
    main()
