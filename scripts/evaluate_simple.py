"""
简化版ATP评估脚本 - 仅使用预计算的前提排序和E-prover进行评估
不需要模型，直接使用rankings文件
"""
import os
import sys
import subprocess
import argparse
import json
import re
from pathlib import Path
from tqdm import tqdm


def parse_problem_file(file_path):
    """
    解析问题文件，提取猜想和所有可用公理

    返回: (猜想名, 猜想字符串, {公理名: 公理字符串})
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    conj_name = None
    conj_str = None
    local_axioms = {}
    buffer = []

    for line in lines:
        # 移除注释
        if '%' in line:
            line = line.split('%')[0]

        line = line.strip()

        if not line or line.startswith('#'):
            continue

        buffer.append(line)

        # 完整的fof语句
        if line.endswith('.'):
            full_fof = " ".join(buffer)
            buffer = []

            if not full_fof.startswith('fof'):
                continue

            try:
                # 提取名称
                name_start = full_fof.find('(') + 1
                name_end = full_fof.find(',', name_start)
                name = full_fof[name_start:name_end].strip()

                clean_fof_check = full_fof.replace(' ', '')

                if ',conjecture,' in clean_fof_check:
                    conj_name = name
                    conj_str = full_fof
                elif ',axiom,' in clean_fof_check:
                    local_axioms[name] = full_fof
            except Exception:
                continue

    return conj_name, conj_str, local_axioms


def run_eprover(problem_file, eprover_path, time_limit=60, memory_limit=10000, strategy='satauto'):
    """
    运行E-prover证明

    参数:
        problem_file: 问题文件路径
        eprover_path: E-prover可执行文件路径
        time_limit: CPU时间限制（秒）
        memory_limit: 内存限制（MB）
        strategy: 策略选择 ('auto', 'satauto', 'none')
    """
    # E-prover命令（参考baseline.py）
    eprover_cmd = [eprover_path]

    # 策略选择
    if strategy == 'auto':
        eprover_cmd.append('--auto-schedule')
    elif strategy == 'satauto':
        eprover_cmd.append('--satauto-schedule')
    # strategy == 'none': 不添加任何策略参数

    eprover_cmd.extend([
        '--free-numbers',         # 允许数字推理（重要！）
        '-s',                     # 打印统计信息
        '-R',                     # 打印资源使用
        '-p',                     # 输出证明过程（用于提取使用的公理）
        f'--cpu-limit={time_limit}',
        f'--memory-limit={memory_limit}',
        '--tstp-format',          # 使用TSTP格式输出
        problem_file
    ])

    start_time = subprocess.time if hasattr(subprocess, 'time') else None
    import time
    start_time = time.time()

    try:
        result = subprocess.run(
            eprover_cmd,
            capture_output=True,
            text=True,
            timeout=time_limit + 10  # 给予额外缓冲时间
        )

        end_time = time.time()
        wall_time = end_time - start_time

        # 检查是否成功证明（schedule模式用%前缀，普通模式用#前缀）
        success = '# Proof found!' in result.stdout or '% Proof found!' in result.stdout
        output = result.stdout

        # 提取CPU时间（更准确，兼容#和%前缀）
        time_match = re.search(r'[#%] Total time\s*:\s*([\d.]+)', output)
        if time_match:
            proof_time = float(time_match.group(1))
        else:
            # 如果没有找到，使用墙钟时间
            proof_time = wall_time

        return success, output, proof_time

    except subprocess.TimeoutExpired:
        end_time = time.time()
        wall_time = end_time - start_time
        return False, 'Timeout', wall_time
    except Exception as e:
        return False, f'Error: {str(e)}', 0.0


def extract_used_axioms(eprover_output):
    """
    从E-prover TSTP证明输出中提取使用的公理名称

    TSTP格式中，输入公理的引用形如:
      fof(c_0_1, axiom, ..., file('path/problem.p', axiom_name)).
    提取其中的 axiom_name，排除 conjecture 本身
    """
    used = set()
    # 匹配 file('...', name) 中的 name
    pattern = re.compile(r"file\('[^']*',\s*(\w+)\)")
    for line in eprover_output.split('\n'):
        if 'file(' not in line:
            continue
        match = pattern.search(line)
        if match:
            name = match.group(1)
            # 排除conjecture本身（只统计被使用的公理/前提）
            if ', conjecture,' not in line and ',conjecture,' not in line:
                used.add(name)
    return sorted(used)


def evaluate_single_problem(prob_file, conj_name, conj_str, local_axioms,
                            eprover_path, k_values, temp_dir, premise_ranking,
                            time_limit=60, memory_limit=10000, strategy='satauto'):
    """
    评估单个问题，测试不同的K值

    参数:
        premise_ranking: 预计算的前提排序列表（必须提供）
        time_limit: E-prover时间限制
        memory_limit: E-prover内存限制
        strategy: E-prover策略

    返回: {k: {'success': bool, 'time': float, 'used_axioms': list, 'selected': list}}
    """
    local_candidates = list(local_axioms.keys())

    if len(local_candidates) == 0:
        return {k: {'success': False, 'time': 0.0, 'used_axioms': [], 'selected': []}
                for k in k_values}

    # 使用预计算的排序，只保留在local_axioms中存在的前提
    ranked_premises = [p for p in premise_ranking if p in local_axioms]

    if not ranked_premises:
        return {k: {'success': False, 'time': 0.0, 'used_axioms': [], 'selected': []}
                for k in k_values}

    results = {}

    # 对每个K值进行测试
    for k in k_values:
        top_k = min(k, len(ranked_premises))
        selected_premises = ranked_premises[:top_k]

        # 生成临时问题文件
        temp_file = os.path.join(temp_dir, f"{conj_name}_k{k}.p")
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(conj_str + "\n")
            for prem in selected_premises:
                if prem in local_axioms:
                    f.write(local_axioms[prem] + "\n")

        # 运行E-prover
        success, output, proof_time = run_eprover(
            temp_file, eprover_path, time_limit, memory_limit, strategy
        )

        used_axioms = extract_used_axioms(output) if success else []

        results[k] = {
            'success': success,
            'time': proof_time,
            'used_axioms': used_axioms,
            'num_used': len(used_axioms),           # NUiP: 证明中使用的前提数
            'num_selected': len(selected_premises),  # NSel: 选择的前提数
            'num_total_axioms': len(local_axioms),   # NAxP: 原始前提总数
            'selected': selected_premises
        }

        # 清理临时文件
        try:
            os.remove(temp_file)
        except:
            pass

        # 如果已经成功证明，更大的K值不需要再测试
        if success:
            for larger_k in k_values:
                if larger_k > k and larger_k not in results:
                    results[larger_k] = results[k].copy()
            break

    # 填充未测试的K值
    for k in k_values:
        if k not in results:
            results[k] = {
                'success': False,
                'time': 0.0,
                'used_axioms': [],
                'selected': []
            }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='ATP评估脚本（使用预计算排序）'
    )

    # 必需参数
    parser.add_argument('--chainy_dir', type=str, default="/home/stu1/zhaozhongxu/prove/MPTP2078-master/chainy",
                       help='Chainy问题目录')
    parser.add_argument('--test_problems', type=str, default="/home/stu1/zhaozhongxu/prove/TWGNN/data/mptp2078_custom/test_problems.json",
                       help='测试问题列表JSON文件')
    parser.add_argument('--premise_rankings', type=str, default="/home/stu1/zhaozhongxu/prove/TWGNN/model-save_v2/rankings_top512.json",
                       help='前提排序JSON文件（必需）')
    parser.add_argument('--eprover_path', type=str, default="/home/stu1/zhaozhongxu/prove/eprover/PROVER/eprover",
                       help='E-prover路径（使用编译的release版本）')
    parser.add_argument('--output_dir', type=str, default="/home/stu1/zhaozhongxu/prove/TWGNN/model-save_v2/results_top_512",
                       help='输出目录')

    # 可选参数
    parser.add_argument('--k_values', type=int, nargs='+',
                       default=[512],
                       help='要测试的K值列表')
    parser.add_argument('--time_limit', type=int, default=60,
                       help='E-prover时间限制（秒）')
    parser.add_argument('--memory_limit', type=int, default=10000,
                       help='E-prover内存限制（MB）')
    parser.add_argument('--strategy', type=str, default='satauto',
                       choices=['auto', 'satauto', 'none'],
                       help='E-prover策略选择')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    print("="*60)
    print("ATP评估（使用预计算排序）")
    print("="*60)
    print(f"Chainy目录: {args.chainy_dir}")
    print(f"测试问题: {args.test_problems}")
    print(f"前提排序: {args.premise_rankings}")
    print(f"E-prover: {args.eprover_path}")
    print(f"K值: {args.k_values}")
    print(f"时间限制: {args.time_limit}秒")
    print(f"内存限制: {args.memory_limit}MB")
    print(f"策略: {args.strategy}")
    print("="*60)

    # 加载测试问题列表
    with open(args.test_problems, 'r') as f:
        test_problems = json.load(f)
        if isinstance(test_problems, dict):
            test_problems = list(test_problems.keys())

    # 加载前提排序
    with open(args.premise_rankings, 'r') as f:
        premise_rankings = json.load(f)

    print(f"\n测试问题数: {len(test_problems)}")
    print(f"有排序的问题数: {len(premise_rankings)}")

    # 评估
    all_results = {}
    success_counts = {k: 0 for k in args.k_values}
    total_times = {k: 0.0 for k in args.k_values}

    for problem in tqdm(test_problems, desc="评估进度"):
        # 查找问题文件
        prob_file = None
        candidates = [
            os.path.join(args.chainy_dir, f"{problem}.p"),
        ]

        # 尝试MPTP2078格式
        match = re.match(r't\d+_(.+)', problem)
        if match:
            theory = match.group(1)
            candidates.append(os.path.join(args.chainy_dir, f"{theory}__{problem}.p"))

        for candidate in candidates:
            if os.path.exists(candidate):
                prob_file = candidate
                break

        if not prob_file:
            continue

        # 解析问题文件
        conj_name, conj_str, local_axioms = parse_problem_file(prob_file)

        if not conj_str or not local_axioms:
            continue

        # 获取前提排序
        if problem not in premise_rankings:
            continue

        # 评估
        results = evaluate_single_problem(
            prob_file, conj_name, conj_str, local_axioms,
            args.eprover_path, args.k_values, temp_dir,
            premise_rankings[problem],
            args.time_limit, args.memory_limit, args.strategy
        )

        all_results[problem] = results

        # 统计
        for k in args.k_values:
            if results[k]['success']:
                success_counts[k] += 1
                total_times[k] += results[k]['time']

    # 保存结果
    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # 计算 Spre（选择精准度）和 Sel（选择度）
    # Spre = (1/|Proved|) * Σ NUiP(prob)/NSel(prob)   值越高越好
    # Sel  = (1/|Proved|) * Σ NSel(prob)/NAxP(prob)    值越低越好
    metrics = {}
    for k in args.k_values:
        spre_sum = 0.0
        sel_sum = 0.0
        proved_count = 0

        for prob, results in all_results.items():
            r = results[k]
            if r['success']:
                proved_count += 1
                n_uip = r.get('num_used', 0)
                n_sel = r.get('num_selected', 1)
                n_axp = r.get('num_total_axioms', 1)

                if n_sel > 0:
                    spre_sum += n_uip / n_sel
                if n_axp > 0:
                    sel_sum += n_sel / n_axp

        spre = spre_sum / proved_count if proved_count > 0 else 0.0
        sel = sel_sum / proved_count if proved_count > 0 else 0.0
        metrics[k] = {'Spre': spre, 'Sel': sel}

    # 保存指标到结果文件
    metrics_file = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    # 打印摘要
    print("\n" + "="*60)
    print("评估完成!")
    print("="*60)
    print(f"评估问题数: {len(all_results)}")
    print("\n各K值结果:")
    print(f"  {'K':>5}  {'证明数':>6}  {'证明率':>8}  {'平均时间':>8}  {'Spre':>8}  {'Sel':>8}")
    print(f"  {'-'*52}")
    for k in args.k_values:
        success_rate = (success_counts[k] / len(all_results) * 100) if all_results else 0
        avg_time = (total_times[k] / success_counts[k]) if success_counts[k] > 0 else 0
        spre = metrics[k]['Spre']
        sel = metrics[k]['Sel']
        print(f"  {k:>5}  {success_counts[k]:>5}   {success_rate:>6.1f}%  {avg_time:>7.2f}s  {spre:>8.4f}  {sel:>8.4f}")

    print(f"\n指标说明:")
    print(f"  Spre (选择精准度): 值越高越好，表示选出的前提中有用前提比例高")
    print(f"  Sel  (选择度):     值越低越好，表示从原始前提中选择的比例小")
    print(f"\n结果已保存到: {output_file}")
    print(f"指标已保存到: {metrics_file}")

    # 清理临时目录
    try:
        import shutil
        shutil.rmtree(temp_dir)
    except:
        pass


if __name__ == '__main__':
    main()
