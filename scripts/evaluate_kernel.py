#!/usr/bin/env python3
"""
TF-IDF (Kernel) 前提选择方法

Phase A: 生成排序
  - 加载statements.pkl中的公式
  - TF-IDF向量化 + 余弦相似度排序
  - 输出rankings_kernel.json

Phase B: 使用evaluate_workflow.py进行工作流评估

使用方式:
  # 生成排序
  python3 evaluate_kernel.py generate \
    --statements /path/to/statements.pkl \
    --test_problems /path/to/test_problems.json \
    --chainy_dir /path/to/chainy \
    --output rankings_kernel.json

  # 评估（使用evaluate_workflow.py）
  python3 evaluate_workflow.py run --split all/8 \
    --premise_rankings rankings_kernel.json \
    --output_dir results_kernel_workflow
"""

import os
import sys
import json
import pickle
import argparse
import re
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_problems import tokenize, extract_formula_body
from run import find_problem_file
from evaluate_simple import parse_problem_file


def extract_body_from_statement(fof_str):
    """从fof语句字符串中提取公式体"""
    return extract_formula_body(fof_str)


def tokenize_formula(fof_str):
    """将fof语句分词，返回空格连接的token字符串（用于TF-IDF）"""
    body = extract_body_from_statement(fof_str)
    if body is None:
        # 回退：对整个字符串分词
        body = fof_str
    tokens = tokenize(body)
    return ' '.join(tokens)


def generate_rankings(args):
    """生成TF-IDF余弦相似度排序"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("错误: 需要scikit-learn。请运行: pip install scikit-learn")
        sys.exit(1)

    print("=" * 60)
    print("TF-IDF (Kernel) 排序生成")
    print("=" * 60)

    # 加载公式
    print("加载statements.pkl...")
    with open(args.statements, 'rb') as f:
        statements = pickle.load(f)
    print(f"  公式总数: {len(statements)}")

    # 加载测试问题列表
    print("加载测试问题列表...")
    with open(args.test_problems, 'r') as f:
        test_problems = json.load(f)
        if isinstance(test_problems, dict):
            test_problems = sorted(test_problems.keys())
    print(f"  测试问题数: {len(test_problems)}")

    # 分词所有公式
    print("分词所有公式...")
    all_names = list(statements.keys())
    all_tokens = []
    for name in all_names:
        all_tokens.append(tokenize_formula(statements[name]))

    # TF-IDF向量化
    print("构建TF-IDF向量...")
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_tokens)
    print(f"  TF-IDF矩阵: {tfidf_matrix.shape}")

    # 建立名称到索引映射
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}

    # 为每个测试问题生成排序
    print("生成排序...")
    rankings = {}
    processed = 0

    for prob_name in test_problems:
        # 找到问题文件
        prob_file = find_problem_file(prob_name, args.chainy_dir)
        if not prob_file:
            continue

        # 解析获取猜想和本地公理
        conj_name, conj_str, local_axioms = parse_problem_file(prob_file)
        if not conj_str or not local_axioms:
            continue

        # 获取猜想的TF-IDF向量
        conj_tokens = tokenize_formula(conj_str)
        conj_vec = vectorizer.transform([conj_tokens])

        # 计算与每个本地公理的余弦相似度
        similarities = []
        for ax_name in local_axioms:
            if ax_name in name_to_idx:
                ax_idx = name_to_idx[ax_name]
                sim = cosine_similarity(conj_vec, tfidf_matrix[ax_idx:ax_idx+1])[0, 0]
            else:
                # 公理不在statements.pkl中，使用本地公式计算
                ax_tokens = tokenize_formula(local_axioms[ax_name])
                ax_vec = vectorizer.transform([ax_tokens])
                sim = cosine_similarity(conj_vec, ax_vec)[0, 0]
            similarities.append((ax_name, float(sim)))

        # 按相似度降序排列
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k = getattr(args, 'top_k', 1024)
        rankings[prob_name] = [name for name, _ in similarities[:top_k]]

        processed += 1
        if processed % 100 == 0:
            print(f"\r  已处理: {processed}/{len(test_problems)}     ", end='', flush=True)

    print(f"\r  已处理: {processed}/{len(test_problems)}     ")

    # 保存排序
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(rankings, f, indent=2)

    print(f"\n排序已保存: {args.output}")
    print(f"有排序的问题数: {len(rankings)}")


def main():
    parser = argparse.ArgumentParser(
        description='TF-IDF (Kernel) 前提选择',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command')

    # === generate ===
    p1 = subparsers.add_parser('generate', help='生成TF-IDF排序')
    p1.add_argument('--statements', type=str,
                    default='/home/stu1/zhaozhongxu/prove/TWGNN/data/mptp2078_custom/statements.pkl',
                    help='公式字符串文件')
    p1.add_argument('--test_problems', type=str,
                    default='/home/stu1/zhaozhongxu/prove/TWGNN/data/mptp2078_custom/test_problems.json',
                    help='测试问题列表')
    p1.add_argument('--chainy_dir', type=str,
                    default='/home/stu1/zhaozhongxu/prove/MPTP2078-master/chainy',
                    help='Chainy问题目录')
    p1.add_argument('--output', type=str,
                    default='/home/stu1/zhaozhongxu/prove/TWGNN/results_kernel/rankings_kernel.json',
                    help='输出排序文件')
    p1.add_argument('--top_k', type=int, default=1024,
                    help='保存前 top_k 个前提 [默认: 1024]')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    start = datetime.now()
    print(f"开始: {start.strftime('%Y-%m-%d %H:%M:%S')}\n")

    if args.command == 'generate':
        generate_rankings(args)

    print(f"\n总用时: {datetime.now() - start}")


if __name__ == '__main__':
    main()
