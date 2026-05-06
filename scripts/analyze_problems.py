#!/usr/bin/env python3
"""
MPTP2078问题统计分析脚本
分析所有2078个问题文件，统计：
  - 子句个数（fof公式数）
  - 文字总数（原子公式数）
  - 变元项个数（变量出现次数）
  - 子句最大文字数（单个公式中最多的原子数）
"""

import os
import re
import json
import argparse
from tqdm import tqdm


# ============================================================
# FOF公式分词器
# ============================================================

def tokenize(text):
    """将FOF公式字符串分词"""
    tokens = []
    i = 0
    n = len(text)
    while i < n:
        # 跳过空白
        if text[i].isspace():
            i += 1
        # 三字符运算符
        elif i + 2 < n and text[i:i+3] in ('<=>', '<~>'):
            tokens.append(text[i:i+3])
            i += 3
        # 双字符运算符
        elif i + 1 < n and text[i:i+2] in ('=>', '!='):
            tokens.append(text[i:i+2])
            i += 2
        # 单字符运算符和标点
        elif text[i] in '()[]~&|,:=!?.':
            tokens.append(text[i])
            i += 1
        # 标识符和数字
        elif text[i].isalnum() or text[i] in '_$':
            j = i
            while j < n and (text[j].isalnum() or text[j] in '_$'):
                j += 1
            tokens.append(text[i:j])
            i = j
        else:
            i += 1
    return tokens


# ============================================================
# FOF公式递归下降分析器
# ============================================================

class FOFAnalyzer:
    """递归下降分析FOF公式，统计文字数和变元项数"""

    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0
        self.literal_count = 0
        self.var_occurrences = 0

    def peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self):
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def analyze(self):
        """分析公式，返回 (文字数, 变元项数)"""
        try:
            self._parse_formula()
        except Exception:
            pass
        return self.literal_count, self.var_occurrences

    def _parse_formula(self):
        """解析公式（处理二元连接词）"""
        self._parse_unary()
        while self.peek() in ('=>', '<=>', '&', '|', '<~>'):
            self.consume()
            self._parse_unary()

    def _parse_unary(self):
        """解析一元公式（否定、量词、括号、原子）"""
        tok = self.peek()
        if tok == '~':
            self.consume()
            self._parse_unary()
        elif tok in ('!', '?'):
            self.consume()       # ! 或 ?
            self.consume()       # [
            # 跳过变量声明列表 [A, B, ...]
            while self.peek() not in (']', None):
                self.consume()
            self.consume()       # ]
            self.consume()       # :
            self._parse_unary()
        elif tok == '(':
            self.consume()       # (
            self._parse_formula()
            if self.peek() == ')':
                self.consume()   # )
        elif tok is not None:
            self._parse_atom()

    def _parse_atom(self):
        """解析原子公式（谓词调用或等式）"""
        self.literal_count += 1
        self._parse_term()
        # 检查等式/不等式
        if self.peek() in ('=', '!='):
            self.consume()
            self._parse_term()

    def _parse_term(self):
        """解析项（变量、常量、函数调用）"""
        tok = self.consume()
        if tok is None:
            return
        # 变量以大写字母开头
        if tok[0].isupper():
            self.var_occurrences += 1
        # 函数/谓词调用
        if self.peek() == '(':
            self.consume()       # (
            if self.peek() != ')':
                self._parse_term()
                while self.peek() == ',':
                    self.consume()   # ,
                    self._parse_term()
            if self.peek() == ')':
                self.consume()   # )


# ============================================================
# 问题文件解析
# ============================================================

def extract_formula_body(fof_line):
    """
    从fof语句中提取公式体
    fof(name, type, FORMULA).  →  FORMULA
    """
    line = fof_line.strip()
    # 去掉末尾的 ).
    if line.endswith(').'):
        line = line[:-2]
    elif line.endswith(')'):
        line = line[:-1]

    if not line.startswith('fof('):
        return None

    # 跳过 fof(name, type, 找到第二个逗号后的公式
    comma_count = 0
    i = 4  # 跳过 "fof("
    while i < len(line) and comma_count < 2:
        if line[i] == ',':
            comma_count += 1
        i += 1

    if comma_count < 2:
        return None

    return line[i:].strip()


def analyze_problem_file(file_path):
    """
    分析单个问题文件

    返回:
        dict: {
            'num_clauses': 子句个数,
            'total_literals': 文字总数,
            'total_var_terms': 变元项个数,
            'max_clause_literals': 子句最大文字数
        }
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 解析所有fof语句（处理可能的多行语句）
    fof_statements = []
    buffer = []

    for line in content.split('\n'):
        # 移除注释
        if '%' in line:
            line = line[:line.index('%')]
        line = line.strip()
        if not line:
            continue
        buffer.append(line)
        if line.endswith('.'):
            full = ' '.join(buffer)
            buffer = []
            if full.startswith('fof('):
                fof_statements.append(full)

    num_clauses = len(fof_statements)
    total_literals = 0
    total_var_terms = 0
    max_clause_literals = 0

    for stmt in fof_statements:
        formula = extract_formula_body(stmt)
        if formula is None:
            continue

        tokens = tokenize(formula)
        analyzer = FOFAnalyzer(tokens)
        literals, var_terms = analyzer.analyze()

        total_literals += literals
        total_var_terms += var_terms
        max_clause_literals = max(max_clause_literals, literals)

    return {
        'num_clauses': num_clauses,
        'total_literals': total_literals,
        'total_var_terms': total_var_terms,
        'max_clause_literals': max_clause_literals
    }


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='MPTP2078问题统计分析',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--chainy_dir', type=str,
                        default='/home/stu1/zhaozhongxu/prove/MPTP2078-master/chainy',
                        help='Chainy问题目录')
    parser.add_argument('--output', type=str,
                        default='/home/stu1/zhaozhongxu/prove/TWGNN/data/problem_statistics.json',
                        help='输出文件路径')
    args = parser.parse_args()

    # 收集所有问题文件
    problem_files = sorted([f for f in os.listdir(args.chainy_dir) if f.endswith('.p')])
    print(f"共 {len(problem_files)} 个问题文件")

    all_stats = {}

    for filename in tqdm(problem_files, desc="分析问题"):
        # 提取问题名: compts_1__t10_compts_1 → t10_compts_1
        prob_name = filename[:-2]
        if '__' in prob_name:
            short_name = prob_name.split('__', 1)[1]
        else:
            short_name = prob_name

        file_path = os.path.join(args.chainy_dir, filename)
        stats = analyze_problem_file(file_path)
        all_stats[short_name] = stats

    # 保存结果
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_stats, f, indent=2)

    # 打印统计摘要
    n = len(all_stats)
    clauses = [s['num_clauses'] for s in all_stats.values()]
    literals = [s['total_literals'] for s in all_stats.values()]
    var_terms = [s['total_var_terms'] for s in all_stats.values()]
    max_lits = [s['max_clause_literals'] for s in all_stats.values()]

    print(f"\n{'='*60}")
    print(f"MPTP2078 问题统计摘要 ({n} 个问题)")
    print(f"{'='*60}")
    print(f"{'指标':<20} {'最小':>8} {'最大':>8} {'平均':>10} {'总计':>12}")
    print(f"{'-'*60}")
    print(f"{'子句个数':<18} {min(clauses):>8} {max(clauses):>8} {sum(clauses)/n:>10.1f} {sum(clauses):>12}")
    print(f"{'文字总数':<18} {min(literals):>8} {max(literals):>8} {sum(literals)/n:>10.1f} {sum(literals):>12}")
    print(f"{'变元项个数':<16} {min(var_terms):>8} {max(var_terms):>8} {sum(var_terms)/n:>10.1f} {sum(var_terms):>12}")
    print(f"{'子句最大文字数':<14} {min(max_lits):>8} {max(max_lits):>8} {sum(max_lits)/n:>10.1f} {'':>12}")
    print(f"{'='*60}")
    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
