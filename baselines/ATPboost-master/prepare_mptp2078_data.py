#!/usr/bin/env python3
"""
为 ATPBoost 准备 MPTP2078 数据

从我们的数据格式转换为 ATPBoost 所需格式:
1. conjectures   - 测试问题名列表
2. train_deps    - 训练依赖 (问题:前提1 前提2 ...)
3. features_binary - 每个公式的二值特征向量
4. chronology    - 公式时间顺序
5. statements    - 公式FOF字符串
"""

import os, sys, json, re
from collections import defaultdict

BASE = "/home/stu1/zhaozhongxu/prove"
BUSHY_DIR = os.path.join(BASE, "MPTP2078-master/bushy")
CHAINY_DIR = os.path.join(BASE, "MPTP2078-master/chainy")
STATEMENTS_FILE = os.path.join(BASE, "LCGS_TWGNN/statements")
TEST_PROBLEMS = os.path.join(BASE, "LCGS_TWGNN/dataset_D0_baseline/test_problems_provable.json")

OUT_DIR = os.path.join(BASE, "ATPboost-master/data/MPTP2078")
os.makedirs(OUT_DIR, exist_ok=True)


def parse_axioms(filepath):
    """从.p文件提取axiom名称"""
    axioms = set()
    buffer = []
    with open(filepath) as f:
        for line in f:
            if '%' in line:
                line = line.split('%')[0]
            line = line.strip()
            if not line:
                continue
            buffer.append(line)
            if line.endswith('.'):
                full = ' '.join(buffer)
                buffer = []
                clean = full.replace(' ', '')
                if full.startswith('fof') and ',axiom,' in clean:
                    idx = full.find('(') + 1
                    name = full[idx:full.find(',', idx)].strip()
                    axioms.add(name)
    return axioms


def tokenize_formula(fof_str):
    """提取公式中的符号作为特征"""
    body = re.sub(r'^fof\s*\([^,]+,\s*\w+\s*,\s*', '', fof_str)
    body = re.sub(r'\s*\)\s*\.\s*$', '', body)
    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', body)
    return tokens


print("=== 生成 ATPBoost MPTP2078 数据 ===")

# 1. statements - 直接复制
print("1. statements...")
stmts = {}
with open(STATEMENTS_FILE) as f:
    for line in f:
        line = line.strip()
        if not line or not line.startswith('fof('):
            continue
        idx = line.find('(') + 1
        name = line[idx:line.find(',', idx)].strip()
        stmts[name] = line

with open(os.path.join(OUT_DIR, "statements"), 'w') as f:
    for name in sorted(stmts):
        f.write(stmts[name] + '\n')
print(f"   {len(stmts)} formulas")

# 2. conjectures - 280个测试问题
print("2. conjectures...")
test_probs = json.load(open(TEST_PROBLEMS))
if isinstance(test_probs, dict):
    test_probs = sorted(test_probs.keys())
else:
    test_probs = sorted(test_probs)

with open(os.path.join(OUT_DIR, "conjectures"), 'w') as f:
    for p in test_probs:
        f.write(p + '\n')
print(f"   {len(test_probs)} conjectures")

# 3. train_deps - 从bushy提取训练依赖
print("3. train_deps...")
# 获取训练集(与D0一致的1000个问题)
d0_data = [json.loads(l) for l in open(
    os.path.join(BASE, "LCGS_TWGNN/dataset_D0_baseline/train/raw/train.json"))]
train_conjs = set(d[0] for d in d0_data)

train_deps = []
for fn in sorted(os.listdir(BUSHY_DIR)):
    if not fn.endswith('.p'):
        continue
    prob_name = fn.replace('.p', '').split('__')[-1]
    if prob_name not in train_conjs:
        continue
    bushy_axioms = parse_axioms(os.path.join(BUSHY_DIR, fn))
    if bushy_axioms:
        deps_str = f"{prob_name}:{' '.join(sorted(bushy_axioms))}"
        train_deps.append(deps_str)

with open(os.path.join(OUT_DIR, "train_deps"), 'w') as f:
    for d in train_deps:
        f.write(d + '\n')
print(f"   {len(train_deps)} training problems with deps")

# 4. chronology - 公式时间顺序(从chainy文件的出现顺序推断)
print("4. chronology...")
# 用chainy中最大的文件来获取全序
chronology = []
seen = set()

# 从FormulasInMMLOrder获取(如果存在)
formulas_order = os.path.join(BASE, "MPTP2078-master/FormulasInMMLOrder")
if os.path.exists(formulas_order):
    with open(formulas_order) as f:
        for line in f:
            name = line.strip()
            if name and name in stmts and name not in seen:
                chronology.append(name)
                seen.add(name)
    print(f"   从FormulasInMMLOrder获取: {len(chronology)}")

# 补充未覆盖的公式
for name in sorted(stmts):
    if name not in seen:
        chronology.append(name)
        seen.add(name)

with open(os.path.join(OUT_DIR, "chronology"), 'w') as f:
    for name in chronology:
        f.write(name + '\n')
print(f"   {len(chronology)} formulas in chronology")

# 5. features_binary - 每个公式的二值符号特征
print("5. features_binary...")
# 收集所有符号
all_symbols = set()
formula_symbols = {}
for name, fof in stmts.items():
    tokens = set(tokenize_formula(fof))
    formula_symbols[name] = tokens
    all_symbols.update(tokens)

with open(os.path.join(OUT_DIR, "features_binary"), 'w') as f:
    for name in sorted(stmts):
        syms = formula_symbols.get(name, set())
        # ATPBoost格式: name:"sym1", "sym2", ...
        features_str = ', '.join(f'"{s}"' for s in sorted(syms))
        f.write(f'{name}:{features_str}\n')
print(f"   {len(stmts)} formulas, {len(all_symbols)} unique symbols")

print(f"\n=== 数据已保存到 {OUT_DIR} ===")
for f in ['conjectures', 'train_deps', 'features_binary', 'chronology', 'statements']:
    path = os.path.join(OUT_DIR, f)
    lines = sum(1 for _ in open(path))
    print(f"   {f}: {lines} lines")
