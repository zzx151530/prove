#!/usr/bin/env python3
"""
DeepMath 训练 + 排序生成
用法:
  python3 train_and_rank.py --mode train --data_dir ../LAH_TWGNN/dataset_D0 ...
  python3 train_and_rank.py --mode rank --model model.pt --output rankings.json ...
  python3 train_and_rank.py --mode both ...
"""

import os, sys, json, glob, pickle, argparse, copy, time, logging
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deepmath_model import DeepMathModel, DeepMathDataset, build_char_vocab, encode_formula, load_statements


BASE = os.environ.get("PROVER_BASE",
    "/home/he3090/users/zhaozhongxu/prover")


def setup_logger(logfile=None):
    logger = logging.getLogger("deepmath")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(sh)
    if logfile:
        fh = logging.FileHandler(logfile)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
    return logger


# ================================================================
# 训练
# ================================================================

def train(args, logger, datasets=None):
    """训练DeepMath模型"""
    device = torch.device(args.device)

    # 构建词表
    statements_dir = os.path.join(args.data_dir, "statements")
    vocab_path = os.path.join(args.model_save, "vocab.pkl")
    os.makedirs(args.model_save, exist_ok=True)

    if os.path.exists(vocab_path):
        vocab = pickle.load(open(vocab_path, 'rb'))
        logger.info(f"加载词表: {len(vocab)} 字符")
    else:
        vocab = build_char_vocab(statements_dir)
        pickle.dump(vocab, open(vocab_path, 'wb'))
        logger.info(f"构建词表: {len(vocab)} 字符")

    # 加载数据集
    if datasets is None:
        datasets = [args.data_dir]

    all_ds = []
    for ds_path in datasets:
        train_json = os.path.join(ds_path, "train", "raw", "train.json")
        stmts_path = os.path.join(ds_path, "statements")
        if not os.path.exists(stmts_path):
            stmts_path = os.path.join(args.data_dir, "statements")
        ds = DeepMathDataset(train_json, stmts_path, vocab, max_len=args.max_len)
        all_ds.append(ds)
        logger.info(f"  数据集: {ds_path} ({len(ds)} 样本)")

    if len(all_ds) > 1:
        combined = ConcatDataset(all_ds)
    else:
        combined = all_ds[0]
    logger.info(f"  总样本: {len(combined)}")

    loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=True)

    # 创建模型
    torch.manual_seed(args.seed)
    model = DeepMathModel(len(vocab), embed_dim=80, out_dim=args.dim).to(device)
    logger.info(f"  模型参数: {sum(p.numel() for p in model.parameters()):,}")

    # 加载已有模型 (fine-tune)
    if args.init_model and os.path.exists(args.init_model):
        ckpt = torch.load(args.init_model, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        logger.info(f"  从已有模型初始化: {args.init_model}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer)

    # 训练
    top_models = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for conj, prem, labels in loader:
            conj, prem, labels = conj.to(device), prem.to(device), labels.to(device)
            optimizer.zero_grad()
            loss, pred = model(conj, prem, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(loader)
        acc = correct / total * 100
        scheduler.step(avg_loss)

        if epoch % 10 == 0 or epoch == args.epochs:
            logger.info(f"  Epoch {epoch}/{args.epochs}: loss={avg_loss:.4f} acc={acc:.1f}%")

        sd = copy.deepcopy(model.state_dict())
        top_models.append((avg_loss, sd))
        top_models = sorted(top_models, key=lambda x: x[0])[:5]

    # 保存
    avg_sd = {}
    sds = [sd for _, sd in top_models]
    for key in sds[0]:
        avg_sd[key] = sum(sd[key] for sd in sds) / len(sds)

    save_data = {'model': avg_sd, 'vocab': vocab, 'dim': args.dim, 'max_len': args.max_len}
    torch.save(save_data, os.path.join(args.model_save, "best.pt"))
    logger.info(f"  模型保存: {args.model_save}/best.pt")
    return vocab


# ================================================================
# 排序
# ================================================================

def rank(args, logger, vocab=None):
    """生成Top-K前提排序"""
    device = torch.device(args.device)

    # 加载模型
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    if vocab is None:
        vocab = ckpt.get('vocab')
        if vocab is None:
            vocab = pickle.load(open(os.path.join(os.path.dirname(args.model), "vocab.pkl"), 'rb'))

    dim = ckpt.get('dim', args.dim)
    max_len = ckpt.get('max_len', args.max_len)
    model = DeepMathModel(len(vocab), embed_dim=80, out_dim=dim).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    logger.info(f"模型加载: {args.model}")

    # 加载测试问题
    test_problems = json.load(open(args.test_problems))
    logger.info(f"测试问题: {len(test_problems)}")

    # 加载chainy目录下所有可用前提
    chainy_dir = args.chainy_dir
    statements_dir = args.statements_dir

    # 缓存所有前提的编码
    all_premises = set()
    for prob in test_problems:
        chainy_file = glob.glob(os.path.join(chainy_dir, f"*__{prob}.p"))
        if not chainy_file:
            continue
        with open(chainy_file[0]) as f:
            for line in f:
                line = line.strip()
                if line.startswith('fof(') and ',axiom,' in line.replace(' ', ''):
                    name = line[line.find('(') + 1:line.find(',')].strip()
                    all_premises.add(name)
    logger.info(f"总前提数: {len(all_premises)}")

    # 加载所有公式文本
    all_formulas = load_statements(statements_dir)
    logger.info(f"公式库: {len(all_formulas)} 条")

    # 预编码所有前提
    prem_cache = {}
    for name in all_premises:
        text = all_formulas.get(name, name)
        prem_cache[name] = torch.tensor(
            encode_formula(text, vocab, max_len), dtype=torch.long)

    # 对每个测试问题生成排序
    rankings = {}
    for i, prob in enumerate(test_problems):
        # 获取该问题可见的前提
        chainy_file = glob.glob(os.path.join(chainy_dir, f"*__{prob}.p"))
        if not chainy_file:
            continue

        visible_prems = []
        with open(chainy_file[0]) as f:
            for line in f:
                line = line.strip()
                if line.startswith('fof(') and ',axiom,' in line.replace(' ', ''):
                    name = line[line.find('(') + 1:line.find(',')].strip()
                    if name != prob and name in prem_cache:
                        visible_prems.append(name)

        if not visible_prems:
            continue

        # 编码猜想
        conj_text = all_formulas.get(prob, prob)
        conj_ids = torch.tensor(
            encode_formula(conj_text, vocab, max_len), dtype=torch.long)

        # 批量打分
        scores = []
        batch_size = 256
        with torch.no_grad():
            for j in range(0, len(visible_prems), batch_size):
                batch_prems = visible_prems[j:j + batch_size]
                conj_batch = conj_ids.unsqueeze(0).expand(len(batch_prems), -1).to(device)
                prem_batch = torch.stack([prem_cache[p] for p in batch_prems]).to(device)
                batch_scores = model.score(conj_batch, prem_batch).cpu().tolist()
                scores.extend(zip(batch_prems, batch_scores))

        # 排序取Top-K
        scores.sort(key=lambda x: -x[1])
        rankings[prob] = [p for p, s in scores[:args.top_k]]

        if (i + 1) % 50 == 0:
            logger.info(f"  排序进度: {i+1}/{len(test_problems)}")

    # 保存
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    json.dump(rankings, open(args.output, 'w'))
    logger.info(f"排序保存: {args.output} ({len(rankings)} 问题)")
    return rankings


# ================================================================
# 主函数
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='DeepMath 训练+排序')
    parser.add_argument('--mode', choices=['train', 'rank', 'both'], default='both')
    # 共享参数
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dim', type=int, default=1024, help='CNN输出维度')
    parser.add_argument('--max_len', type=int, default=2048, help='最大字符长度')
    parser.add_argument('--seed', type=int, default=24)
    # 训练参数
    parser.add_argument('--data_dir', default=f'{BASE}/LAH_TWGNN/dataset_D0')
    parser.add_argument('--model_save', default=f'{BASE}/DeepMath/models/deepmath_D0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--init_model', default=None)
    # 排序参数
    parser.add_argument('--model', default=None, help='模型路径(rank模式)')
    parser.add_argument('--statements_dir', default=f'{BASE}/LAH_TWGNN/statements')
    parser.add_argument('--chainy_dir', default=f'{BASE}/MPTP2078-master/chainy')
    parser.add_argument('--test_problems', default=f'{BASE}/LAH_TWGNN/test_problems_provable.json')
    parser.add_argument('--output', default=f'{BASE}/DeepMath/rankings_deepmath_1024.json')
    parser.add_argument('--top_k', type=int, default=1024)

    args = parser.parse_args()
    os.makedirs(args.model_save, exist_ok=True)
    logger = setup_logger(os.path.join(args.model_save, "train.log") if args.mode != 'rank' else None)

    if args.mode in ('train', 'both'):
        logger.info("=" * 50)
        logger.info("DeepMath 训练")
        vocab = train(args, logger)

    if args.mode in ('rank', 'both'):
        if args.model is None:
            args.model = os.path.join(args.model_save, "best.pt")
        logger.info("=" * 50)
        logger.info("DeepMath 排序")
        rank(args, logger, vocab=vocab if args.mode == 'both' else None)


if __name__ == '__main__':
    main()
