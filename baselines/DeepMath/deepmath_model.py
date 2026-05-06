#!/usr/bin/env python3
"""
DeepMath 字符级CNN前提选择模型 (PyTorch重写)
参考: Alemi et al., "DeepMath - Deep Sequence Models for Premise Selection", NeurIPS 2016

架构: 2-layer char-CNN (论文最佳单阶段模型)
  - 公式 → 字符序列 → one-hot → Conv1d(128,k5,s2) → Conv1d(1024,k5,s1) → GlobalMaxPool → 1024d
  - 猜想编码器和前提编码器不共享权重
  - 分类器: concat(2048d) → Linear(1024) → ReLU → Linear(2)
"""

import os, json, glob, pickle, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset


# ================================================================
# 字符词表
# ================================================================

def load_statements(statements_path):
    """加载statements文件 → {name: formula_text}"""
    stmts = {}
    if os.path.isdir(statements_path):
        for f in glob.glob(os.path.join(statements_path, "*")):
            name = os.path.basename(f)
            with open(f) as fh:
                stmts[name] = fh.read().strip()
    else:
        # 单文件格式: 每行一个 fof(name, type, formula).
        with open(statements_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('fof('):
                    idx = line.find('(') + 1
                    name = line[idx:line.find(',', idx)].strip()
                    stmts[name] = line
    return stmts


def build_char_vocab(statements_path):
    """从公式文件构建字符词表"""
    stmts = load_statements(statements_path)
    chars = set()
    for text in stmts.values():
        chars.update(text)
    vocab = {c: i + 1 for i, c in enumerate(sorted(chars))}  # 0=padding
    vocab['<PAD>'] = 0
    return vocab


def encode_formula(text, vocab, max_len=2048):
    """将公式文本编码为字符ID序列"""
    ids = [vocab.get(c, 0) for c in text[:max_len]]
    # padding
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


# ================================================================
# 数据集
# ================================================================

class DeepMathDataset(Dataset):
    """D0格式训练数据集"""

    def __init__(self, train_json, statements_path, vocab, max_len=2048):
        self.vocab = vocab
        self.max_len = max_len
        self.samples = []

        # 预加载所有公式
        self.formula_cache = load_statements(statements_path)

        # 加载训练对
        with open(train_json) as f:
            for line in f:
                conj, prem, label = json.loads(line.strip())
                self.samples.append((conj, prem, label))

    def _load_formula(self, name):
        return self.formula_cache.get(name, name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        conj, prem, label = self.samples[idx]
        conj_text = self._load_formula(conj)
        prem_text = self._load_formula(prem)
        conj_ids = torch.tensor(encode_formula(conj_text, self.vocab, self.max_len), dtype=torch.long)
        prem_ids = torch.tensor(encode_formula(prem_text, self.vocab, self.max_len), dtype=torch.long)
        return conj_ids, prem_ids, torch.tensor(label, dtype=torch.long)


# ================================================================
# 模型
# ================================================================

class CharCNNEncoder(nn.Module):
    """2-layer 字符级CNN编码器 (DeepMath最佳架构)"""

    def __init__(self, vocab_size, embed_dim=80, hidden_dim=128, out_dim=1024):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Conv1d: (batch, channels, length)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(hidden_dim, out_dim, kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        """x: (batch, seq_len) -> (batch, out_dim)"""
        x = self.embedding(x)          # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)         # (batch, embed_dim, seq_len)
        x = F.relu(self.conv1(x))      # (batch, hidden_dim, seq_len/2)
        x = F.relu(self.conv2(x))      # (batch, out_dim, seq_len/2)
        x = x.max(dim=2).values        # global max pooling → (batch, out_dim)
        return x


class DeepMathModel(nn.Module):
    """DeepMath前提选择模型"""

    def __init__(self, vocab_size, embed_dim=80, out_dim=1024):
        super().__init__()
        self.conj_encoder = CharCNNEncoder(vocab_size, embed_dim, 128, out_dim)
        self.prem_encoder = CharCNNEncoder(vocab_size, embed_dim, 128, out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 2)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, conj_ids, prem_ids, labels=None):
        """
        conj_ids: (batch, seq_len)
        prem_ids: (batch, seq_len)
        labels: (batch,) optional
        Returns: loss (if labels), logits
        """
        conj_emb = self.conj_encoder(conj_ids)   # (batch, 1024)
        prem_emb = self.prem_encoder(prem_ids)    # (batch, 1024)
        combined = torch.cat([conj_emb, prem_emb], dim=1)  # (batch, 2048)
        logits = self.classifier(combined)        # (batch, 2)

        if labels is not None:
            loss = self.criterion(logits, labels)
            pred = logits.argmax(dim=1)
            return loss, pred
        return logits

    def score(self, conj_ids, prem_ids):
        """返回正类概率分数 (用于排序)"""
        logits = self.forward(conj_ids, prem_ids)
        probs = F.softmax(logits, dim=1)
        return probs[:, 1]  # 正类概率
