#!/usr/bin/env python3
"""
计算前提选择评价指标:
1. Recall@K (前提召回率)
2. Average Proving K (平均证明K值)
3. S_pre (Premise Ranking Score)
"""

import json, os, sys, glob

BASE = os.environ.get("PROVER_BASE", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
BUSHY_DIR = os.environ.get("BUSHY_DIR", f"{BASE}/dataset/MPTP2078-master/bushy")

# ================================================================
# 加载Gold Standard前提 (bushy依赖)
# ================================================================
def load_bushy_deps(test_problems):
    """从bushy目录提取每个测试问题的gold standard前提"""
    deps = {}
    for prob in test_problems:
        # bushy文件名格式: theory__problem.p
        pattern = os.path.join(BUSHY_DIR, f"*__{prob}.p")
        matches = glob.glob(pattern)
        if not matches:
            continue

        bushy_file = matches[0]
        axioms = set()
        with open(bushy_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith('fof(') and ',axiom,' in line.replace(' ', ''):
                    # 提取公理名
                    idx = line.find('(') + 1
                    name = line[idx:line.find(',', idx)].strip()
                    if name != prob:  # 排除问题本身
                        axioms.add(name)
        if axioms:
            deps[prob] = axioms
    return deps


# ================================================================
# Recall@K
# ================================================================
def compute_recall_at_k(rankings, gold_deps, K_values=[32, 64, 128, 256, 512, 1024]):
    """计算各K值下的平均Recall"""
    results = {}
    for K in K_values:
        recalls = []
        for prob, gold in gold_deps.items():
            if prob not in rankings:
                continue
            selected = set(rankings[prob][:K])
            recall = len(selected & gold) / len(gold) if gold else 0
            recalls.append(recall)
        results[K] = sum(recalls) / len(recalls) if recalls else 0
    return results


# ================================================================
# Average Proving K
# ================================================================
def compute_avg_proving_k(cascade_merged):
    """已证明问题的平均证明K值"""
    ks = []
    for prob, res in cascade_merged.items():
        if res.get('proved'):
            k = res.get('proved_K')
            if k:
                ks.append(k)
    if not ks:
        return None, None, None
    avg_k = sum(ks) / len(ks)
    median_k = sorted(ks)[len(ks) // 2]
    return avg_k, median_k, len(ks)


# ================================================================
# S_pre (Premise Ranking Score)
# ================================================================
def compute_spre(rankings, gold_deps):
    """
    S_pre = 1/|Q| * sum_{q in Q} 1/|D_q| * sum_{d in D_q} (1 - rank(d)/|A|)
    其中 Q=测试问题集, D_q=gold前提集, rank(d)=d在排序中的位置, |A|=候选前提总数
    """
    scores = []
    for prob, gold in gold_deps.items():
        if prob not in rankings:
            continue
        ranked_list = rankings[prob]
        total_premises = len(ranked_list)
        rank_map = {p: i + 1 for i, p in enumerate(ranked_list)}

        prob_scores = []
        for d in gold:
            rank = rank_map.get(d, total_premises + 1)
            prob_scores.append(1.0 - rank / (total_premises + 1))

        if prob_scores:
            scores.append(sum(prob_scores) / len(prob_scores))

    return sum(scores) / len(scores) if scores else 0


# ================================================================
# 主函数
# ================================================================
def compute_all_metrics(name, ranking_path, cascade_path, gold_deps):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # 加载ranking
    rankings = json.load(open(ranking_path))

    # Recall@K
    recall = compute_recall_at_k(rankings, gold_deps)
    print(f"\n  Recall@K:")
    for k, v in recall.items():
        print(f"    K={k:<5}: {v:.4f} ({v*100:.2f}%)")

    # S_pre
    spre = compute_spre(rankings, gold_deps)
    print(f"\n  S_pre: {spre:.4f}")

    # Average Proving K
    if cascade_path and os.path.exists(cascade_path):
        cascade = json.load(open(cascade_path))
        avg_k, median_k, n_proved = compute_avg_proving_k(cascade)
        if avg_k:
            print(f"\n  Average Proving K: {avg_k:.1f} (中位数: {median_k}, 已证明: {n_proved})")
    else:
        avg_k, median_k, n_proved = None, None, None
        print(f"\n  Average Proving K: 无cascade数据")

    return {
        "recall": recall,
        "spre": spre,
        "avg_proving_k": avg_k,
        "median_proving_k": median_k,
        "n_proved": n_proved
    }


def main():
    # 加载测试问题
    tp_file = f"{BASE}/LAH_TWGNN/test_problems_provable.json"
    test_problems = json.load(open(tp_file))
    print(f"测试问题: {len(test_problems)} 个")

    # 加载gold standard
    gold_deps = load_bushy_deps(test_problems)
    print(f"有bushy依赖的问题: {len(gold_deps)} 个")
    avg_gold = sum(len(v) for v in gold_deps.values()) / len(gold_deps)
    print(f"平均gold前提数: {avg_gold:.1f}")

    # 各方法配置
    methods = {
        "LCGS (d256_l1, λ=0.3)": {
            "ranking": f"{BASE}/LCGS_TWGNN/models/lcgs_D0_d256_l1_lam0_3/rankings_D0_d256_l1_lam0_3.json",
            "cascade": f"{BASE}/LCGS_TWGNN/eval_cascade_D0_d256_l1_lam0_3/cascade_merged.json",
        },
        "LAH (d128_l1)": {
            "ranking": f"{BASE}/LAH_TWGNN/models/lah_D0_d128_l1/rankings_1024.json",
            "cascade": f"{BASE}/LAH_TWGNN/eval_cascade_D0_d128_l1/cascade_merged.json",
        },
        "CL (d512_l1)": {
            "ranking": f"{BASE}/CL-TWGNN/models/cl_D0_d512_l1/rankings_1024.json",
            "cascade": f"{BASE}/CL-TWGNN/eval_cascade_D0_d512_l1/cascade_merged.json",
        },
        "ATPBoost": {
            "ranking": f"{BASE}/ATPboost-master/data/MPTP2078/rankings_atpboost_1024.json",
            "cascade": f"{BASE}/ATPboost-master/data/MPTP2078/eval_cascade_atpboost/cascade_merged.json",
        },
        "Kernel (TF-IDF)": {
            "ranking": f"{BASE}/TWGNN/results_kernel_rerun/rankings_kernel_1024.json",
            "cascade": f"{BASE}/TWGNN/results_kernel_rerun/eval_cascade_kernel/cascade_merged.json",
        },
        "Levenshtein": {
            "ranking": f"{BASE}/TWGNN/results_levenshtein/rankings_levenshtein_1024.json",
            "cascade": None,
        },
    }

    all_results = {}
    for name, cfg in methods.items():
        if not os.path.exists(cfg["ranking"]):
            print(f"\n  {name}: ranking文件不存在, 跳过")
            continue
        all_results[name] = compute_all_metrics(
            name, cfg["ranking"], cfg.get("cascade"), gold_deps)

    # 汇总表
    print(f"\n\n{'='*80}")
    print(f"  汇总")
    print(f"{'='*80}")

    print(f"\n{'方法':<25} {'R@32':>7} {'R@64':>7} {'R@128':>7} {'R@256':>7} {'R@512':>7} {'R@1024':>7} {'S_pre':>7} {'AvgK':>7}")
    print("-" * 95)
    for name, r in all_results.items():
        rec = r["recall"]
        avgk = f"{r['avg_proving_k']:.1f}" if r['avg_proving_k'] else "-"
        print(f"{name:<25} {rec[32]:>7.4f} {rec[64]:>7.4f} {rec[128]:>7.4f} {rec[256]:>7.4f} {rec[512]:>7.4f} {rec[1024]:>7.4f} {r['spre']:>7.4f} {avgk:>7}")

    # 保存JSON
    output = {}
    for name, r in all_results.items():
        output[name] = {
            "recall_at_k": {str(k): round(v, 4) for k, v in r["recall"].items()},
            "spre": round(r["spre"], 4),
            "avg_proving_k": round(r["avg_proving_k"], 1) if r["avg_proving_k"] else None,
            "median_proving_k": r["median_proving_k"],
            "n_proved": r["n_proved"]
        }

    out_path = f"{BASE}/evaluation_metrics.json"
    json.dump(output, open(out_path, 'w'), indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {out_path}")


if __name__ == '__main__':
    main()
