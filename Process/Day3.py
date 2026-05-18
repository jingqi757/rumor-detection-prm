"""
Day3: ORM标注 + NLI评分 + 噪声分析
======================================
Pipeline:
  1. 加载Day2输出 (best_of_n_results.jsonl)
  2. ORM标注：基于最终verdict判断每条chain是否"正确"
  3. NLI评分：用NLI模型对每个步骤的claim↔evidence打分
  4. 一致率分析：ORM标注 vs NLI评分
  5. 噪声分布分析 + 可视化

依赖：
  pip install transformers torch datasets matplotlib seaborn tqdm
"""

import json
import os
import re
import random
import argparse
from pathlib import Path
from typing import Optional
import numpy as np

# ─────────────────────────────────────────────
# 0. 配置
# ─────────────────────────────────────────────
RESULTS_PATH = Path(__file__).parent.parent / "data" / "day2_results.jsonl"
OUTPUT_DIR   = Path(__file__).parent.parent / "data" / "day3_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# AVeriTeC verdict标准化映射
VERDICT_NORM = {
    # Supported
    "supported": "Supported",
    "true": "Supported",
    "support": "Supported",
    # Refuted
    "refuted": "Refuted",
    "false": "Refuted",
    "refute": "Refuted",
    # NEI
    "not enough evidence": "NEI",
    "nei": "NEI",
    "unknown": "NEI",
    "insufficient": "NEI",
    # Conflicting
    "conflicting evidence/cherrypicking": "Conflicting",
    "conflicting": "Conflicting",
    "cherrypicking": "Conflicting",
    "cherry-picking": "Conflicting",
}

VERDICT_ORDER = ["Supported", "Refuted", "NEI", "Conflicting"]

NLI_LABEL_MAP = {
    "entailment":    1.0,
    "neutral":       0.5,
    "contradiction": 0.0,
}


# ─────────────────────────────────────────────
# 1. 工具函数
# ─────────────────────────────────────────────

def normalize_verdict(raw: str) -> str:
    """把模型输出的verdict字符串标准化到四类。"""
    if not raw:
        return "NEI"
    key = raw.strip().lower()
    # 精确匹配
    if key in VERDICT_NORM:
        return VERDICT_NORM[key]
    # 子串匹配
    for k, v in VERDICT_NORM.items():
        if k in key:
            return v
    return "NEI"


def parse_verdict_from_text(text: str) -> str:
    """从模型生成文本中抽取verdict。支持多种格式。"""
    if not text:
        return ""
    patterns = [
        r"(?:final\s+)?verdict[:\s]+([^\n\.]+)",
        r"claim\s+is\s+(supported|refuted|true|false|not enough evidence)",
        r"\*\*verdict\*\*[:\s]+([^\n\*]+)",
        r"conclusion[:\s]+([^\n\.]+)",
        r"answer[:\s]+([^\n\.]+)",
        r"(True|False)\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""


def extract_steps(chain_text: str) -> list[dict]:
    """
    从chain文本中解析步骤。
    支持格式：
      Step N: <question>  Evidence: <evidence>
      Q: <question>  A: <answer>
      数字编号段落
    返回 [{"step": int, "claim": str, "evidence": str}]
    """
    steps = []
    # 方案A：Step N: ... Evidence: ...
    pattern_a = re.finditer(
        r"Step\s+(\d+)[:\.]?\s*(.*?)(?=Step\s+\d+|$)",
        chain_text, re.IGNORECASE | re.DOTALL
    )
    for m in pattern_a:
        step_n = int(m.group(1))
        content = m.group(2).strip()
        # 拆 claim / evidence
        ev_match = re.search(r"[Ee]vidence[:\s]+(.*)", content, re.DOTALL)
        claim = content[:ev_match.start()].strip() if ev_match else content
        evidence = ev_match.group(1).strip() if ev_match else ""
        steps.append({"step": step_n, "claim": claim[:300], "evidence": evidence[:300]})

    if steps:
        return steps

    # 方案B：Q/A对（改进正则：匹配到下一个Q或Verdict或结尾）
    pattern_b = re.finditer(
        r"Q(?:uestion)?\s*\d*[:\.]?\s*(.*?)\n\s*A(?:nswer)?\s*\d*[:\.]?\s*(.*?)(?=\nQ(?:uestion)?|\nVerdict|\Z)",
        chain_text, re.IGNORECASE | re.DOTALL
    )
    idx = 1
    for m in pattern_b:
        steps.append({
            "step": idx,
            "claim": m.group(1).strip()[:300],
            "evidence": m.group(2).strip()[:300],
        })
        idx += 1

    if steps:
        return steps

    # 方案C：按行分块（fallback）— 排除Verdict行
    lines = [
        l.strip() for l in chain_text.split("\n")
        if l.strip() and not re.match(r"^(verdict|conclusion|answer)\s*:", l.strip(), re.IGNORECASE)
    ]
    for i, line in enumerate(lines[:8]):
        steps.append({
            "step": i + 1,
            "claim": line[:300],
            "evidence": "",
        })
    return steps


# ─────────────────────────────────────────────
# 2. 生成合成Day2数据（若无真实文件）
# ─────────────────────────────────────────────

def generate_synthetic_day2(n=50, chains_per=5, seed=42) -> list[dict]:
    """生成与Day2输出格式一致的合成数据，用于快速测试。"""
    random.seed(seed)
    np.random.seed(seed)
    verdicts = ["Supported", "Refuted", "NEI", "Conflicting"]
    weights  = [0.35, 0.35, 0.20, 0.10]

    samples = []
    for i in range(n):
        gt = random.choices(verdicts, weights=weights)[0]
        chains = []
        for c in range(chains_per):
            # 正确概率约0.5（模拟Day2的2.52/5）
            correct = random.random() < 0.504
            pred = gt if correct else random.choice([v for v in verdicts if v != gt])
            # 构造假步骤文本
            n_steps = random.randint(2, 5)
            step_texts = []
            for s in range(1, n_steps + 1):
                step_texts.append(
                    f"Step {s}: Is the claim about {['dates', 'numbers', 'persons', 'events'][s%4]}? "
                    f"Evidence: According to source {s}, the information {'confirms' if correct else 'contradicts'} the claim."
                )
            chain_text = "\n".join(step_texts) + f"\nVerdict: {pred}"
            chains.append({
                "chain_id": c,
                "text": chain_text,
                "verdict_raw": pred,
                "verdict_norm": normalize_verdict(pred),
                "parse_failed": False,
            })

        samples.append({
            "sample_id": i,
            "claim": f"Synthetic claim #{i}: Some verifiable statement about current events.",
            "gt_verdict": gt,
            "chains": chains,
        })
    return samples


# ─────────────────────────────────────────────
# 3. 加载Day2数据
# ─────────────────────────────────────────────

def load_day2_data() -> list[dict]:
    if RESULTS_PATH.exists():
        print(f"[+] 加载Day2输出: {RESULTS_PATH}")
        samples = []
        with open(RESULTS_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        # 统一格式：补全 verdict_norm 字段
        for s in samples:
            for ch in s.get("chains", []):
                if "verdict_norm" not in ch:
                    raw = ch.get("verdict_raw", ch.get("verdict", ""))
                    ch["verdict_norm"] = normalize_verdict(raw)
        print(f"    已加载 {len(samples)} 条样本")
        return samples
    else:
        print(f"[!] 未找到 {RESULTS_PATH}，使用合成数据（50条）")
        return generate_synthetic_day2(n=50, chains_per=5)


# ─────────────────────────────────────────────
# 4. ORM标注
# ─────────────────────────────────────────────

def orm_annotate(samples: list[dict]) -> list[dict]:
    """
    ORM标注：判断每条chain的每个step是否"对"。
    策略：
      - chain级别：pred_verdict == gt_verdict → chain_correct=True
      - step级别：从chain结论反向推导
          * chain正确 → 所有step标注为1（正向贡献）
          * chain错误 → 找第一个"偏轨"步骤，该步之后标注为0
          * 简化版：正确chain全1，错误chain后半段0
    这是"ORM反向推导"的合理近似，后续Day4引入置信度探针精细化。
    """
    print("[+] ORM标注中...")
    annotated = []
    orm_stats = {"correct_chains": 0, "wrong_chains": 0, "parse_fail": 0}

    for s in samples:
        gt = s.get("gt_verdict", s.get("ground_truth_verdict", ""))
        gt_norm = normalize_verdict(gt)
        chains_out = []

        for ch in s.get("chains", []):
            if ch.get("parse_failed", False):
                orm_stats["parse_fail"] += 1
                ch["orm_label"] = None
                ch["step_orm_labels"] = []
                chains_out.append(ch)
                continue

            pred_norm = ch.get("verdict_norm", "")
            chain_correct = (pred_norm == gt_norm) and (gt_norm != "NEI")

            # 解析步骤
            steps = extract_steps(ch.get("text", ""))
            n_steps = len(steps)

            if chain_correct:
                orm_stats["correct_chains"] += 1
                step_labels = [1] * n_steps
            else:
                orm_stats["wrong_chains"] += 1
                # 假设前半步骤合理，后半步骤出错（简化近似）
                pivot = max(1, n_steps // 2)
                step_labels = [1] * pivot + [0] * (n_steps - pivot)

            for step, label in zip(steps, step_labels):
                step["orm_label"] = label

            ch["orm_label"]       = 1 if chain_correct else 0
            ch["step_orm_labels"] = step_labels
            ch["steps_parsed"]    = steps
            chains_out.append(ch)

        s["chains"] = chains_out
        annotated.append(s)

    total_chains = orm_stats["correct_chains"] + orm_stats["wrong_chains"] + orm_stats["parse_fail"]
    print(f"    总chain数:    {total_chains}")
    print(f"    正确chain:    {orm_stats['correct_chains']} ({100*orm_stats['correct_chains']/max(1,total_chains):.1f}%)")
    print(f"    错误chain:    {orm_stats['wrong_chains']} ({100*orm_stats['wrong_chains']/max(1,total_chains):.1f}%)")
    print(f"    解析失败:     {orm_stats['parse_fail']}")
    return annotated


# ─────────────────────────────────────────────
# 5. NLI评分
# ─────────────────────────────────────────────

def load_nli_model(model_name: str = "cross-encoder/nli-MiniLM2-L6-H768"):
    """加载轻量级NLI模型（适合8GB GPU）。"""
    try:
        from transformers import pipeline
        print(f"[+] 加载NLI模型: {model_name}")
        nli = pipeline(
            "text-classification",
            model=model_name,
            device=0 if _cuda_available() else -1,
            batch_size=32,
        )
        return nli
    except Exception as e:
        print(f"[!] NLI模型加载失败: {e}")
        print("    将使用随机基线（仅用于测试流程）")
        return None


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def nli_score_steps(samples: list[dict], nli_model, batch_size: int = 64) -> list[dict]:
    """
    对每个步骤计算NLI分数：premise=evidence，hypothesis=claim。
    分数 ∈ [0, 1]，1=entailment，0.5=neutral，0=contradiction。
    """
    print("[+] NLI评分中...")
    # 收集所有 (premise, hypothesis) 对
    pairs = []   # (sample_idx, chain_idx, step_idx)
    texts = []

    for si, s in enumerate(samples):
        for ci, ch in enumerate(s.get("chains", [])):
            for sti, step in enumerate(ch.get("steps_parsed", [])):
                claim    = ch.get("claim", s.get("claim", ""))
                evidence = step.get("evidence", "") or step.get("claim", "")
                if not evidence:
                    evidence = "No evidence provided."
                pairs.append((si, ci, sti))
                texts.append((evidence[:512], claim[:256]))

    if not texts:
        print("    无可评分步骤")
        return samples

    # 批量推理
    scores_map: dict[tuple, float] = {}

    if nli_model is None:
        # 随机基线（仅测试流程）
        np.random.seed(0)
        raw_scores = np.random.dirichlet([2, 1, 1], size=len(texts))
        # entailment, neutral, contradiction → 加权分
        for idx, (prob_e, prob_n, prob_c) in zip(pairs, raw_scores):
            scores_map[idx] = float(prob_e * 1.0 + prob_n * 0.5 + prob_c * 0.0)
    else:
        from tqdm import tqdm
        inputs = [f"{p} [SEP] {h}" for p, h in texts]
        results = []
        for i in tqdm(range(0, len(inputs), batch_size), desc="  NLI batch"):
            batch = inputs[i:i + batch_size]
            results.extend(nli_model(batch))
        for idx, res in zip(pairs, results):
            label = res["label"].lower()
            score = NLI_LABEL_MAP.get(label, 0.5)
            scores_map[idx] = score

    # 写回
    for (si, ci, sti), score in scores_map.items():
        samples[si]["chains"][ci]["steps_parsed"][sti]["nli_score"] = score

    # 汇总：每条chain的平均NLI分
    for s in samples:
        for ch in s.get("chains", []):
            step_scores = [
                st.get("nli_score", 0.5)
                for st in ch.get("steps_parsed", [])
            ]
            ch["chain_nli_mean"] = float(np.mean(step_scores)) if step_scores else 0.5

    total_steps = len(texts)
    print(f"    已评分步骤数: {total_steps}")
    return samples


# ─────────────────────────────────────────────
# 6. 一致率分析
# ─────────────────────────────────────────────

def agreement_analysis(samples: list[dict]) -> dict:
    """
    计算ORM标注 vs NLI评分的一致率。
    - ORM步骤标签: 0/1
    - NLI步骤分数: 连续值，用阈值0.6二值化
    返回详细统计字典。
    """
    print("[+] 计算ORM↔NLI一致率...")
    NLI_THRESH = 0.6

    orm_labels, nli_binary = [], []
    chain_level_orm, chain_level_nli = [], []

    for s in samples:
        for ch in s.get("chains", []):
            if ch.get("orm_label") is None:
                continue
            chain_level_orm.append(ch["orm_label"])
            chain_level_nli.append(1 if ch.get("chain_nli_mean", 0.5) >= NLI_THRESH else 0)

            for step in ch.get("steps_parsed", []):
                ol = step.get("orm_label")
                ns = step.get("nli_score")
                if ol is not None and ns is not None:
                    orm_labels.append(ol)
                    nli_binary.append(1 if ns >= NLI_THRESH else 0)

    orm_labels  = np.array(orm_labels)
    nli_binary  = np.array(nli_binary)
    chain_orm   = np.array(chain_level_orm)
    chain_nli   = np.array(chain_level_nli)

    def agreement(a, b):
        if len(a) == 0:
            return {"agreement": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
        tp = int(((a == 1) & (b == 1)).sum())
        tn = int(((a == 0) & (b == 0)).sum())
        fp = int(((a == 0) & (b == 1)).sum())
        fn = int(((a == 1) & (b == 0)).sum())
        return {
            "agreement": (tp + tn) / len(a),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": tp / max(1, tp + fp),
            "recall":    tp / max(1, tp + fn),
            "n": len(a),
        }

    step_agr  = agreement(orm_labels, nli_binary)
    chain_agr = agreement(chain_orm, chain_nli)

    # Cohen's kappa（步骤级）
    def kappa(a, b):
        if len(a) == 0:
            return 0.0
        po = (a == b).mean()
        p1 = a.mean() * b.mean()
        p0 = (1 - a.mean()) * (1 - b.mean())
        pe = p1 + p0
        return (po - pe) / max(1e-9, 1 - pe)

    step_agr["kappa"]  = float(kappa(orm_labels, nli_binary))
    chain_agr["kappa"] = float(kappa(chain_orm, chain_nli))

    print(f"    步骤级一致率:  {step_agr['agreement']:.3f}  (κ={step_agr['kappa']:.3f}, n={step_agr['n']})")
    print(f"    Chain级一致率: {chain_agr['agreement']:.3f}  (κ={chain_agr['kappa']:.3f}, n={chain_agr['n']})")
    print(f"    步骤级 TP={step_agr['tp']} TN={step_agr['tn']} FP={step_agr['fp']} FN={step_agr['fn']}")

    return {
        "step_level":  step_agr,
        "chain_level": chain_agr,
        "nli_threshold": NLI_THRESH,
    }


# ─────────────────────────────────────────────
# 7. 噪声分布分析
# ─────────────────────────────────────────────

def noise_analysis(samples: list[dict]) -> dict:
    """
    分析噪声分布：
    - ORM标注为1但NLI分低的步骤（可能是推理幻觉）
    - ORM标注为0但NLI分高的步骤（可能是步骤分解问题）
    - 按verdict类别分析noise rate
    """
    print("[+] 噪声分布分析...")
    NLI_THRESH = 0.6

    # 噪声类型
    noise_records = []
    verdict_noise: dict[str, list] = {v: [] for v in VERDICT_ORDER}

    for s in samples:
        gt = normalize_verdict(s.get("gt_verdict", s.get("ground_truth_verdict", "")))
        for ch in s.get("chains", []):
            if ch.get("orm_label") is None:
                continue
            for step in ch.get("steps_parsed", []):
                ol = step.get("orm_label")
                ns = step.get("nli_score")
                if ol is None or ns is None:
                    continue
                nli_b = 1 if ns >= NLI_THRESH else 0
                noise = (ol != nli_b)
                noise_type = None
                if ol == 1 and nli_b == 0:
                    noise_type = "hallucination"    # ORM说好，NLI说差
                elif ol == 0 and nli_b == 1:
                    noise_type = "step_decomp_err"  # ORM说差，NLI说好

                noise_records.append({
                    "is_noise":   noise,
                    "noise_type": noise_type,
                    "orm":        ol,
                    "nli":        ns,
                    "gt_verdict": gt,
                })
                if gt in verdict_noise:
                    verdict_noise[gt].append(int(noise))

    total   = len(noise_records)
    n_noise = sum(r["is_noise"] for r in noise_records)
    n_hall  = sum(1 for r in noise_records if r["noise_type"] == "hallucination")
    n_step  = sum(1 for r in noise_records if r["noise_type"] == "step_decomp_err")

    verdict_noise_rate = {
        k: float(np.mean(v)) if v else 0.0
        for k, v in verdict_noise.items()
    }

    print(f"    总步骤数:         {total}")
    print(f"    噪声步骤:         {n_noise} ({100*n_noise/max(1,total):.1f}%)")
    print(f"      └ 幻觉型:       {n_hall} (ORM=1, NLI<θ)")
    print(f"      └ 步骤分解误差: {n_step} (ORM=0, NLI≥θ)")
    print(f"    按verdict噪声率:")
    for v, r in verdict_noise_rate.items():
        count = len(verdict_noise.get(v, []))
        print(f"      {v:<30} {r:.3f}  (n={count})")

    return {
        "total_steps": total,
        "noise_count": n_noise,
        "noise_rate":  n_noise / max(1, total),
        "hallucination_count":    n_hall,
        "step_decomp_err_count":  n_step,
        "verdict_noise_rate":     verdict_noise_rate,
        "noise_records":          noise_records,
    }


# ─────────────────────────────────────────────
# 8. 可视化
# ─────────────────────────────────────────────

def plot_results(agr: dict, noise: dict, samples: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("[!] matplotlib不可用，跳过可视化")
        return

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("Day3: ORM Annotation + NLI Scoring Analysis", fontsize=14, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors = {"Supported": "#4CAF50", "Refuted": "#F44336", "NEI": "#FF9800", "Conflicting": "#9C27B0"}

    # ── 图1：步骤级ORM vs NLI混淆矩阵 ──
    ax1 = fig.add_subplot(gs[0, 0])
    agr_s = agr["step_level"]
    cm = np.array([[agr_s["tn"], agr_s["fp"]],
                   [agr_s["fn"], agr_s["tp"]]])
    im = ax1.imshow(cm, cmap="Blues", aspect="auto")
    ax1.set_xticks([0, 1]); ax1.set_yticks([0, 1])
    ax1.set_xticklabels(["NLI=0", "NLI=1"])
    ax1.set_yticklabels(["ORM=0", "ORM=1"])
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() * 0.5 else "black", fontsize=12)
    ax1.set_title(f"步骤级混淆矩阵\n一致率={agr_s['agreement']:.3f} κ={agr_s['kappa']:.3f}", fontsize=10)
    fig.colorbar(im, ax=ax1, fraction=0.046)

    # ── 图2：NLI分数分布（按ORM标注） ──
    ax2 = fig.add_subplot(gs[0, 1])
    nli_pos, nli_neg = [], []
    for s in samples:
        for ch in s.get("chains", []):
            for step in ch.get("steps_parsed", []):
                ns = step.get("nli_score")
                ol = step.get("orm_label")
                if ns is None or ol is None:
                    continue
                (nli_pos if ol == 1 else nli_neg).append(ns)
    bins = np.linspace(0, 1, 21)
    ax2.hist(nli_pos, bins=bins, alpha=0.6, label=f"ORM=1 (n={len(nli_pos)})", color="#2196F3", density=True)
    ax2.hist(nli_neg, bins=bins, alpha=0.6, label=f"ORM=0 (n={len(nli_neg)})", color="#FF5722", density=True)
    ax2.axvline(agr["nli_threshold"], color="black", linestyle="--", linewidth=1.5, label=f"θ={agr['nli_threshold']}")
    ax2.set_xlabel("NLI分数"); ax2.set_ylabel("密度"); ax2.legend(fontsize=8)
    ax2.set_title("NLI分数分布（按ORM标注）", fontsize=10)

    # ── 图3：按verdict的噪声率 ──
    ax3 = fig.add_subplot(gs[0, 2])
    vkeys = [v for v in VERDICT_ORDER if v in noise["verdict_noise_rate"]]
    vvals = [noise["verdict_noise_rate"][v] for v in vkeys]
    vcols = [colors.get(v, "#607D8B") for v in vkeys]
    bars = ax3.bar(vkeys, vvals, color=vcols, edgecolor="white", linewidth=1.5)
    ax3.set_ylim(0, 1); ax3.set_ylabel("噪声率")
    ax3.set_title("按Verdict类别噪声率", fontsize=10)
    for bar, val in zip(bars, vvals):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.2f}", ha="center", fontsize=9)
    ax3.tick_params(axis="x", labelsize=8)

    # ── 图4：噪声类型饼图 ──
    ax4 = fig.add_subplot(gs[1, 0])
    clean   = noise["total_steps"] - noise["noise_count"]
    hall    = noise["hallucination_count"]
    step_e  = noise["step_decomp_err_count"]
    if clean + hall + step_e > 0:
        ax4.pie([clean, hall, step_e],
                labels=[f"Clean\n({clean})", f"Hallucination\n({hall})", f"StepDecomp\n({step_e})"],
                colors=["#4CAF50", "#F44336", "#FF9800"],
                autopct="%1.1f%%", startangle=90, textprops={"fontsize": 8})
    ax4.set_title(f"噪声类型分布\n总噪声率={noise['noise_rate']:.3f}", fontsize=10)

    # ── 图5：每个样本的正确chain数分布 ──
    ax5 = fig.add_subplot(gs[1, 1])
    correct_counts = []
    for s in samples:
        n_correct = sum(1 for ch in s.get("chains", []) if ch.get("orm_label") == 1)
        correct_counts.append(n_correct)
    max_chains = max(correct_counts) if correct_counts else 5
    ax5.hist(correct_counts, bins=range(0, max_chains + 2), align="left",
             color="#3F51B5", edgecolor="white", rwidth=0.8)
    ax5.set_xlabel("正确chain数"); ax5.set_ylabel("样本数")
    mean_c = np.mean(correct_counts) if correct_counts else 0
    ax5.axvline(mean_c, color="red", linestyle="--", label=f"均值={mean_c:.2f}")
    ax5.legend(fontsize=9)
    ax5.set_title("每样本正确Chain数分布", fontsize=10)

    # ── 图6：ORM与NLI联合热力图（chain级） ──
    ax6 = fig.add_subplot(gs[1, 2])
    chain_orm_scores, chain_nli_scores = [], []
    for s in samples:
        for ch in s.get("chains", []):
            ol = ch.get("orm_label")
            ns = ch.get("chain_nli_mean")
            if ol is not None and ns is not None:
                chain_orm_scores.append(ol + np.random.uniform(-0.05, 0.05))
                chain_nli_scores.append(ns)
    if chain_orm_scores:
        ax6.scatter(chain_orm_scores, chain_nli_scores,
                    alpha=0.4, s=20, c="#009688")
        ax6.axhline(agr["nli_threshold"], color="red", linestyle="--", linewidth=1)
        ax6.set_xlabel("ORM标注（0=错，1=对，加噪显示）")
        ax6.set_ylabel("Chain平均NLI分")
        ax6.set_title("Chain级 ORM vs NLI散点图", fontsize=10)

    out_path = OUTPUT_DIR / "day3_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"[+] 可视化已保存: {out_path}")
    plt.close()


# ─────────────────────────────────────────────
# 9. 保存输出
# ─────────────────────────────────────────────

def save_outputs(samples: list[dict], agr: dict, noise: dict):
    # 9a. 带标注的完整数据
    annotated_path = OUTPUT_DIR / "annotated_samples.jsonl"
    with open(annotated_path, "w") as f:
        for s in samples:
            # 去掉 noise_records（太大）
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"[+] 带标注数据已保存: {annotated_path}")

    # 9b. 统计报告
    report = {
        "agreement": agr,
        "noise": {k: v for k, v in noise.items() if k != "noise_records"},
    }
    report_path = OUTPUT_DIR / "day3_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[+] 统计报告已保存: {report_path}")

    # 9c. 高噪声步骤样本（用于Day4置信度探针分析）
    noisy_steps = []
    for s in samples:
        for ch in s.get("chains", []):
            for step in ch.get("steps_parsed", []):
                ol = step.get("orm_label")
                ns = step.get("nli_score")
                if ol is None or ns is None:
                    continue
                nli_b = 1 if ns >= agr["nli_threshold"] else 0
                if ol != nli_b:  # 噪声步骤
                    noisy_steps.append({
                        "sample_id":  s.get("sample_id"),
                        "claim":      s.get("claim", "")[:200],
                        "gt_verdict": s.get("gt_verdict", ""),
                        "step":       step,
                        "chain_nli":  ch.get("chain_nli_mean"),
                        "noise_type": "hallucination" if ol == 1 and nli_b == 0 else "step_decomp_err",
                    })
    noisy_path = OUTPUT_DIR / "noisy_steps_for_day4.jsonl"
    with open(noisy_path, "w") as f:
        for ns in noisy_steps:
            f.write(json.dumps(ns, ensure_ascii=False) + "\n")
    print(f"[+] 噪声步骤已保存（Day4用）: {noisy_path} ({len(noisy_steps)}条)")


# ─────────────────────────────────────────────
# 10. 主流程
# ─────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print("Day3: ORM标注 + NLI评分 + 噪声分析")
    print("=" * 60)

    # 加载
    samples = load_day2_data()

    # ORM标注
    samples = orm_annotate(samples)

    # NLI评分
    nli_model = None
    if not args.skip_nli:
        nli_model = load_nli_model(args.nli_model)
    samples = nli_score_steps(samples, nli_model, batch_size=args.batch_size)

    # 一致率
    agr = agreement_analysis(samples)

    # 噪声分析
    noise = noise_analysis(samples)

    # 可视化
    plot_results(agr, noise, samples)

    # 保存
    save_outputs(samples, agr, noise)

    # ── Day3小结 ──
    print()
    print("=" * 60)
    print("Day3 小结")
    print("=" * 60)
    step_a = agr["step_level"]
    print(f"  步骤级 ORM↔NLI 一致率: {step_a['agreement']:.3f}  (κ={step_a['kappa']:.3f})")
    print(f"  总噪声率:               {noise['noise_rate']:.3f}")
    print(f"    幻觉型 (ORM=1,NLI<θ): {noise['hallucination_count']}")
    print(f"    分解误差(ORM=0,NLI≥θ): {noise['step_decomp_err_count']}")
    print()

    agr_val = step_a['agreement']
    if agr_val >= 0.75:
        decision = "✅ 一致率高，ORM标注可靠，可直接用于PRM训练"
    elif agr_val >= 0.60:
        decision = "⚠️  一致率中等，建议Day4引入置信度探针精细化标注"
    else:
        decision = "❌ 一致率低，需重审步骤解析或NLI阈值，Day4重点排查"

    print(f"  Day4路线建议: {decision}")
    print()
    print(f"  输出文件目录: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Day3: ORM + NLI 分析")
    parser.add_argument("--skip-nli",   action="store_true",
                        help="跳过NLI推理（用随机基线，测试流程用）")
    parser.add_argument("--nli-model",  default="cross-encoder/nli-MiniLM2-L6-H768",
                        help="NLI模型名称（HuggingFace Hub）")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="NLI推理batch size")
    args = parser.parse_args()
    main(args)